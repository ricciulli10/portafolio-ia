"""The agent loop: budgets, approvals, recoverable errors and a full trace."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .llm import LLM, ToolCall
from .tools import Tool, ToolError

MAX_RESULT_CHARS = 20_000
MAX_IDENTICAL_CALLS = 3


def deny_all(_: ToolCall) -> bool:
    """Default approver. Without a human attached, sensitive actions do not run."""
    return False


@dataclass(frozen=True)
class Limits:
    max_steps: int = 12
    max_total_tokens: int = 200_000


@dataclass
class RunResult:
    status: str  # completed | step_limit | token_budget | refusal | truncated
    text: str
    steps: int
    input_tokens: int
    output_tokens: int
    trace: list[dict] = field(default_factory=list)

    def save_trace(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = (json.dumps(e, ensure_ascii=False, default=str) for e in self.trace)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        system: str,
        limits: Limits | None = None,
        approver: Callable[[ToolCall], bool] = deny_all,
    ):
        self.llm, self.system, self.approver = llm, system, approver
        self.tools = {t.name: t for t in tools}
        self.limits = limits or Limits()

    def run(self, task: str) -> RunResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        schemas = [t.schema() for t in self.tools.values()]
        trace: list[dict] = [{"event": "start", "task": task}]
        seen: Counter[str] = Counter()
        tokens_in = tokens_out = 0
        text = ""

        def finish(status: str, steps: int) -> RunResult:
            trace.append({"event": "finish", "status": status, "steps": steps})
            return RunResult(status, text, steps, tokens_in, tokens_out, trace)

        for step in range(1, self.limits.max_steps + 1):
            response = self.llm.complete(self.system, messages, schemas)
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            text = response.text
            trace.append(
                {
                    "event": "llm",
                    "step": step,
                    "stop_reason": response.stop_reason,
                    "text": response.text,
                    "calls": [c.name for c in response.tool_calls],
                }
            )

            if response.stop_reason == "end_turn":
                return finish("completed", step)
            # Both of these can leave a tool_use cut off mid-input: never execute it.
            if response.stop_reason == "refusal":
                return finish("refusal", step)
            if response.stop_reason == "max_tokens":
                return finish("truncated", step)

            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue

            results = [self._execute(c, seen, trace) for c in response.tool_calls]
            # One user message with every result, as parallel tool use requires.
            messages.append({"role": "user", "content": results})

            if tokens_in + tokens_out > self.limits.max_total_tokens:
                return finish("token_budget", step)

        return finish("step_limit", self.limits.max_steps)

    def _execute(self, call: ToolCall, seen: Counter, trace: list[dict]) -> dict:
        ok, output = self._dispatch(call, seen)
        body = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        if len(body) > MAX_RESULT_CHARS:
            body = body[:MAX_RESULT_CHARS] + f"\n[truncated, {len(body)} chars in total]"
        trace.append(
            {"event": "tool", "name": call.name, "input": call.input, "ok": ok, "output": body}
        )
        result = {"type": "tool_result", "tool_use_id": call.id, "content": body}
        if not ok:
            result["is_error"] = True
        return result

    def _dispatch(self, call: ToolCall, seen: Counter) -> tuple[bool, object]:
        tool = self.tools.get(call.name)
        if tool is None:
            return False, f"Unknown tool '{call.name}'. Available: {sorted(self.tools)}"
        try:
            args = tool.input_model.model_validate(call.input)
        except ValidationError as exc:
            return False, f"Invalid input: {exc.errors(include_url=False)}"

        key = json.dumps([call.name, call.input], sort_keys=True, default=str)
        seen[key] += 1
        if seen[key] > MAX_IDENTICAL_CALLS:
            return False, "Same call repeated too many times. Change approach or finish."

        if tool.needs_approval(args) and not self.approver(call):
            return False, "A human reviewer did not approve this action. It was not executed."
        try:
            return True, tool.handler(args)
        except ToolError as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001 - a broken tool must not kill the run
            return False, f"Tool failed unexpectedly ({type(exc).__name__})."
