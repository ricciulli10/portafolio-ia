"""The runtime talks to this interface, which keeps it testable without a network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    stop_reason: str  # end_turn | tool_use | max_tokens | refusal | pause_turn
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: Any = None  # provider blocks, replayed unchanged on the next request
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicLLM:
    """Claude through the official SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        max_tokens: int = 16000,
        client: Any = None,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        return LLMResponse(
            stop_reason=response.stop_reason,
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[
                ToolCall(b.id, b.name, dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ],
            content=response.content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class ScriptedLLM:
    """Replays fixed responses. Used to test the runtime, including hostile behaviour."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls = 0

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls += 1
        if self.calls > len(self.script):
            return self.script[-1]
        return self.script[self.calls - 1]


def text(message: str) -> LLMResponse:
    return LLMResponse("end_turn", text=message, content=[{"type": "text", "text": message}])


def call(name: str, tool_input: dict, id: str = "t1", stop_reason: str = "tool_use") -> LLMResponse:
    return calls([(name, tool_input)], ids=[id], stop_reason=stop_reason)


def calls(
    items: list[tuple[str, dict]], ids: list[str] | None = None, stop_reason: str = "tool_use"
) -> LLMResponse:
    ids = ids or [f"t{i}" for i in range(len(items))]
    tool_calls = [ToolCall(i, n, a) for i, (n, a) in zip(ids, items, strict=True)]
    blocks = [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tool_calls
    ]
    return LLMResponse(stop_reason, tool_calls=tool_calls, content=blocks)
