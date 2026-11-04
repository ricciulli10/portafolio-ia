from types import SimpleNamespace

from anthropic.types.beta import BetaMessage
from pydantic import BaseModel, ConfigDict

from agentkit.llm import AnthropicLLM, ScriptedLLM, call, calls, text
from agentkit.runtime import Agent, Limits
from agentkit.tools import Tool, ToolError


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int


def tool(handler, **kwargs):
    return Tool("double", "Doubles.", Input, handler, **kwargs)


def agent(script, handler=lambda a: a.x * 2, limits=None, approver=None, **tool_kwargs):
    kwargs = {"approver": approver} if approver else {}
    return Agent(ScriptedLLM(script), [tool(handler, **tool_kwargs)], "sys", limits, **kwargs)


def tool_events(result):
    return [e for e in result.trace if e["event"] == "tool"]


def test_happy_path():
    result = agent([call("double", {"x": 2}), text("4")]).run("go")
    assert result.status == "completed"
    assert result.text == "4"
    assert tool_events(result)[0]["output"] == "4"


def test_schema_is_strict_compatible():
    schema = tool(lambda a: a).schema()
    assert schema["strict"] is True
    assert schema["input_schema"]["additionalProperties"] is False
    assert schema["input_schema"]["required"] == ["x"]


def test_tool_error_is_returned_to_model():
    def boom(_):
        raise ToolError("nope")

    llm = ScriptedLLM([call("double", {"x": 1}), text("ok")])
    seen = []
    original = llm.complete

    def spy(system, messages, tools):
        seen.append(list(messages))
        return original(system, messages, tools)

    llm.complete = spy
    Agent(llm, [tool(boom)], "sys").run("go")
    result_block = seen[1][-1]["content"][0]
    assert result_block["is_error"] is True
    assert result_block["content"] == "nope"


def test_unexpected_exception_does_not_leak_details():
    def boom(_):
        raise RuntimeError("secret path /etc/x")

    event = tool_events(agent([call("double", {"x": 1}), text("ok")], boom).run("go"))[0]
    assert event["ok"] is False
    assert "secret" not in event["output"]


def test_parallel_results_go_in_one_message():
    llm = ScriptedLLM([calls([("double", {"x": 1}), ("double", {"x": 2})]), text("ok")])
    captured = {}
    original = llm.complete

    def spy(system, messages, tools):
        captured["messages"] = list(messages)
        return original(system, messages, tools)

    llm.complete = spy
    Agent(llm, [tool(lambda a: a.x)], "sys").run("go")
    assert captured["messages"][-1]["role"] == "user"
    assert len(captured["messages"][-1]["content"]) == 2


def test_approval_defaults_to_deny():
    ran = []
    result = agent(
        [call("double", {"x": 1}), text("ok")],
        lambda a: ran.append(a),
        needs_approval=lambda _: True,
    ).run("go")
    assert not ran
    assert tool_events(result)[0]["ok"] is False


def test_approval_granted_runs_tool():
    ran = []
    agent(
        [call("double", {"x": 1}), text("ok")],
        lambda a: ran.append(a) or 1,
        approver=lambda _: True,
        needs_approval=lambda _: True,
    ).run("go")
    assert len(ran) == 1


def test_step_limit():
    result = agent([call("double", {"x": 1})], limits=Limits(max_steps=2)).run("go")
    assert result.status == "step_limit"


def test_token_budget():
    response = call("double", {"x": 1})
    response.input_tokens = 500
    result = agent([response], limits=Limits(max_total_tokens=100)).run("go")
    assert result.status == "token_budget"


def test_pause_turn_continues_without_tools():
    paused = call("double", {"x": 1}, stop_reason="pause_turn")
    result = agent([paused, text("ok")]).run("go")
    assert result.status == "completed"
    assert not tool_events(result)


def test_long_output_is_marked_as_truncated():
    event = tool_events(
        agent([call("double", {"x": 1}), text("k")], lambda _: "y" * 30000).run("g")
    )
    assert "truncated, 30000 chars" in event[0]["output"]


def test_anthropic_adapter_parses_real_sdk_types_and_sends_required_params():
    message = BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [
                {"type": "text", "text": "Scoring."},
                {"type": "tool_use", "id": "tu_1", "name": "double", "input": {"x": 3}},
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
    )
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return message

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    response = AnthropicLLM(client=client).complete("sys", [], [])
    assert response.stop_reason == "tool_use"
    assert response.text == "Scoring."
    assert response.tool_calls[0].input == {"x": 3}
    assert response.content is message.content
    assert sent["model"] == "claude-opus-5-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert "temperature" not in sent
