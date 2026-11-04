"""One full run with no key. Run: python -m underwriter.demo"""

from __future__ import annotations

from credit_risk.data import REPORTS

from .evals import fresh, make_world, run_with
from .planner import PlannerLLM


def main() -> None:
    world = make_world()
    case = fresh(world)
    result = run_with(world, PlannerLLM(world.high), case)
    result.save_trace(REPORTS / "sample_agent_trace.jsonl")
    for event in result.trace:
        if event["event"] == "tool":
            print(f"[tool] {event['name']} ok={event['ok']}\n  -> {event['output'][:160]}")
    print("ledger:", case.decisions)


if __name__ == "__main__":
    main()
