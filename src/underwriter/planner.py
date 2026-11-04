"""A deterministic stand-in for the LLM so the full loop runs with no key and no cost.

It follows a fixed procedure and makes no judgement calls. It exists to demo and test the
plumbing. It says nothing about how well a real model would reason.
"""

from __future__ import annotations

import json

from agentkit.llm import LLMResponse, calls, text


def _last_results(messages: list[dict]) -> list[dict]:
    last = messages[-1]["content"]
    return (
        [json.loads(r["content"]) for r in last if not r.get("is_error")]
        if isinstance(last, list)
        else []
    )


class PlannerLLM:
    def __init__(self, application_id: str):
        self.id = application_id
        self.stage = 0
        self.application: dict = {}
        self.score: dict = {}

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.stage += 1
        results = _last_results(messages)
        args = {"application_id": self.id}
        if self.stage == 1:
            return calls([("get_application", args)])
        if self.stage == 2:
            self.application = results[0]
            return calls([("score_application", args)])
        if self.stage == 3:
            self.score = results[0]
            return calls([("search_policy", {"query": "who may approve above the cutoff"})])
        if self.stage == 4:
            sections = [hit["section"] for hit in results[0]]
            return self._decide(sections)
        return text(f"Decision recorded for {self.id}.")

    def _decide(self, sections: list[str]) -> LLMResponse:
        fields = self.application["fields"]
        pd_value = self.score["probability_of_default"]
        risky = self.score["recommendation"] != "approve"
        large = fields["credit_amount"] > 5000
        burden = fields["installment_commitment"] == 4 and fields["duration"] > 36
        many = fields["existing_credits"] >= 3
        decision = "refer" if risky or large or burden or many else "approve"
        rationale = (
            f"PD {pd_value:.3f} against cutoff {self.score['threshold']:.3f}, "
            f"amount {fields['credit_amount']:.0f}, term {fields['duration']} months."
        )
        return calls(
            [
                (
                    "record_decision",
                    {
                        "application_id": self.id,
                        "decision": decision,
                        "rationale": rationale,
                        "cited_sections": sections[:2],
                    },
                )
            ]
        )
