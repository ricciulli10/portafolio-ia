"""Applications, the decision ledger and the controls no prompt can override."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from agentkit.tools import ToolError
from credit_risk.data import PROTECTED, load_credit
from credit_risk.train import SEED

SENIOR_LIMIT = 5000
PRODUCT_LIMIT = 15000
BANNED_REASONING = re.compile(
    r"\b(sex|gender|male|female|women|woman|men|man|marital|married|single|divorced|widow\w*"
    r"|age|aged|old|young|elderly|nationality|foreign\w*|migrant|immigrant)\b",
    re.IGNORECASE,
)


def load_applications(limit: int | None = None) -> dict[str, dict]:
    """Applications come from the held-out test split, so the model never saw them."""
    X, y = load_credit()
    _, X_te, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows = X_te.drop(columns=PROTECTED)
    rows = pd.DataFrame(rows).head(limit) if limit else rows
    return {
        f"APP-{i:04d}": {"fields": row.to_dict(), "customer_note": ""}
        for i, (_, row) in enumerate(rows.iterrows(), start=1)
    }


@dataclass
class CaseFile:
    """What actually happened in this run, as observed by the tools."""

    applications: dict[str, dict]
    scores: dict[str, dict] = field(default_factory=dict)
    sections_read: set[str] = field(default_factory=set)
    decisions: dict[str, dict] = field(default_factory=dict)

    def application(self, application_id: str) -> dict:
        if application_id not in self.applications:
            raise ToolError(f"No application with id {application_id}.")
        return self.applications[application_id]

    def must_refer(self, application_id: str) -> list[str]:
        """Conditions that remove an analyst's authority to approve."""
        fields = self.application(application_id)["fields"]
        score = self.scores[application_id]
        reasons = []
        if score["recommendation"] != "approve":
            reasons.append("probability of default is at or above the cutoff")
        if fields["credit_amount"] > SENIOR_LIMIT:
            reasons.append(f"amount exceeds {SENIOR_LIMIT}")
        if fields["installment_commitment"] == 4 and fields["duration"] > 36:
            reasons.append("heaviest instalment burden with a term over 36 months")
        if fields["existing_credits"] >= 3:
            reasons.append("three or more live credits need statements on file")
        return reasons

    def check_decision(
        self, application_id: str, decision: str, rationale: str, cited: list[str]
    ) -> None:
        self.application(application_id)
        if application_id in self.decisions:
            raise ToolError("A decision is already recorded for this application.")
        if application_id not in self.scores:
            raise ToolError("Score the application before recording a decision.")
        if not cited:
            raise ToolError("Cite at least one policy section.")
        unread = sorted(set(cited) - self.sections_read)
        if unread:
            raise ToolError(f"These sections were not returned by search_policy: {unread}")
        match = BANNED_REASONING.search(rationale)
        if match:
            raise ToolError(f"Rationale uses a protected attribute ('{match.group()}').")
        if decision == "approve":
            reasons = self.must_refer(application_id)
            if reasons:
                raise ToolError("Approval blocked, refer instead: " + "; ".join(reasons))
