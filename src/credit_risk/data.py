"""Load the German Credit dataset (OpenML 31, snapshot in data/credit_g.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "credit_g.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

TARGET = "class"

# Excluded from the model on purpose. They are kept in the frame only for the fairness audit.
PROTECTED = ["personal_status", "age", "foreign_worker"]

# The dataset ships with this cost matrix: lending to a bad payer costs 5x
# rejecting a good one.
COST_FN = 5.0  # bad applicant approved
COST_FP = 1.0  # good applicant rejected


def load_credit(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return all columns as X and y=1 for a bad payer."""
    df = pd.read_csv(path)
    y = (df[TARGET] == "bad").astype(int).rename("bad")
    return df.drop(columns=[TARGET]), y


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in PROTECTED]
