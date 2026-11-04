"""Model candidates and the deployable scoring bundle."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ARTIFACTS, feature_columns

MODEL_PATH = ARTIFACTS / "credit_model.joblib"


def preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    cols = feature_columns(frame)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in cols if c not in numeric]
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def candidates(frame: pd.DataFrame) -> dict[str, Pipeline]:
    models = {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=0.5, max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=8, random_state=42
        ),
    }
    return {
        name: Pipeline([("prep", preprocessor(frame)), ("clf", m)]) for name, m in models.items()
    }


@dataclass
class CreditModel:
    """Everything needed to score: estimator, cost-based threshold, reference row."""

    estimator: object
    threshold: float
    reference: pd.DataFrame  # one row of medians/modes for local explanations
    version: str

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame)[:, 1]

    def score(self, applicant: dict, explain: bool = True) -> dict:
        frame = pd.DataFrame([applicant])
        p = float(self.predict_proba(frame)[0])
        return {
            "probability_of_default": round(p, 4),
            "threshold": round(self.threshold, 4),
            "recommendation": "review" if p >= self.threshold else "approve",
            "top_factors": self.explain(frame, p) if explain else [],
            "model_version": self.version,
        }

    def explain(self, frame: pd.DataFrame, p: float | None = None, k: int = 3) -> list[dict]:
        """Occlusion: how much does p move when a feature takes its reference value?

        Model-agnostic and easy to explain. It ignores feature interactions, so it is
        a reading aid and not an attribution with SHAP's guarantees.
        """
        p = float(self.predict_proba(frame)[0]) if p is None else p
        columns = feature_columns(frame)
        probes = pd.concat([frame] * len(columns), ignore_index=True)
        for i, col in enumerate(columns):
            probes.loc[i, col] = self.reference[col].iloc[0]
        deltas = p - self.predict_proba(probes)
        order = np.argsort(-np.abs(deltas))[:k]
        return [
            {
                "feature": columns[i],
                "value": _plain(frame[columns[i]].iloc[0]),
                "effect_on_pd": round(float(deltas[i]), 4),
            }
            for i in order
        ]

    def save(self, path=MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH) -> CreditModel:
        return joblib.load(path)


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def reference_row(frame: pd.DataFrame) -> pd.DataFrame:
    row = {
        c: frame[c].median() if pd.api.types.is_numeric_dtype(frame[c]) else frame[c].mode().iloc[0]
        for c in frame.columns
    }
    return pd.DataFrame([row])[list(frame.columns)]
