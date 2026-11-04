import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from credit_risk import metrics as M
from credit_risk.data import PROTECTED, feature_columns, load_credit
from credit_risk.model import candidates


def test_data_shape_and_prior():
    X, y = load_credit()
    assert X.shape == (1000, 20)
    assert y.mean() == 0.3


def test_protected_attributes_never_reach_the_estimator():
    X, _ = load_credit()
    assert not set(PROTECTED) & set(feature_columns(X))
    prep = candidates(X)["logistic_regression"].named_steps["prep"]
    used = [c for _, _, cols in prep.transformers for c in cols]
    assert not set(PROTECTED) & set(used)


def test_expected_cost_by_hand():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    # one bad approved (5) + one good rejected (1) over four applicants
    assert M.expected_cost(y, p, 0.5) == 1.5


def test_best_threshold_finds_separable_split():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.full(50, 0.2), np.full(50, 0.8)]
    threshold, cost = M.best_threshold(y, p)
    assert 0.2 < threshold <= 0.8 and cost == 0


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    assert M.psi(base, rng.normal(size=5000)) < 0.1
    assert M.psi(base, rng.normal(loc=1.5, size=5000)) > 0.25


def test_drift_report_flags_new_category():
    ref = pd.DataFrame({"a": ["x"] * 100})
    cur = pd.DataFrame({"a": ["y"] * 100})
    assert M.drift_report(ref, cur).loc[0, "status"] == "investigate"


def _payload():
    X, _ = load_credit()
    row = X.drop(columns=PROTECTED).iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def test_api_scores_and_validates(credit_model):
    from credit_risk.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/score", json=_payload())
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["probability_of_default"] <= 1
        assert len(body["top_factors"]) == 3
        assert client.post("/score", json={**_payload(), "age": 30}).status_code == 422
        assert client.post("/score", json={**_payload(), "duration": -1}).status_code == 422


def test_api_key_is_enforced_when_configured(credit_model, monkeypatch):
    from credit_risk.api import app

    monkeypatch.setenv("CREDIT_API_KEY", "unit-test-value")
    with TestClient(app) as client:
        assert client.post("/score", json=_payload()).status_code == 401
        wrong = {"X-API-Key": "nope"}
        assert client.post("/score", json=_payload(), headers=wrong).status_code == 401
        right = {"X-API-Key": "unit-test-value"}
        assert client.post("/score", json=_payload(), headers=right).status_code == 200


def test_api_rejects_oversized_strings(credit_model):
    from credit_risk.api import app

    with TestClient(app) as client:
        body = {**_payload(), "purpose": "x" * 65}
        assert client.post("/score", json=body).status_code == 422
