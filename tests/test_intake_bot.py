import json
import time

import pytest
from fastapi.testclient import TestClient

from credit_risk.data import PROTECTED, load_credit
from intake_bot.app import MAX_BODY_BYTES, app, sign, verify

SECRET = "unit-test-value"  # pragma: allowlist secret
URL = "/webhooks/application"


def payload() -> bytes:
    X, _ = load_credit()
    row = X.drop(columns=PROTECTED).iloc[0].to_dict()
    return json.dumps({k: v.item() if hasattr(v, "item") else v for k, v in row.items()}).encode()


def headers(body: bytes, key: str = "k1", timestamp: str | None = None, secret: str = SECRET):
    timestamp = timestamp or str(int(time.time()))
    return {
        "X-Timestamp": timestamp,
        "X-Signature": sign(secret, timestamp, body),
        "X-Idempotency-Key": key,
        "Content-Type": "application/json",
    }


@pytest.fixture
def client(credit_model, tmp_path, monkeypatch):
    monkeypatch.setenv("INTAKE_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("INTAKE_DB", str(tmp_path / "intake.sqlite"))
    with TestClient(app) as c:
        yield c


def test_valid_signature_routes_application(client):
    body = payload()
    response = client.post(URL, content=body, headers=headers(body))
    assert response.status_code == 200
    assert response.json()["route"] in {"analyst_queue", "senior_review"}
    assert response.json()["duplicate"] is False


def test_wrong_secret_is_refused(client):
    body = payload()
    response = client.post(URL, content=body, headers=headers(body, secret="other"))
    assert response.status_code == 401


def test_tampered_body_is_refused(client):
    body = payload()
    sent = headers(body)
    response = client.post(URL, content=body.replace(b"radio", b"cars!"), headers=sent)
    assert response.status_code == 401


def test_replayed_old_request_is_refused(client):
    body = payload()
    old = str(int(time.time()) - 3600)
    assert client.post(URL, content=body, headers=headers(body, timestamp=old)).status_code == 401


def test_retry_returns_stored_result(client, credit_model, monkeypatch):
    body = payload()
    first = client.post(URL, content=body, headers=headers(body, key="same")).json()

    def fail(*_):
        raise AssertionError("a retry must not run the model")

    monkeypatch.setattr("intake_bot.app.triage", fail)
    second = client.post(URL, content=body, headers=headers(body, key="same")).json()
    assert second["duplicate"] is True
    assert second["route"] == first["route"]


def test_fails_closed_without_a_secret(client, monkeypatch):
    monkeypatch.delenv("INTAKE_WEBHOOK_SECRET")
    body = payload()
    assert client.post(URL, content=body, headers=headers(body)).status_code == 503


def test_oversized_body_is_refused(client):
    body = b"x" * (MAX_BODY_BYTES + 1)
    assert client.post(URL, content=body, headers=headers(body)).status_code == 413


def test_protected_fields_are_rejected(client):
    body = json.dumps({**json.loads(payload()), "age": 30}).encode()
    response = client.post(URL, content=body, headers=headers(body))
    assert response.status_code == 422
    assert "age" not in response.text


def test_non_ascii_signature_is_refused_cleanly(client):
    body = payload()
    sent = {**headers(body), "X-Signature": "ñ".encode("latin-1")}
    assert client.post(URL, content=body, headers=sent).status_code == 401


def test_verify_handles_garbage_timestamp():
    assert verify(SECRET, "not-a-number", b"", "sha256=0", time.time()) is False
