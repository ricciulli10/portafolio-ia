"""Signed webhook that triages loan applications. Run: uvicorn intake_bot.app:app --port 8001

Built for an automation tool (n8n, Make, Zapier) calling it over HTTP, and written for the
ways webhooks fail in practice:

- forged calls: HMAC-SHA256 over timestamp and raw body, compared in constant time;
- replayed calls: timestamps older than five minutes are refused;
- retries: an idempotency key returns the stored result and does no new work;
- misconfiguration: with no secret set the endpoint refuses everything (fails closed).

It only routes work. A person or the underwriting agent makes the credit decision.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from credit_risk.api import Applicant
from credit_risk.data import ARTIFACTS
from credit_risk.model import CreditModel
from underwriter.domain import CaseFile

MAX_BODY_BYTES = 16_384
MAX_SKEW_SECONDS = 300
MAX_KEY_LENGTH = 128


def sign(secret: str, timestamp: str, body: bytes) -> str:
    message = timestamp.encode() + b"." + body
    return "sha256=" + hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp: str, body: bytes, signature: str, now: float) -> bool:
    try:
        fresh = abs(now - int(timestamp)) <= MAX_SKEW_SECONDS
    except ValueError:
        return False
    return fresh and hmac.compare_digest(sign(secret, timestamp, body), signature)


class Ledger:
    """Idempotency store. A new connection per call keeps it safe across threads."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL)"
            )

    def get(self, key: str) -> dict | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT response FROM events WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?)",
                (key, json.dumps(response), time.time()),
            )


def triage(fields: dict, model: CreditModel) -> dict:
    """Apply the same written rules the agent is held to."""
    score = model.score(fields)
    case = CaseFile({"event": {"fields": fields}}, scores={"event": score})
    reasons = case.must_refer("event")
    return {
        "route": "senior_review" if reasons else "analyst_queue",
        "reasons": reasons,
        "probability_of_default": score["probability_of_default"],
        "model_version": score["model_version"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = CreditModel.load()
    db = Path(os.environ.get("INTAKE_DB", ARTIFACTS / "intake.sqlite"))
    app.state.ledger = Ledger(db)
    yield


app = FastAPI(title="Application intake", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/webhooks/application")
async def receive(
    request: Request,
    x_timestamp: str = Header(max_length=20),
    x_signature: str = Header(max_length=128),
    x_idempotency_key: str = Header(min_length=1, max_length=MAX_KEY_LENGTH),
) -> dict:
    secret = os.environ.get("INTAKE_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(503, "Endpoint not configured")

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, "Payload too large")
    if not verify(secret, x_timestamp, body, x_signature, time.time()):
        raise HTTPException(401, "Invalid signature")

    ledger: Ledger = request.app.state.ledger
    stored = ledger.get(x_idempotency_key)
    if stored is not None:
        return {**stored, "duplicate": True}

    try:
        applicant = Applicant.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(422, "Invalid application payload") from exc

    result = triage(applicant.model_dump(), request.app.state.model)
    ledger.put(x_idempotency_key, result)
    return {**result, "duplicate": False}
