"""Scoring API. Run: uvicorn credit_risk.api:app --port 8000"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .model import CreditModel

Category = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class Applicant(BaseModel):
    """Field names follow the dataset. Categories stay strings; unknown ones are tolerated."""

    model_config = ConfigDict(extra="forbid")

    checking_status: Category
    duration: int = Field(gt=0, le=120, description="Loan length in months")
    credit_history: Category
    purpose: Category
    credit_amount: float = Field(gt=0, le=1_000_000)
    savings_status: Category
    employment: Category
    installment_commitment: int = Field(ge=1, le=4)
    other_parties: Category
    residence_since: int = Field(ge=1, le=4)
    property_magnitude: Category
    other_payment_plans: Category
    housing: Category
    existing_credits: int = Field(ge=0, le=50)
    job: Category
    num_dependents: int = Field(ge=0, le=30)
    own_telephone: Category


class Factor(BaseModel):
    feature: str
    value: str | int | float
    effect_on_pd: float


class Score(BaseModel):
    probability_of_default: float
    threshold: float
    recommendation: Literal["approve", "review"]
    top_factors: list[Factor]
    model_version: str


state: dict[str, CreditModel] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["model"] = CreditModel.load()
    yield
    state.clear()


app = FastAPI(title="Credit risk scoring", version="1.0", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforced whenever CREDIT_API_KEY is set. The key only ever comes from the environment."""
    expected = os.environ.get("CREDIT_API_KEY")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(401, "Invalid or missing API key")


@app.get("/health")
def health() -> dict:
    model = state.get("model")
    return {
        "status": "ok" if model else "no_model",
        "model_version": getattr(model, "version", None),
    }


@app.post("/score", response_model=Score, dependencies=[Depends(require_api_key)])
def score(applicant: Applicant, explain: bool = True) -> dict:
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "Model not available")
    result = model.score(applicant.model_dump(), explain=explain)
    if not np.isfinite(result["probability_of_default"]):
        raise HTTPException(500, "Model returned a non-finite score")
    return result
