"""Serving latency, in process. Run: python -m credit_risk.benchmark"""

from __future__ import annotations

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from .api import app
from .data import PROTECTED, REPORTS, load_credit


def measure(client: TestClient, rows: list[dict], explain: bool) -> dict:
    timings = []
    for row in rows:
        start = time.perf_counter()
        client.post("/score", json=row, params={"explain": explain}).raise_for_status()
        timings.append((time.perf_counter() - start) * 1000)
    settled = np.array(timings[20:])  # skip warm-up
    return {
        "p50_ms": round(float(np.percentile(settled, 50)), 1),
        "p95_ms": round(float(np.percentile(settled, 95)), 1),
    }


def main(requests: int = 300) -> None:
    X, _ = load_credit()
    rows = X.drop(columns=PROTECTED).head(requests).to_dict("records")
    with TestClient(app) as client:
        report = {
            "requests": requests - 20,
            "note": "In-process over ASGI on a desktop CPU, one client, no network.",
            "with_explanation": measure(client, rows, True),
            "score_only": measure(client, rows, False),
        }
    REPORTS.joinpath("serving_latency.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
