"""A/B test toolkit. Run: python -m credit_risk.experiment

The experiment here is simulated, because the dataset has no intervention. The simulation
is also the check: an A/A test should flag about 5% of runs, and the power should match
the sample size formula.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import norm

from .data import REPORTS


def sample_size(base: float, lift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm size to detect an absolute lift in a proportion, two-sided."""
    other = base + lift
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(z**2 * (base * (1 - base) + other * (1 - other)) / lift**2)


def two_proportion_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled if se_pooled else 0.0
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    margin = norm.ppf(0.975) * se
    return {
        "rate_a": round(p_a, 4),
        "rate_b": round(p_b, 4),
        "lift": round(p_b - p_a, 4),
        "ci95": [round(p_b - p_a - margin, 4), round(p_b - p_a + margin, 4)],
        "p_value": round(float(2 * norm.sf(abs(z))), 4),
    }


def rejection_rate(p_a: float, p_b: float, n: int, runs: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.binomial(n, p_a, runs)
    b = rng.binomial(n, p_b, runs)
    hits = sum(
        two_proportion_test(int(x), n, int(y), n)["p_value"] < 0.05
        for x, y in zip(a, b, strict=True)
    )
    return round(hits / runs, 3)


def main() -> None:
    base, lift = 0.70, 0.03  # simulated on-time payment rate and the lift worth detecting
    n = sample_size(base, lift)
    rng = np.random.default_rng(7)
    observed = two_proportion_test(
        int(rng.binomial(n, base)), n, int(rng.binomial(n, base + lift)), n
    )
    report = {
        "simulated": True,
        "question": "Does an SMS reminder raise on-time payment?",
        "per_arm_sample_size": n,
        "single_run": observed,
        "aa_false_positive_rate": rejection_rate(base, base, n, 4000, 1),
        "ab_power": rejection_rate(base, base + lift, n, 4000, 2),
    }
    REPORTS.joinpath("ab_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
