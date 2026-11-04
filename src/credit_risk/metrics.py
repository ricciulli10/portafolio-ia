"""Business metrics, drift and fairness helpers. Pure functions, no I/O."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .data import COST_FN, COST_FP


def expected_cost(
    y_true: np.ndarray,
    p_bad: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Mean cost per applicant. An applicant is rejected when p_bad >= threshold."""
    y_true = np.asarray(y_true)
    reject = np.asarray(p_bad) >= threshold
    fn = np.sum((y_true == 1) & ~reject)
    fp = np.sum((y_true == 0) & reject)
    return float((cost_fn * fn + cost_fp * fp) / len(y_true))


def best_threshold(y_true: np.ndarray, p_bad: np.ndarray) -> tuple[float, float]:
    """Return the threshold with the lowest cost, searched on a fixed grid."""
    grid = np.linspace(0.02, 0.98, 97)
    costs = np.array([expected_cost(y_true, p_bad, t) for t in grid])
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def bootstrap_ci(
    y_true: np.ndarray,
    p: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile bootstrap interval, skipping one-class resamples."""
    rng = np.random.default_rng(seed)
    y_true, p = np.asarray(y_true), np.asarray(p)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        values.append(stat(y_true[idx], p[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of a numeric variable, with reference-quantile bins."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
    e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI per column. Usual reading: <0.1 stable, 0.1-0.25 watch, >0.25 investigate."""
    rows = []
    for col in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            value = psi(reference[col].to_numpy(), current[col].to_numpy())
        else:
            value = psi_categorical(reference[col], current[col])
        status = "stable" if value < 0.1 else "watch" if value < 0.25 else "investigate"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def group_rates(y_true: pd.Series, reject: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Approval rate and true-good approval rate (equal opportunity) per group."""
    rows = []
    for name, idx in groups.groupby(groups).groups.items():
        mask = groups.index.isin(idx)
        good = mask & (y_true.to_numpy() == 0)
        rows.append(
            {
                "group": str(name),
                "n": int(mask.sum()),
                "approval_rate": round(float(1 - reject[mask].mean()), 3),
                "good_payer_approval_rate": round(float(1 - reject[good].mean()), 3)
                if good.any()
                else None,
            }
        )
    return pd.DataFrame(rows)
