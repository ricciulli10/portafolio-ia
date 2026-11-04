"""Train, select, calibrate and audit the model. Run: python -m credit_risk.train

Leakage rules:
- the test set is touched once, after model, calibration and threshold are frozen;
- the threshold comes from out-of-fold predictions on the training set;
- protected attributes never reach the estimator (see data.PROTECTED).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from . import metrics as M
from .data import ARTIFACTS, COST_FN, COST_FP, PROTECTED, REPORTS, ROOT, load_credit
from .model import CreditModel, candidates, reference_row

SEED = 42
INK, MUTED, GRID, BLUE = "#1f2937", "#6b7280", "#e5e7eb", "#2a78d6"


def compare_candidates(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Repeated CV. Cost uses the Bayes threshold, so it needs no tuning per fold."""
    bayes = COST_FP / (COST_FP + COST_FN)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for name, pipe in candidates(X).items():
        scores = {"auc": [], "ap": [], "brier": [], "cost": []}
        for train, val in cv.split(X, y):
            model = clone(pipe).fit(X.iloc[train], y.iloc[train])
            p = model.predict_proba(X.iloc[val])[:, 1]
            y_val = y.iloc[val]
            scores["auc"].append(roc_auc_score(y_val, p))
            scores["ap"].append(average_precision_score(y_val, p))
            scores["brier"].append(brier_score_loss(y_val, p))
            scores["cost"].append(M.expected_cost(y_val, p, bayes))
        row = {"model": name}
        for key, values in scores.items():
            row[f"{key}_mean"] = round(float(np.mean(values)), 4)
            row[f"{key}_std"] = round(float(np.std(values)), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cost_mean").reset_index(drop=True)


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_cost_curve(y: np.ndarray, p: np.ndarray, threshold: float, path) -> None:
    grid = np.linspace(0.02, 0.98, 97)
    costs = [M.expected_cost(y, p, t) for t in grid]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(grid, costs, color=BLUE, linewidth=2)
    ax.axvline(threshold, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"threshold chosen on train: {threshold:.2f}",
        (threshold, max(costs)),
        xytext=(6, -2),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        va="top",
    )
    ax.set_xlabel("Reject when P(default) is at least", color=INK)
    ax.set_ylabel("Cost per applicant", color=INK)
    ax.set_title("Test-set cost by decision threshold", loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y: np.ndarray, p: np.ndarray, path) -> None:
    observed, predicted = calibration_curve(y, p, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.72, 0.62, "perfect", color=MUTED, fontsize=9)
    ax.plot(
        predicted,
        observed,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax.set_xlabel("Predicted P(default)", color=INK)
    ax.set_ylabel("Observed default rate", color=INK)
    ax.set_title("Calibration on the test set", loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def log_to_mlflow(result: dict) -> None:
    """Track the run when MLflow is installed. The store sits in git-ignored artifacts/."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{ARTIFACTS.relative_to(ROOT).as_posix()}/mlflow.db")
        mlflow.set_experiment("credit-risk")
    except ImportError:
        print("MLflow unavailable: run not tracked")
        return
    test = result["test"]
    with mlflow.start_run(run_name=result["winner"]):
        mlflow.log_params({"model": result["winner"], "threshold": result["threshold"]})
        keys = ["roc_auc", "average_precision", "brier", "cost_per_applicant"]
        mlflow.log_metrics({k: test[k] for k in keys})


def main() -> None:
    X, y = load_credit()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    table = compare_candidates(X_tr, y_tr)
    winner = table.iloc[0]["model"]
    print(table.to_string(index=False))

    base = candidates(X_tr)[winner]
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    oof = cross_val_predict(
        calibrated,
        X_tr,
        y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba",
    )[:, 1]
    threshold, train_cost = M.best_threshold(y_tr.to_numpy(), oof)
    calibrated.fit(X_tr, y_tr)

    # ---- frozen from here on: first and only use of the test set ----
    p = calibrated.predict_proba(X_te)[:, 1]
    y_arr = y_te.to_numpy()
    cost = lambda a, b: M.expected_cost(a, b, threshold)  # noqa: E731
    reject = p >= threshold
    test = {
        "n": len(y_te),
        "roc_auc": round(roc_auc_score(y_te, p), 4),
        "roc_auc_ci95": [round(v, 4) for v in M.bootstrap_ci(y_arr, p, roc_auc_score)],
        "average_precision": round(average_precision_score(y_te, p), 4),
        "brier": round(brier_score_loss(y_te, p), 4),
        "cost_per_applicant": round(cost(y_arr, p), 4),
        "cost_ci95": [round(v, 4) for v in M.bootstrap_ci(y_arr, p, cost)],
        "cost_approve_everyone": round(M.expected_cost(y_arr, p, 2.0), 4),
        "cost_reject_everyone": round(M.expected_cost(y_arr, p, -1.0), 4),
        "approval_rate": round(float(1 - reject.mean()), 4),
        "default_rate_among_approved": round(float(y_arr[~reject].mean()), 4),
    }

    # Sex is only available fused with marital status; "male" prefix is parsed from it.
    sex = X_te["personal_status"].str.split().str[0].rename("sex")
    age = pd.cut(X_te["age"], [0, 24, 200], labels=["under_25", "25_plus"]).astype(str)
    fairness = {
        "sex": M.group_rates(y_te, reject, sex).to_dict("records"),
        "age": M.group_rates(y_te, reject, age).to_dict("records"),
        "foreign_worker": M.group_rates(y_te, reject, X_te["foreign_worker"]).to_dict("records"),
    }

    drift = M.drift_report(X_tr.drop(columns=PROTECTED), X_te.drop(columns=PROTECTED))

    bundle = CreditModel(calibrated, threshold, reference_row(X_tr), f"{winner}-v1")
    bundle.save()

    REPORTS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    plot_cost_curve(y_arr, p, threshold, REPORTS / "figures" / "credit_cost_curve.png")
    plot_calibration(y_arr, p, REPORTS / "figures" / "credit_calibration.png")
    result = {
        "winner": winner,
        "threshold": round(threshold, 4),
        "train_oof_cost": round(train_cost, 4),
        "cv": table.to_dict("records"),
        "test": test,
        "fairness": fairness,
        "max_psi_train_vs_test": drift.head(3).to_dict("records"),
    }
    REPORTS.joinpath("credit_risk.json").write_text(json.dumps(result, indent=2))
    log_to_mlflow(result)
    print(json.dumps({"winner": winner, "threshold": threshold, "test": test}, indent=2))


if __name__ == "__main__":
    main()
