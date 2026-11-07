"""Comparable monthly/fixed model evidence and complete feature monitoring."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json
import math

from utils.config import GOLD_DIR, PROJECT_ROOT
from utils.model_pipeline import (MONITORING_DIR, FIXED_MONITORING_DIR,
                                  FEATURE_MONITORING_DIR, FIXED_FEATURE_MONITORING_DIR,
                                  MONITORED_FEATURES, MODEL_BANK, PREDICTIONS_DIR,
                                  FIXED_PREDICTIONS_DIR, SCORE_THRESHOLD)

REPORTS_DIR = PROJECT_ROOT / "reports"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
REVIEW_SHARES = (0.10, 0.20, 0.30, 0.50)


def business_view(as_of: str) -> pd.DataFrame:
    """Usefulness in business terms, on loans whose month-six outcome is known by as_of.

    Each month, reviewers check the riskiest X% of that month's applications.
    For each X we report how many flagged loans really went 30+ days late and
    what share of all late loans the review would have caught.
    """
    labels = pd.read_parquet(GOLD_DIR / "label_store.parquet")[["loan_id", "label_observation_date", "label"]]
    rows = []
    for strategy, directory in [("monthly_selected", PREDICTIONS_DIR), ("fixed_june_2024", FIXED_PREDICTIONS_DIR)]:
        paths = [p for p in sorted(directory.glob("cohort_date=*/predictions.parquet"))
                 if p.parent.name.split("=", 1)[1] <= as_of]
        if not paths:
            continue
        scored = pd.concat([pd.read_parquet(p) for p in paths]).merge(labels, on="loan_id", validate="one_to_one")
        known = scored[scored["label_observation_date"].le(pd.Timestamp(as_of))].copy()
        if known.empty:
            continue
        late_total = int(known["label"].sum())
        rank = known.groupby("feature_as_of_date")["default_probability"].rank(method="first", ascending=False)
        size = known.groupby("feature_as_of_date")["loan_id"].transform("size")
        rules = [(f"riskiest {int(share * 100)}% each month", rank <= np.ceil(share * size))
                 for share in REVIEW_SHARES]
        rules.append((f"probability >= {SCORE_THRESHOLD}", known["default_probability"].ge(SCORE_THRESHOLD)))
        for rule, flagged in rules:
            caught = int((flagged & known["label"].eq(1)).sum())
            rows.append({"as_of_date": as_of, "scoring_strategy": strategy, "review_rule": rule,
                         "cohorts_with_known_outcome": int(known["feature_as_of_date"].nunique()),
                         "applications": int(len(known)), "late_loans": late_total,
                         "overall_late_rate": late_total / len(known),
                         "flagged": int(flagged.sum()), "flagged_share": float(flagged.mean()),
                         "late_among_flagged": caught / flagged.sum() if flagged.sum() else math.nan,
                         "share_of_late_loans_caught": caught / late_total if late_total else math.nan})
    return pd.DataFrame(rows)


def tuning_summary(as_of: str) -> pd.DataFrame:
    """Every grid setting tried by every model trained up to as_of."""
    rows = []
    for path in sorted(MODEL_BANK.glob("credit_model_*.json")):
        metadata = json.loads(path.read_text())
        if metadata["as_of_date"] > as_of:
            continue
        for trial in metadata.get("tuning_results", []):
            rows.append({"model_version": metadata["model_version"], "algorithm": trial["algorithm"],
                         "params": json.dumps(trial["params"], sort_keys=True),
                         "validation_auc": trial["validation_auc"],
                         "selected": (trial["algorithm"] == metadata["selected_algorithm"]
                                      and trial["params"] == metadata.get("selected_params"))})
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#DCE5E9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def render_monitoring(as_of: str) -> dict:
    data = pd.read_parquet(MONITORING_DIR / f"as_of_date={as_of}" / "metrics.parquet").sort_values("cohort_date")
    fixed = pd.read_parquet(FIXED_MONITORING_DIR / f"as_of_date={as_of}" / "metrics.parquet").sort_values("cohort_date")
    if not data["cohort_date"].reset_index(drop=True).equals(fixed["cohort_date"].reset_index(drop=True)):
        raise ValueError("Strategy comparisons require the same application cohorts")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(REPORTS_DIR / f"monitoring_as_of_{as_of}.csv", index=False)
    fixed.to_csv(REPORTS_DIR / f"fixed_monitoring_as_of_{as_of}.csv", index=False)
    comparison = data[["cohort_date", "model_version", "n_predictions", "roc_auc", "psi_score", "max_csi"]].merge(
        fixed[["cohort_date", "model_version", "n_predictions", "roc_auc", "psi_score", "max_csi"]],
        on="cohort_date", suffixes=("_monthly", "_fixed"), validate="one_to_one")
    if not comparison["n_predictions_monthly"].equals(comparison["n_predictions_fixed"]):
        raise ValueError("Strategy comparison populations differ")
    comparison["auc_difference_monthly_minus_fixed"] = comparison["roc_auc_monthly"] - comparison["roc_auc_fixed"]
    comparison.to_csv(REPORTS_DIR / f"strategy_comparison_as_of_{as_of}.csv", index=False)

    labels = pd.to_datetime(data["cohort_date"]).dt.strftime("%b %Y")
    x = np.arange(len(data))
    fig, axes = plt.subplots(3, 1, figsize=(12.4, 10.2), sharex=True, layout="constrained")
    for frame, color, line, label in [(data, TEAL, "-", "Monthly selected versions"),
                                       (fixed, ORANGE, "--", "Fixed June 2024 model")]:
        observed = frame["roc_auc"].notna().to_numpy()
        axes[0].plot(x[observed], frame.loc[observed, "roc_auc"], color=color, linestyle=line,
                     marker="o", linewidth=2, label=label)
        axes[1].plot(x, frame["psi_score"], color=color, linestyle=line, marker="o", linewidth=2, label=label)
        axes[2].plot(x, frame["max_csi"], color=color, linestyle=line, marker="o", linewidth=2, label=label)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Observed ROC AUC")
    axes[0].set_title("Same applications and matured outcomes; distinct model strategies", loc="left", fontsize=12)
    axes[1].set_ylabel("Prediction PSI")
    axes[2].set_ylabel("Maximum CSI (16 inputs)")
    for ax in axes[1:]:
        ax.axhline(0.10, color="#D9A441", linestyle=":", linewidth=1.2, label="0.10 review")
        ax.axhline(0.25, color="#9A354A", linestyle="--", linewidth=1.2, label="0.25 alert")
        ax.set_ylim(bottom=0)
    for ax in axes:
        _style(ax)
        ax.legend(loc="best", frameon=False, ncol=2, fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_xlabel("Application cohort")
    fig.suptitle(f"Historical model strategy comparison | as of {as_of}",
                 x=0.02, ha="left", fontsize=17, fontweight="bold", color="#123C4B")
    chart = REPORTS_DIR / f"monitoring_as_of_{as_of}.png"
    fig.savefig(chart, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    all_details = []
    for directory in [FEATURE_MONITORING_DIR, FIXED_FEATURE_MONITORING_DIR]:
        all_details.append(pd.read_parquet(directory / f"as_of_date={as_of}" / "metrics.parquet"))
    detail = pd.concat(all_details, ignore_index=True)
    detail.to_csv(REPORTS_DIR / f"feature_monitoring_as_of_{as_of}.csv", index=False)
    monthly_detail = detail[detail["scoring_strategy"].eq("monthly_selected")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout="constrained")
    for ax, value, title, cmap in zip(axes, ["csi", "missing_rate"],
                                    ["CSI across all 16 model inputs", "Missing rates before imputation"],
                                    ["YlOrRd", "Blues"]):
        pivot = monthly_detail.pivot(index="feature_name", columns="cohort_date", values=value).reindex(MONITORED_FEATURES)
        heat = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=0,
                         vmax=max(0.10, float(pivot.max().max())))
        ax.set_yticks(range(len(pivot)), pivot.index, fontsize=9)
        ax.set_xticks(range(len(pivot.columns)), pd.to_datetime(pivot.columns).strftime("%b %y"),
                      rotation=35, ha="right", fontsize=9)
        ax.set_title(title, loc="left", fontsize=12)
        for row in range(len(pivot)):
            for col in range(len(pivot.columns)):
                number = pivot.iloc[row, col]
                ax.text(col, row, f"{number:.2f}" if value == "csi" else f"{number:.0%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if number > heat.norm.vmax * 0.6 else "#173847")
        fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.03)
    fig.suptitle(f"Monthly version input monitoring | as of {as_of}", fontsize=16, color="#123C4B")
    feature_chart = REPORTS_DIR / f"feature_monitoring_as_of_{as_of}.png"
    fig.savefig(feature_chart, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    business = business_view(as_of)
    if not business.empty:
        business.to_csv(REPORTS_DIR / f"business_view_as_of_{as_of}.csv", index=False)
    tuning = tuning_summary(as_of)
    if not tuning.empty:
        tuning.to_csv(REPORTS_DIR / f"tuning_results_as_of_{as_of}.csv", index=False)
    return {"as_of_date": as_of, "chart": str(chart), "feature_chart": str(feature_chart),
            "cohorts": int(len(data)), "mature_cohorts": int(data["roc_auc"].notna().sum()),
            "monitored_features": len(MONITORED_FEATURES)}
