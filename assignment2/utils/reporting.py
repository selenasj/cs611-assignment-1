"""Static monitoring evidence for the datamart and slideument."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import PROJECT_ROOT
from utils.model_pipeline import MONITORING_DIR


REPORTS_DIR = PROJECT_ROOT / "reports"


def render_monitoring(as_of: str) -> dict:
    source = MONITORING_DIR / f"as_of_date={as_of}" / "metrics.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Monitoring Gold table is missing: {source}")
    data = pd.read_parquet(source).sort_values("cohort_date").copy()
    data["cohort_label"] = pd.to_datetime(data["cohort_date"]).dt.strftime("%b %Y")
    data["max_csi"] = data[["csi_age", "csi_annual_income", "csi_debt_to_annual_income"]].max(axis=1)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(REPORTS_DIR / f"monitoring_as_of_{as_of}.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12.4, 7.0), sharex=True, layout="constrained")
    fig.patch.set_facecolor("#F9FBFC")
    x = np.arange(len(data))
    observed = data["roc_auc"].notna()
    axes[0].plot(x[observed], data.loc[observed, "roc_auc"], color="#006B73", marker="o", linewidth=2.2,
                 label="Observed ROC AUC")
    axes[0].plot(x, data["reference_validation_auc"], color="#7D8790", linestyle="--", linewidth=1.5,
                 label="Model validation AUC")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower left", frameon=False, ncol=2)
    axes[0].set_title("Performance is shown only after the MOB 6 outcome is observed", loc="left", fontsize=12)

    axes[1].plot(x, data["psi_score"], color="#BA5D36", marker="o", linewidth=2.2, label="Prediction PSI")
    axes[1].plot(x, data["max_csi"], color="#395B9A", marker="s", linewidth=2.0, label="Highest monitored feature CSI")
    axes[1].axhline(0.1, color="#D9A441", linestyle=":", linewidth=1.4, label="PSI 0.10 review")
    axes[1].axhline(0.25, color="#9A354A", linestyle="--", linewidth=1.4, label="PSI 0.25 alert")
    axes[1].set_ylabel("Distribution shift index")
    axes[1].set_xlabel("Application cohort")
    axes[1].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].set_title("Stability uses each model's training distribution as its reference", loc="left", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(data["cohort_label"], rotation=35, ha="right")
    for ax in axes:
        ax.set_facecolor("#F9FBFC")
        ax.grid(axis="y", color="#DCE5E9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Loan default model monitoring | information available by {as_of}",
                 x=0.02, ha="left", fontsize=17, fontweight="bold", color="#122C3A")
    destination = REPORTS_DIR / f"monitoring_as_of_{as_of}.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"as_of_date": as_of, "chart": str(destination), "cohorts": int(len(data)),
            "mature_cohorts": int(observed.sum())}
