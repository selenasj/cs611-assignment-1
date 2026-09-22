import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.feature_policy import (
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_NUMERIC_FEATURES,
    CLICKSTREAM_FEATURES,
    EXCLUDED_FROM_SANITY_MODEL,
)


OUTPUT_DIR = Path("outputs/slide_assets")
BLUE = "#1F4E79"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#D95D5D"
GREY = "#64748B"
LIGHT_GREY = "#E2E8F0"


def apply_chart_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": LIGHT_GREY,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_horizontal_bars(ax: plt.Axes, bars, values) -> None:
    maximum = max(values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            color="#0F172A",
            fontweight="bold",
        )


def build_inventory_assets() -> None:
    inventory = pd.DataFrame(
        {
            "source": [
                "Customer attributes",
                "Customer financials",
                "Clickstream",
                "Loan snapshots",
            ],
            "rows": [12_500, 12_500, 215_376, 137_500],
            "columns": [6, 22, 22, 11],
            "date_range": [
                "2023-01 to 2025-01",
                "2023-01 to 2025-01",
                "2023-01 to 2024-12",
                "2023-01 to 2025-11",
            ],
        }
    )
    inventory.to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)

    plot_data = inventory.sort_values("rows")
    fig, ax = plt.subplots(figsize=(10, 5.625))
    bars = ax.barh(plot_data["source"], plot_data["rows"], color=[TEAL, TEAL, BLUE, BLUE])
    label_horizontal_bars(ax, bars, plot_data["rows"].tolist())
    ax.set_title("Four raw sources feed the application-time feature store", loc="left")
    ax.set_xlabel("Rows")
    ax.set_xlim(0, plot_data["rows"].max() * 1.18)
    ax.tick_params(axis="y", length=0)
    save_figure(fig, "data_inventory.png")


def build_quality_assets(features: pd.DataFrame) -> None:
    quality = pd.DataFrame(
        {
            "metric": [
                "Missing clickstream coverage",
                "Invalid age",
                "Unknown occupation",
                "Financial parse error",
                "Financial domain warning",
                "Income consistency warning",
            ],
            "rows": [
                int((features["clickstream_available"] == 0).sum()),
                int((features["age_invalid_flag"] == 1).sum()),
                int((features["occupation_unknown_flag"] == 1).sum()),
                int((features["financial_parse_error_flag"] == 1).sum()),
                int((features["financial_domain_warning_flag"] == 1).sum()),
                int((features["income_consistency_warning_flag"] == 1).sum()),
            ],
        }
    ).sort_values("rows")
    quality["share"] = quality["rows"] / len(features)
    quality.to_csv(OUTPUT_DIR / "quality_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.625))
    colors = [GREY if "warning" in value.lower() else ORANGE for value in quality["metric"]]
    bars = ax.barh(quality["metric"], quality["rows"], color=colors)
    label_horizontal_bars(ax, bars, quality["rows"].tolist())
    ax.set_title("Quality issues are preserved as auditable flags", loc="left")
    ax.set_xlabel("Affected applications")
    ax.set_xlim(0, quality["rows"].max() * 1.2)
    ax.tick_params(axis="y", length=0)
    save_figure(fig, "data_quality.png")

    parse_flag_columns = [
        column for column in features.columns if column.endswith("_parse_error_flag")
    ]
    parse_details = pd.DataFrame(
        {
            "field": [column.removesuffix("_parse_error_flag") for column in parse_flag_columns],
            "parse_error_rows": [int(features[column].fillna(0).sum()) for column in parse_flag_columns],
        }
    ).sort_values("parse_error_rows", ascending=False)
    parse_details.to_csv(OUTPUT_DIR / "financial_parse_details.csv", index=False)


def build_coverage_assets(features: pd.DataFrame) -> None:
    cohort = (
        features.assign(feature_as_of_date=pd.to_datetime(features["feature_as_of_date"]))
        .groupby(["feature_as_of_date", "clickstream_available"], observed=False)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "clickstream_missing", 1: "clickstream_available"})
        .reset_index()
    )
    for column in ["clickstream_missing", "clickstream_available"]:
        if column not in cohort:
            cohort[column] = 0
    cohort.to_csv(OUTPUT_DIR / "clickstream_coverage_by_cohort.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.625))
    x = np.arange(len(cohort))
    ax.bar(x, cohort["clickstream_available"], color=TEAL, label="Available")
    ax.bar(
        x,
        cohort["clickstream_missing"],
        bottom=cohort["clickstream_available"],
        color=ORANGE,
        label="Missing",
    )
    ax.axvline(17.5, color=RED, linestyle="--", linewidth=1.5)
    ax.text(17.7, ax.get_ylim()[1] * 0.92, "Coverage boundary", color=RED)
    labels = cohort["feature_as_of_date"].dt.strftime("%Y-%m")
    tick_positions = list(range(0, len(cohort), 3))
    ax.set_xticks(tick_positions, labels.iloc[tick_positions], rotation=45, ha="right")
    ax.set_title("Clickstream missingness is a time-cohort coverage change", loc="left")
    ax.set_ylabel("Applications")
    ax.set_xlabel("Application cohort")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save_figure(fig, "clickstream_coverage_by_cohort.png")


def build_label_assets(labels: pd.DataFrame) -> None:
    counts = labels["label"].value_counts().sort_index()
    label_summary = pd.DataFrame(
        {
            "label": ["No 30+ DPD at MOB6", "30+ DPD at MOB6"],
            "rows": [int(counts.get(0, 0)), int(counts.get(1, 0))],
        }
    )
    label_summary["share"] = label_summary["rows"] / label_summary["rows"].sum()
    label_summary.to_csv(OUTPUT_DIR / "label_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.625))
    bars = ax.bar(label_summary["label"], label_summary["rows"], color=[BLUE, ORANGE], width=0.58)
    for bar, row, share in zip(bars, label_summary["rows"], label_summary["share"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 180,
            f"{row:,}\n({share:.1%})",
            ha="center",
            fontweight="bold",
        )
    ax.set_title("MOB6 30+ DPD proxy label is moderately imbalanced", loc="left")
    ax.set_ylabel("Loans")
    ax.set_ylim(0, label_summary["rows"].max() * 1.18)
    ax.tick_params(axis="x", length=0)
    save_figure(fig, "label_distribution.png")


def build_model_assets() -> None:
    metrics = json.loads((OUTPUT_DIR / "model_sanity_metrics.json").read_text(encoding="utf-8"))
    metric_values = pd.DataFrame(
        {
            "metric": ["ROC AUC", "Accuracy", "Precision", "Recall", "F1"],
            "value": [
                metrics["oot_roc_auc"],
                metrics["oot_accuracy"],
                metrics["oot_precision"],
                metrics["oot_recall"],
                metrics["oot_f1"],
            ],
        }
    )
    metric_values.to_csv(OUTPUT_DIR / "model_sanity_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.625))
    bars = ax.bar(metric_values["metric"], metric_values["value"], color=[BLUE, TEAL, GREY, ORANGE, BLUE])
    for bar, value in zip(bars, metric_values["value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("OOT score")
    ax.set_title("Simple temporal OOT model confirms ML-compatible Gold tables", loc="left")
    ax.text(
        0,
        -0.18,
        f"Train: {metrics['train_rows']:,} applications through 2024-06 | "
        f"OOT: {metrics['oot_rows']:,} later applications | compatibility check only",
        transform=ax.transAxes,
        color=GREY,
    )
    ax.tick_params(axis="x", length=0)
    save_figure(fig, "model_sanity_metrics.png")


def build_feature_policy_table() -> None:
    rows = []
    for feature in APPROVED_NUMERIC_FEATURES:
        rows.append(
            {
                "feature": feature,
                "status": "approved_sanity_model",
                "reason": "application-time numeric feature available across cohorts",
            }
        )
    for feature in APPROVED_CATEGORICAL_FEATURES:
        rows.append(
            {
                "feature": feature,
                "status": "approved_future_model",
                "reason": "requires encoder fitted on training data",
            }
        )
    for feature in CLICKSTREAM_FEATURES:
        rows.append(
            {
                "feature": feature,
                "status": "conditional",
                "reason": "available only for early application cohorts",
            }
        )
    for feature, reason in EXCLUDED_FROM_SANITY_MODEL.items():
        if feature in APPROVED_CATEGORICAL_FEATURES or feature in CLICKSTREAM_FEATURES:
            continue
        rows.append({"feature": feature, "status": "excluded", "reason": reason})
    pd.DataFrame(rows).drop_duplicates("feature").to_csv(
        OUTPUT_DIR / "feature_policy.csv", index=False
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_chart_style()

    features = pd.read_parquet("datamart/gold/feature_store")
    labels = pd.read_parquet("datamart/gold/label_store")

    build_inventory_assets()
    build_quality_assets(features)
    build_coverage_assets(features)
    build_label_assets(labels)
    build_model_assets()
    build_feature_policy_table()

    print(f"Generated slide assets in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

