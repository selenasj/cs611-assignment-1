"""Local entry point; the Airflow DAG calls the same functions."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from utils.data_pipeline import prepare_datamart
from utils.model_pipeline import monitor_as_of, score_cohort, train_and_select
from utils.reporting import render_monitoring


def run_one(as_of: str) -> dict:
    prepare_datamart()
    trained = train_and_select(as_of)
    scored = score_cohort(as_of)
    monitored = monitor_as_of(as_of)
    report = render_monitoring(as_of)
    return {"trained": trained, "scored": scored, "monitored": monitored, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="CS611 Assignment 2 pipeline")
    parser.add_argument("command", choices=["prepare", "train", "score", "monitor", "report", "run", "backfill"])
    parser.add_argument("--as-of", help="First day of month, YYYY-MM-DD")
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2025-08-01")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_datamart()
    elif args.command == "backfill":
        result = []
        for date in pd.date_range(args.start, args.end, freq="MS"):
            row = run_one(date.strftime("%Y-%m-%d"))
            result.append({"as_of_date": date.strftime("%Y-%m-%d"),
                           "model_status": row["trained"]["status"],
                           "scored_rows": row["scored"]["rows"],
                           "mature_cohorts": row["monitored"]["mature_cohorts"]})
            print(json.dumps(result[-1]), flush=True)
    else:
        if not args.as_of:
            parser.error("--as-of is required")
        result = {
            "train": train_and_select,
            "score": score_cohort,
            "monitor": monitor_as_of,
            "report": render_monitoring,
            "run": run_one,
        }[args.command](args.as_of)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
