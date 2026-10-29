"""Local entry point; the Airflow DAG calls the same functions."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from utils.data_pipeline import prepare_datamart
from utils.model_pipeline import monitor_as_of, score_cohort, train_and_select
from utils.reporting import render_monitoring


def run_one(as_of: str, force_train: bool = False) -> dict:
    prepare_datamart()
    trained = train_and_select(as_of, force=force_train)
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
    parser.add_argument("--force-train", action="store_true",
                        help="Rebuild saved models; use with backfill to refresh downstream outputs")
    args = parser.parse_args()
    if args.force_train and args.command not in {"train", "run", "backfill"}:
        parser.error("--force-train applies only to train, run, or backfill")
    if args.command == "prepare":
        result = prepare_datamart()
    elif args.command == "backfill":
        result = []
        for date in pd.date_range(args.start, args.end, freq="MS"):
            row = run_one(date.strftime("%Y-%m-%d"), force_train=args.force_train)
            result.append({"as_of_date": date.strftime("%Y-%m-%d"),
                           "model_status": row["trained"]["status"],
                           "scored_rows": row["scored"]["rows"],
                           "mature_cohorts": row["monitored"]["mature_cohorts"]})
            print(json.dumps(result[-1]), flush=True)
    else:
        if not args.as_of:
            parser.error("--as-of is required")
        if args.command == "train":
            result = train_and_select(args.as_of, force=args.force_train)
        elif args.command == "run":
            result = run_one(args.as_of, force_train=args.force_train)
        else:
            result = {
                "score": score_cohort,
                "monitor": monitor_as_of,
                "report": render_monitoring,
            }[args.command](args.as_of)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
