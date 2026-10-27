"""Entrypoint.

    uv run main.py train
    uv run main.py score
"""

import argparse
from pathlib import Path

import pandas as pd

from assignment.inference import funnel, run_inference
from assignment.pipeline import train_all
from assignment.utils import validate_targets

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
OUT_DIR = Path("out")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Fit on data/, write model artifacts to artifacts/.")
    sub.add_parser("score", help="Score patients, write out/targets.csv.")
    args = parser.parse_args()

    if args.command == "train":
        # Surgery-risk, PT- and WM-enrollment models with the default settings
        # (see assignment/pipeline.py and DECISIONS.md), each scored on val and test.
        summary = train_all(DATA_DIR, ARTIFACTS_DIR)
        print(summary.round(3).to_string(index=False))
        print(f"\nwrote {ARTIFACTS_DIR / 'manifest.json'} and {ARTIFACTS_DIR / 'training_summary.csv'}")
    else:
        # Holdout -> surgery model -> PT/WM models on surgery positives -> eligibility
        # rules -> list ordered by surgery score (see assignment/inference.py).
        targets: pd.DataFrame
        targets, detail = run_inference(DATA_DIR, ARTIFACTS_DIR)
        validate_targets(targets)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        targets.to_csv(OUT_DIR / "targets.csv", index=False)
        detail.to_csv(OUT_DIR / "targets_detail.csv", index=False)
        stats = pd.concat([
            funnel(detail),
            detail[["surgery_threshold", "PT_threshold", "WM_threshold"]].iloc[0].round(4),
        ]).rename("value").rename_axis("step")
        stats.to_csv(OUT_DIR / "targets_summary.csv")
        print(stats.to_string())
        print(f"\nwrote {OUT_DIR / 'targets.csv'} ({len(targets):,} patients), "
              f"{OUT_DIR / 'targets_detail.csv'} and {OUT_DIR / 'targets_summary.csv'}")


if __name__ == "__main__":
    cli()
