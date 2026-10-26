"""Entrypoint.

    uv run main.py train
    uv run main.py score
"""

import argparse
from pathlib import Path

import pandas as pd

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
        # Call your code here ...
        ...
    else:
        # Call your code here ...
        targets: pd.DataFrame = ...
        validate_targets(targets)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        targets.to_csv(OUT_DIR / "targets.csv", index=False)


if __name__ == "__main__":
    cli()
