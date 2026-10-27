"""Task labels for a splits file, so one splits + features set can train several tasks.

A splits file (`assignment.splits`) fixes the patients, their split and their
index date; its own `label` column is the surgery label. This module builds a
labels file, `patient_token, label`, for another task on the same patients.
`assignment.train` swaps it in with `labels_file=`.

Program enrollment (`enrollment_labels`), one binary task per program:

    label 1    enrolled in the program on a date in [index_date, index_date +
               `horizon_quarters` quarters)
    label 0    not enrolled in that period
    missing    (excluded from training and evaluation)
               - holdout patients: no outcome to learn from
               - already enrolled in the program before the index date: the
                 task is predicting new enrollment
               - follow-up period runs past the extract end, so a 0 could be
                 an enrollment we cannot see yet

    from assignment.labels import write_enrollment_labels
    path = write_enrollment_labels(lookback_quarters=4, program="PT", horizon_quarters=4)

As a script (both programs):

    uv run python -m assignment.labels --lookback-quarters 4 --programs PT WM --horizon-quarters 4
"""

import argparse
from pathlib import Path

import pandas as pd

from assignment.features import SPLITS_DIR, read_splits
from assignment.splits import EXTRACT_END, HOLDOUT, splits_filename, write_splits

PROGRAMS = ("PT", "WM")
LABELS_DIR = Path("artifacts/labels")


def load_enrollments(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(
        data_dir / "meridian_health_enrollments.csv",
        usecols=["patient_token", "program", "enrollment_date"],
        dtype=str,
        parse_dates=["enrollment_date"],
    )


def enrollment_labels(
    splits: pd.DataFrame,
    enrollments: pd.DataFrame,
    program: str,
    horizon_quarters: int = 4,
    extract_end: pd.Timestamp = EXTRACT_END,
) -> pd.DataFrame:
    """`patient_token, label` for "enrolls in `program` within `horizon_quarters` of the index date".

    One row per patient in `splits`, in order; `label` is Int8 with missing for
    excluded patients (see module docstring).
    """
    if program not in PROGRAMS:
        raise ValueError(f"program must be one of {PROGRAMS}")
    if horizon_quarters < 1:
        raise ValueError("horizon_quarters must be >= 1")

    rows = splits[["patient_token", "split", "index_date"]].reset_index(drop=True)
    horizon_end = rows["index_date"] + pd.DateOffset(months=3 * horizon_quarters)  # exclusive
    dates = enrollments.loc[enrollments["program"] == program, ["patient_token", "enrollment_date"]]
    joined = dates.merge(rows[["patient_token", "index_date"]].assign(horizon_end=horizon_end), on="patient_token")

    prior = set(joined.loc[joined["enrollment_date"] < joined["index_date"], "patient_token"])
    in_horizon = joined["enrollment_date"].between(joined["index_date"], joined["horizon_end"], inclusive="left")
    enrolled = set(joined.loc[in_horizon, "patient_token"])

    label = rows["patient_token"].isin(enrolled).astype("Int8")
    excluded = (
        (rows["split"] == HOLDOUT)
        | rows["patient_token"].isin(prior)
        | (horizon_end - pd.Timedelta(days=1) > extract_end)
    )
    label[excluded] = pd.NA
    return pd.DataFrame({"patient_token": rows["patient_token"], "label": label})


def task_name(program: str, horizon_quarters: int) -> str:
    return f"enroll-{program}_h{horizon_quarters}q"


def labels_filename(splits_file: Path | str, program: str, horizon_quarters: int) -> str:
    """`patient_windows_lookback4q_neg1x_seed0.csv` -> `labels_enroll-PT_h4q_lookback4q_neg1x_seed0.csv`."""
    stem = Path(splits_file).stem.removeprefix("patient_windows_")
    return f"labels_{task_name(program, horizon_quarters)}_{stem}.csv"


def task_from_labels_file(labels_file: Path | str) -> str:
    """`labels_enroll-PT_h4q_lookback4q_...csv` -> `enroll-PT_h4q`."""
    return Path(labels_file).stem.removeprefix("labels_").split("_lookback")[0]


def read_labels(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_token": str, "label": "Int8"})


def write_enrollment_labels(
    lookback_quarters: int,
    program: str,
    horizon_quarters: int = 4,
    splits_file: Path | None = None,
    data_dir: Path = Path("data"),
    out_dir: Path = LABELS_DIR,
) -> Path:
    """Build enrollment labels for a splits file and write them to `out_dir`; returns the path.

    `splits_file` defaults to the default-parameter splits file for
    `lookback_quarters`, generated if it does not exist yet.
    """
    if splits_file is None:
        splits_file = SPLITS_DIR / splits_filename(lookback_quarters)
        if not splits_file.exists():
            splits_file = write_splits(lookback_quarters, data_dir=data_dir, out_dir=SPLITS_DIR)
    labels = enrollment_labels(read_splits(splits_file), load_enrollments(data_dir), program, horizon_quarters)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / labels_filename(splits_file, program, horizon_quarters)
    labels.to_csv(path, index=False)
    return path


def label_summary(splits: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per split: patients, labelled, excluded, positives and positive rate."""
    joined = splits[["patient_token", "split"]].merge(labels, on="patient_token")
    grouped = joined.groupby("split")["label"]
    return pd.DataFrame(
        {
            "patients": grouped.size(),
            "labelled": grouped.count(),
            "excluded": grouped.size() - grouped.count(),
            "positives": grouped.sum(),
            "positive_rate": (grouped.sum() / grouped.count()).round(3),
        }
    ).reindex(["train", "val", "test", HOLDOUT])


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build program-enrollment labels for a splits file.")
    parser.add_argument("--lookback-quarters", type=int, required=True)
    parser.add_argument("--programs", nargs="+", choices=PROGRAMS, default=list(PROGRAMS))
    parser.add_argument("--horizon-quarters", type=int, default=4)
    parser.add_argument("--splits-file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=LABELS_DIR)
    args = parser.parse_args()

    for program in args.programs:
        path = write_enrollment_labels(
            args.lookback_quarters, program, args.horizon_quarters, args.splits_file, args.data_dir, args.out_dir
        )
        splits_file = args.splits_file or SPLITS_DIR / splits_filename(args.lookback_quarters)
        print(f"wrote {path}")
        print(label_summary(read_splits(splits_file), read_labels(path)).to_string())


if __name__ == "__main__":
    cli()
