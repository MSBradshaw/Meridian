"""Splits for predicting program enrollment (PT or WM), one cohort per program.

Writes the same splits format as `assignment.splits` (`patient_token, split,
label, index_date, start_date, end_date`), so `assignment.features` and
`assignment.train` work on it unchanged.

    Positives  patients whose first enrollment in the program came before
               their first surgery-code date, or who never had surgery.
               Index date = that first enrollment date.
    Negatives  patients contacted for the program whose attempt ended in a
               refusal (`REFUSAL_RESULTS`: Declined, Opted Out, Unsubscribed),
               who never enrolled in the program, and whose first refusal came
               before any surgery. Index date = the first refusal date.
    Period     with `match_period` (default), positives are kept only if their
               index date falls within the negatives' index-date range.
               Negatives only exist while outreach ran (2024-07 to 2026-02);
               without this, early positives would have windows truncated by
               the extract start that negatives never have, and a model could
               learn "short history = positive".
    Window     the `lookback_quarters` quarters before the index date, ending
               the day before it, as in `assignment.splits`.
    Splits     train/val/test by the same token buckets as `assignment.splits`;
               holdout by the same rules (patients table, active coverage,
               recent data, no surgery evidence, not labelled).

    from assignment.enrollment_splits import write_enrollment_splits
    path = write_enrollment_splits(lookback_quarters=4, program="PT")
"""

from pathlib import Path

import pandas as pd

from assignment.splits import EXTRACT_END, _events, _window, finish_splits, load_tables

PROGRAMS = ("PT", "WM")
REFUSAL_RESULTS = ("Declined", "Opted Out", "Unsubscribed")
SPLITS_DIR = Path("artifacts/splits")


def load_program_tables(data_dir: Path = Path("data")) -> dict[str, pd.DataFrame]:
    """`assignment.splits.load_tables` plus enrollments and outreach."""
    return {
        **load_tables(data_dir),
        "enrollments": pd.read_csv(
            data_dir / "meridian_health_enrollments.csv",
            usecols=["patient_token", "program", "enrollment_date"],
            dtype=str,
            parse_dates=["enrollment_date"],
        ),
        "outreach": pd.read_csv(
            data_dir / "meridian_health_outreach.csv",
            usecols=["patient_token", "timestamp", "program", "attempt_result"],
            dtype=str,
            parse_dates=["timestamp"],
        ),
    }


def _before_surgery(first_dates: pd.Series, first_surgery: pd.Series) -> pd.Series:
    """Keep patients whose date is before their first surgery, or who never had surgery."""
    surgery = first_dates.index.to_series().map(first_surgery)
    return first_dates[surgery.isna() | (first_dates < surgery)]


def build_enrollment_splits(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    claims: pd.DataFrame,
    enrollments: pd.DataFrame,
    outreach: pd.DataFrame,
    program: str,
    lookback_quarters: int,
    match_period: bool = True,
    holdout_recent_quarters: int = 1,
    extract_end: pd.Timestamp = EXTRACT_END,
) -> pd.DataFrame:
    """One row per patient in train/val/test/holdout for the `program` enrollment task."""
    if program not in PROGRAMS:
        raise ValueError(f"program must be one of {PROGRAMS}")
    if lookback_quarters < 1:
        raise ValueError("lookback_quarters must be >= 1")

    events = _events(encounters, claims)
    first_surgery = events[events["is_surgery_code"]].groupby("patient_token")["date"].min()
    any_surgery = set(events.loc[events["is_surgery_any"], "patient_token"])
    split_pool = set(patients["patient_token"].dropna()) | set(claims["patient_token"].dropna())

    enrolled = enrollments[(enrollments["program"] == program) & enrollments["patient_token"].isin(split_pool)]
    first_enrollment = enrolled.groupby("patient_token")["enrollment_date"].min()
    positives = _before_surgery(first_enrollment, first_surgery)

    refusals = outreach[
        (outreach["program"] == program)
        & outreach["attempt_result"].isin(REFUSAL_RESULTS)
        & outreach["patient_token"].isin(split_pool)
        & ~outreach["patient_token"].isin(set(enrolled["patient_token"]))
    ]
    first_refusal = refusals.groupby("patient_token")["timestamp"].min().dt.normalize()
    negatives = _before_surgery(first_refusal, first_surgery)

    if match_period and not negatives.empty:
        positives = positives[positives.between(negatives.min(), negatives.max())]

    pos = _window(pd.Index(sorted(positives.index)), positives, lookback_quarters).assign(label=1)
    neg = _window(pd.Index(sorted(negatives.index)), negatives, lookback_quarters).assign(label=0)
    labelled = pd.concat([pos, neg], ignore_index=True)
    return finish_splits(labelled, patients, events, any_surgery, lookback_quarters, holdout_recent_quarters, extract_end)


def enrollment_splits_filename(program: str, lookback_quarters: int, match_period: bool = True) -> str:
    period = "" if match_period else "_allperiod"
    return f"patient_windows_enroll-{program}{period}_lookback{lookback_quarters}q.csv"


def write_enrollment_splits(
    lookback_quarters: int,
    program: str,
    match_period: bool = True,
    data_dir: Path = Path("data"),
    out_dir: Path = SPLITS_DIR,
) -> Path:
    """Build the enrollment splits for `program` and write them to `out_dir`; returns the file path."""
    splits = build_enrollment_splits(
        **load_program_tables(data_dir), program=program, lookback_quarters=lookback_quarters, match_period=match_period
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / enrollment_splits_filename(program, lookback_quarters, match_period)
    splits.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
