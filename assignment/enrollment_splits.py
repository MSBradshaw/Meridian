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
    Prior care patients with a claim line from any of
               `exclude_prior_specialties` dated before their index date are
               dropped from every split, holdout included. Default: Physical
               Therapy for PT (PT program sessions are billed as PT-specialty
               claims, and patients already in PT care are not PT prospects);
               nothing for WM.
    Features   build them with `ignore_codes=DEFAULT_IGNORE_CODES[program]`: for
               PT this also drops the PT referral codes (97001, 97110).

    from assignment.enrollment_splits import write_enrollment_splits
    path = write_enrollment_splits(lookback_quarters=4, program="PT")
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from assignment.features import PT_REFERRAL_CODES
from assignment.splits import EXTRACT_END, SURGERY_CODES, _events, _window, finish_splits, load_tables

PROGRAMS = ("PT", "WM")
REFUSAL_RESULTS = ("Declined", "Opted Out", "Unsubscribed")
DEFAULT_EXCLUDE_PRIOR_SPECIALTIES = {"PT": ("Physical Therapy",), "WM": ()}
# Codes to leave out of each program's features (pass as `ignore_codes` to
# `assignment.features.write_features`): surgery codes always; for PT also the
# PT referral codes, so no PT activity at all feeds the PT model. (The surgical
# consult codes are always removed by `features` itself.)
DEFAULT_IGNORE_CODES = {"PT": (*SURGERY_CODES, *PT_REFERRAL_CODES), "WM": tuple(SURGERY_CODES)}
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
    exclude_prior_specialties: Iterable[str] | None = None,
    holdout_recent_quarters: int = 1,
    extract_end: pd.Timestamp = EXTRACT_END,
) -> pd.DataFrame:
    """One row per patient in train/val/test/holdout for the `program` enrollment task.

    `exclude_prior_specialties` defaults to `DEFAULT_EXCLUDE_PRIOR_SPECIALTIES[program]`.
    """
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
    splits = finish_splits(labelled, patients, events, any_surgery, lookback_quarters, holdout_recent_quarters, extract_end)

    if exclude_prior_specialties is None:
        exclude_prior_specialties = DEFAULT_EXCLUDE_PRIOR_SPECIALTIES[program]
    prior = prior_specialty_patients(splits, claims, exclude_prior_specialties)
    return splits[~splits["patient_token"].isin(prior)].reset_index(drop=True)


def prior_specialty_patients(splits: pd.DataFrame, claims: pd.DataFrame, specialties: Iterable[str]) -> set:
    """Patients in `splits` with a claim line from any of `specialties` dated before their index date."""
    specialties = set(specialties)
    if not specialties:
        return set()
    lines = claims.loc[claims["provider_specialty"].isin(specialties), ["patient_token", "service_date"]]
    joined = lines.merge(splits[["patient_token", "index_date"]], on="patient_token")
    return set(joined.loc[joined["service_date"] < joined["index_date"], "patient_token"])


def enrollment_splits_filename(
    program: str,
    lookback_quarters: int,
    match_period: bool = True,
    exclude_prior_specialties: Iterable[str] | None = None,
) -> str:
    """e.g. `patient_windows_enroll-PT_noprior-physical_therapy_lookback4q.csv` (PT defaults)."""
    if exclude_prior_specialties is None:
        exclude_prior_specialties = DEFAULT_EXCLUDE_PRIOR_SPECIALTIES[program]
    period = "" if match_period else "_allperiod"
    slugs = sorted(s.lower().replace(" ", "_") for s in exclude_prior_specialties)
    prior = f"_noprior-{'-'.join(slugs)}" if slugs else ""
    return f"patient_windows_enroll-{program}{period}{prior}_lookback{lookback_quarters}q.csv"


def write_enrollment_splits(
    lookback_quarters: int,
    program: str,
    match_period: bool = True,
    exclude_prior_specialties: Iterable[str] | None = None,
    data_dir: Path = Path("data"),
    out_dir: Path = SPLITS_DIR,
) -> Path:
    """Build the enrollment splits for `program` and write them to `out_dir`; returns the file path."""
    splits = build_enrollment_splits(
        **load_program_tables(data_dir),
        program=program,
        lookback_quarters=lookback_quarters,
        match_period=match_period,
        exclude_prior_specialties=exclude_prior_specialties,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / enrollment_splits_filename(program, lookback_quarters, match_period, exclude_prior_specialties)
    splits.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
