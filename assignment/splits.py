"""Cohort and train/val/test/holdout splits for the surgery-risk model.

Each patient gets one row: which split it belongs to, its label, an index date,
and the window of data (`start_date`..`end_date`, inclusive) that features may
be built from. The window is the `lookback_quarters` quarters before the index
date, ending the day before it, so nothing on or after the index date leaks in.

    Positives  patients with any of the surgery CPT codes (encounters or claims,
               any status). Index date = first surgery-code date.
    Negatives  no surgery evidence at all (no surgery code, no `Spine Surgery`
               encounter, any status), >= `min_history_years` of observed data.
               Sampled at random, `neg_ratio` negatives per positive.
               Index date = drawn from the positives' index dates (restricted
               to dates inside the negative's observed data), so both classes
               have the same calendar timing and history truncation.
    Splits     positives and sampled negatives, assigned by the last two decimal
               digits of the patient token: 0-69 train, 70-84 val, 85-99 test.
    Holdout    patients-table patients with no surgery evidence that were not
               sampled as negatives, whose coverage is active at the extract
               end and who have an event in the last `holdout_recent_quarters`
               quarters: the population to score for real.
               Index date = the day after the extract end.

Train/val/test may draw on patients from the patients table and the claims
feed; the holdout is drawn from the patients table only.

    from assignment.splits import write_splits
    path = write_splits(lookback_quarters=4)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SURGERY_CODES = ["22630", "22633", "63030", "63047"]
SURGERY_CATEGORY = "Spine Surgery"
EXTRACT_END = pd.Timestamp("2026-06-30")

SPLIT_BUCKETS = {"train": (0, 70), "val": (70, 85), "test": (85, 100)}
HOLDOUT = "holdout"

OUTPUT_COLUMNS = ["patient_token", "split", "label", "index_date", "start_date", "end_date"]


def split_bucket(patient_token: str) -> int:
    """Last two decimal digits (0-99) of the hex patient token read as a number."""
    return int(patient_token, 16) % 100


def assign_split(patient_token: str) -> str:
    bucket = split_bucket(patient_token)
    for split, (lo, hi) in SPLIT_BUCKETS.items():
        if lo <= bucket < hi:
            return split
    raise AssertionError(f"bucket {bucket} not covered by SPLIT_BUCKETS")


def claims_path(data_dir: Path) -> Path:
    """The claims feed, shipped as .csv.gz per the data dictionary but sometimes unpacked."""
    for path in (data_dir / "data_web_claims.csv.gz", data_dir / "data_web_claims.csv"):
        if path.exists():
            return path
    raise FileNotFoundError(f"no data_web_claims.csv(.gz) in {data_dir}")


def load_tables(data_dir: Path = Path("data")) -> dict[str, pd.DataFrame]:
    """Read the columns this module needs from the patients, encounters and claims files."""
    return {
        "patients": pd.read_csv(
            data_dir / "meridian_health_patients.csv",
            usecols=["patient_token", "coverage_start", "coverage_end"],
            dtype=str,
            parse_dates=["coverage_start", "coverage_end"],
        ),
        "encounters": pd.read_csv(
            data_dir / "meridian_health_encounters.csv",
            usecols=["patient_token", "event_date", "event_category", "event_code"],
            dtype=str,
            parse_dates=["event_date"],
        ),
        "claims": pd.read_csv(
            claims_path(data_dir),
            usecols=["patient_token", "service_date", "code_system", "code"],
            dtype=str,
            parse_dates=["service_date"],
        ),
    }


def _events(encounters: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Encounters and claims stacked into one long frame of dated events."""
    enc = pd.DataFrame(
        {
            "patient_token": encounters["patient_token"],
            "date": encounters["event_date"],
            "is_surgery_code": encounters["event_code"].isin(SURGERY_CODES),
            "is_surgery_any": encounters["event_code"].isin(SURGERY_CODES)
            | (encounters["event_category"] == SURGERY_CATEGORY),
        }
    )
    surgery_claim = (claims["code_system"] == "CPT") & claims["code"].isin(SURGERY_CODES)
    clm = pd.DataFrame(
        {
            "patient_token": claims["patient_token"],
            "date": claims["service_date"],
            "is_surgery_code": surgery_claim,
            "is_surgery_any": surgery_claim,
        }
    )
    return pd.concat([enc, clm], ignore_index=True).dropna(subset=["patient_token", "date"])


def _window(tokens: pd.Index, index_dates: pd.Series, lookback_quarters: int) -> pd.DataFrame:
    index_dates = pd.Series(index_dates.loc[tokens].to_numpy(), index=tokens)
    return pd.DataFrame(
        {
            "patient_token": tokens,
            "index_date": index_dates.to_numpy(),
            "start_date": (index_dates - pd.DateOffset(months=3 * lookback_quarters)).to_numpy(),
            "end_date": (index_dates - pd.Timedelta(days=1)).to_numpy(),
        }
    )


def _matched_index_dates(
    neg_pool: pd.DataFrame, positive_index: pd.Series, neg_ratio: float, seed: int
) -> pd.Series:
    """Sample negatives and give each an index date drawn from the positives' index dates.

    `neg_pool` has one row per candidate with its first (`min`) and last (`max`)
    observed event. A candidate can only take a positive index date that falls
    after its first event and no later than the day after its last event, so its
    window sits inside its own observed data. Candidates with no such date are
    not eligible. Returns index dates keyed by patient token.
    """
    pos_dates = np.sort(positive_index.to_numpy())
    lo = np.searchsorted(pos_dates, neg_pool["min"].to_numpy(), side="right")
    hi = np.searchsorted(pos_dates, (neg_pool["max"] + pd.Timedelta(days=1)).to_numpy(), side="right")
    eligible = np.flatnonzero(hi > lo)

    rng = np.random.default_rng(seed)
    n_neg = min(len(eligible), round(neg_ratio * len(pos_dates)))
    pick = np.sort(rng.choice(eligible, size=n_neg, replace=False))
    draw = lo[pick] + np.floor(rng.random(n_neg) * (hi[pick] - lo[pick])).astype(int)
    return pd.Series(pos_dates[draw], index=neg_pool.index[pick])


def build_splits(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    claims: pd.DataFrame,
    lookback_quarters: int,
    neg_ratio: float = 1.0,
    min_history_years: int = 2,
    seed: int = 0,
    holdout_recent_quarters: int = 1,
    extract_end: pd.Timestamp = EXTRACT_END,
) -> pd.DataFrame:
    """One row per patient in train/val/test/holdout with its label and data window.

    `label` is 1 for positives, 0 for negatives, and missing for the holdout.
    Patients that fit none of these (ambiguous surgery evidence, or holdout
    candidates without active coverage or recent data) are left out.
    """
    if lookback_quarters < 1:
        raise ValueError("lookback_quarters must be >= 1")

    events = _events(encounters, claims)
    patient_ids = set(patients["patient_token"].dropna())
    split_pool = patient_ids | set(claims["patient_token"].dropna())

    first_surgery = events[events["is_surgery_code"]].groupby("patient_token")["date"].min()
    any_surgery = set(events.loc[events["is_surgery_any"], "patient_token"])

    positives = pd.Index(sorted(set(first_surgery.index) & split_pool))

    span = events.groupby("patient_token")["date"].agg(["min", "max"])
    long_history = span[span["max"] >= span["min"] + pd.DateOffset(years=min_history_years)].index
    neg_pool = span.loc[sorted((set(long_history) & split_pool) - any_surgery)]
    neg_index = _matched_index_dates(neg_pool, first_surgery.loc[positives], neg_ratio, seed)

    pos = _window(positives, first_surgery, lookback_quarters).assign(label=1)
    neg = _window(neg_index.index, neg_index, lookback_quarters).assign(label=0)
    labelled = pd.concat([pos, neg], ignore_index=True)
    return finish_splits(
        labelled, patients, events, any_surgery, lookback_quarters, holdout_recent_quarters, extract_end
    )


def finish_splits(
    labelled: pd.DataFrame,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    any_surgery: set,
    lookback_quarters: int,
    holdout_recent_quarters: int = 1,
    extract_end: pd.Timestamp = EXTRACT_END,
) -> pd.DataFrame:
    """Assign train/val/test to labelled windows by token, add the holdout, and return the splits table.

    Holdout: patients-table patients with active coverage at the extract end, an
    event in the last `holdout_recent_quarters` quarters, no surgery evidence and
    not in `labelled`. Index date = the day after the extract end.
    """
    labelled = labelled.assign(split=labelled["patient_token"].map(assign_split))
    covered = patients[
        (patients["coverage_start"] <= extract_end)
        & (patients["coverage_end"].isna() | (patients["coverage_end"] >= extract_end))
    ]
    recent_start = extract_end + pd.Timedelta(days=1) - pd.DateOffset(months=3 * holdout_recent_quarters)
    recently_seen = set(events.loc[events["date"] >= recent_start, "patient_token"])
    holdout_pool = (set(covered["patient_token"]) & recently_seen) - any_surgery - set(labelled["patient_token"])
    holdout_ids = pd.Index(sorted(holdout_pool))
    holdout_index = pd.Series(extract_end + pd.Timedelta(days=1), index=holdout_ids)
    holdout = _window(holdout_ids, holdout_index, lookback_quarters).assign(label=pd.NA, split=HOLDOUT)

    out = pd.concat([labelled, holdout], ignore_index=True)[OUTPUT_COLUMNS]
    out["label"] = out["label"].astype("Int8")
    return out.sort_values(["split", "patient_token"], ignore_index=True)


def splits_filename(lookback_quarters: int, neg_ratio: float = 1.0, seed: int = 0) -> str:
    return f"patient_windows_lookback{lookback_quarters}q_neg{neg_ratio:g}x_seed{seed}.csv"


def write_splits(
    lookback_quarters: int,
    data_dir: Path = Path("data"),
    out_dir: Path = Path("artifacts/splits"),
    neg_ratio: float = 1.0,
    seed: int = 0,
) -> Path:
    """Build the splits from `data_dir` and write them to `out_dir`; returns the file path."""
    tables = load_tables(data_dir)
    splits = build_splits(**tables, lookback_quarters=lookback_quarters, neg_ratio=neg_ratio, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / splits_filename(lookback_quarters, neg_ratio, seed)
    splits.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
