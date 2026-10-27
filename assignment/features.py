"""Code-count features per patient, built inside each patient's splits window.

Input is a splits file from `assignment.splits` (one row per patient with
`split`, `label`, `start_date`, `end_date`). Output is one row per patient in
that file: `patient_token, split, label`, then one column per code in the
vocabulary, named `{code_system}_{code}` (e.g. `CPT_72148`, `ICD10CM_M54.50`,
`NDC_65862003900`). Every split has the same columns.

    Encoding   `count` (default): rows for the code in the window, across
               sources. `binary`: 1 if the code appears at all.
    Scaling    min-max on the code columns and `age`, with min/max fitted on
               train rows only and applied unchanged to val/test/holdout (no
               leakage). On by default; `scale=False` writes raw values.

    Vocabulary codes that at least `min_train_patients` (default 11, i.e. more
               than 10) train patients have in their window. Chosen on train
               only; val/test/holdout get exactly those columns.

    Codes      claims (ICD10CM, CPT, HCPCS, NDC) and encounters (ICD10CM, CPT),
               any claim/event status. A code counts if it appears on a date in
               [start_date, end_date], inclusive, in either source.
    Ignored    `ignore_codes` are dropped before encoding; by default the
               surgery codes, which define the label.
    Demographics  `age` (2026 - birth_year) and one-hot `sex_F/M/U` from the
               patients table, placed after the ID columns and always kept.
    Specialties  `specialty_<name>`: claim lines in the window per provider
               specialty, for all 15 specialties in the data dictionary; always
               kept, counted and min-max scaled like the codes.

As a module:

    from assignment.features import write_features, feature_coverage
    path = write_features(lookback_quarters=4)

As a script (builds the splits file first if it does not exist):

    uv run python -m assignment.features --lookback-quarters 4
"""

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from assignment.splits import HOLDOUT, SURGERY_CODES, claims_path, splits_filename, write_splits

SOURCES = ("claims", "encounters")
ID_COLUMNS = ["patient_token", "split", "label"]
SPLIT_ORDER = ["train", "val", "test", HOLDOUT]

MIN_TRAIN_PATIENTS = 11  # keep features more than 10 train patients have
ENCODINGS = {"count": "count", "binary": "onehot"}  # encoding -> filename tag

AGE_REFERENCE_YEAR = 2026
SEX_CATEGORIES = ("F", "M", "U")
DEMOGRAPHIC_COLUMNS = ["age", *(f"sex_{s}" for s in SEX_CATEGORIES)]

# Every provider_specialty value in the data dictionary; one count column each,
# kept whatever the vocabulary so the schema is stable across extracts.
SPECIALTIES = (
    "Orthopedic Surgery", "Neurosurgery", "Physical Therapy", "Pain Management", "Primary Care",
    "Rheumatology", "Radiology", "Emergency Medicine", "Anesthesiology", "Chiropractic",
    "Cardiology", "Nephrology", "Endocrinology", "Behavioral Health", "Other",
)  # fmt: skip
SPECIALTY_PREFIX = "specialty_"


def specialty_column(specialty: str) -> str:
    """`Orthopedic Surgery` -> `specialty_orthopedic_surgery`."""
    return SPECIALTY_PREFIX + specialty.lower().replace(" ", "_")

SPLITS_DIR = Path("artifacts/splits")
FEATURES_DIR = Path("artifacts/features")


def load_code_events(data_dir: Path = Path("data"), sources: Iterable[str] = SOURCES) -> pd.DataFrame:
    """Every dated event as `patient_token, date, code_system, code, provider_specialty`.

    `code` may be missing (such rows still count toward specialties);
    `provider_specialty` is only present on claims.
    """
    frames = []
    if "claims" in sources:
        claims = pd.read_csv(
            claims_path(data_dir),
            usecols=["patient_token", "service_date", "code_system", "code", "provider_specialty"],
            dtype=str,
            parse_dates=["service_date"],
        )
        frames.append(claims.rename(columns={"service_date": "date"}))
    if "encounters" in sources:
        encounters = pd.read_csv(
            data_dir / "meridian_health_encounters.csv",
            usecols=["patient_token", "event_date", "code_system", "event_code"],
            dtype=str,
            parse_dates=["event_date"],
        )
        frames.append(encounters.rename(columns={"event_date": "date", "event_code": "code"}))
    if not frames:
        raise ValueError(f"no known sources in {list(sources)}; choose from {SOURCES}")
    events = pd.concat(frames, ignore_index=True)
    return events.dropna(subset=["patient_token", "date"])


def load_demographics(data_dir: Path = Path("data")) -> pd.DataFrame:
    """`patient_token, sex, birth_year` from the patients table."""
    return pd.read_csv(
        data_dir / "meridian_health_patients.csv",
        usecols=["patient_token", "sex", "birth_year"],
        dtype={"patient_token": str, "sex": str, "birth_year": "Int64"},
    )


def demographic_features(splits: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """`age` and one-hot `sex_F/M/U`, one row per patient in `splits` order.

    age = AGE_REFERENCE_YEAR - birth_year. Patients without a birth year (e.g.
    claims-only patients) get the median age of the demographics table, which
    uses no labels. A missing or unknown-category sex leaves all sex columns 0.
    """
    demo = splits[["patient_token"]].merge(demographics, on="patient_token", how="left")
    age = (AGE_REFERENCE_YEAR - demo["birth_year"]).astype("Float64")
    age = age.fillna(float((AGE_REFERENCE_YEAR - demographics["birth_year"]).median()))
    out = pd.DataFrame({"age": age.astype(float).to_numpy()})
    for s in SEX_CATEGORIES:
        out[f"sex_{s}"] = (demo["sex"] == s).fillna(False).astype("int8").to_numpy()
    return out


def specialty_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c.startswith(SPECIALTY_PREFIX)]


def code_columns(features: pd.DataFrame) -> list[str]:
    other = {*ID_COLUMNS, *DEMOGRAPHIC_COLUMNS, *specialty_columns(features)}
    return [c for c in features.columns if c not in other]


def train_vocabulary(onehot: pd.DataFrame, splits: pd.DataFrame, min_train_patients: int) -> list[str]:
    """Features present for at least `min_train_patients` train patients."""
    train_ids = splits.loc[splits["split"] == "train", "patient_token"]
    if train_ids.empty:
        raise ValueError("splits has no train patients to choose a vocabulary from; pass `vocabulary`")
    train_patients = (onehot.loc[train_ids] > 0).sum()
    return sorted(train_patients[train_patients >= min_train_patients].index)


def build_features(
    splits: pd.DataFrame,
    events: pd.DataFrame,
    ignore_codes: Iterable[str] = SURGERY_CODES,
    min_train_patients: int = MIN_TRAIN_PATIENTS,
    vocabulary: Iterable[str] | None = None,
    demographics: pd.DataFrame | None = None,
    encoding: str = "count",
    specialties: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per patient in `splits` with a column per code in the vocabulary.

    `encoding="count"` (default): number of rows for the code in the patient's
    window, across all sources. `encoding="binary"`: 1 if the code appears, else 0.
    Values are unscaled; see `fit_minmax` / `apply_minmax`.

    The vocabulary is chosen on train only: codes that at least
    `min_train_patients` train patients have. Every split gets exactly those
    columns, so codes never (or rarely) seen in train are dropped. Pass
    `vocabulary` to reuse an existing one instead, e.g. when scoring.
    Patients with no vocabulary codes in their window keep a row of all zeros.

    If `demographics` (see `load_demographics`) is given, `age` and `sex_F/M/U`
    are added after the ID columns. They are always kept, whatever the vocabulary.

    If `specialties` (e.g. `SPECIALTIES`) is given, one `specialty_<name>` column
    per specialty counts the patient's claim lines in the window from a provider
    of that specialty (any code, or none), after the demographics. Always kept.
    Rows with an ignored code count toward neither codes nor specialties.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {list(ENCODINGS)}")
    windows = splits[["patient_token", "start_date", "end_date"]]
    events = events[~events["code"].isin(set(ignore_codes))]
    in_window = events.merge(windows, on="patient_token")
    in_window = in_window[in_window["date"].between(in_window["start_date"], in_window["end_date"])]

    coded = in_window.dropna(subset=["code_system", "code"])
    feature = (coded["code_system"] + "_" + coded["code"]).rename("feature")
    codes = pd.crosstab(coded["patient_token"], feature).reindex(splits["patient_token"], fill_value=0)
    if encoding == "binary":
        codes = codes.clip(upper=1)
    if vocabulary is None:
        vocabulary = train_vocabulary(codes, splits, min_train_patients)
    codes = codes.reindex(columns=sorted(vocabulary), fill_value=0).astype("int32")
    codes.columns.name = None
    parts = [splits[ID_COLUMNS].reset_index(drop=True)]
    if demographics is not None:
        parts.append(demographic_features(splits, demographics))
    if specialties is not None:
        parts.append(specialty_features(splits, in_window, list(specialties)))
    return pd.concat([*parts, codes.reset_index(drop=True)], axis=1)


def specialty_features(splits: pd.DataFrame, in_window: pd.DataFrame, specialties: list[str]) -> pd.DataFrame:
    """Claim-line counts per provider specialty, one row per patient in `splits` order."""
    if "provider_specialty" in in_window.columns:
        rows = in_window[in_window["provider_specialty"].isin(specialties)]
    else:
        rows = in_window.iloc[:0].assign(provider_specialty=pd.Series(dtype=str))
    counts = (
        pd.crosstab(rows["patient_token"], rows["provider_specialty"])
        .reindex(index=splits["patient_token"], columns=specialties, fill_value=0)
        .astype("int32")
    )
    counts.columns = [specialty_column(s) for s in specialties]
    return counts.reset_index(drop=True)


def scaled_columns(features: pd.DataFrame) -> list[str]:
    """Columns min-max scaled: age, specialty counts and code columns. Sex columns are already 0/1."""
    counted = {*code_columns(features), *specialty_columns(features)}
    return [c for c in features.columns if c == "age" or c in counted]


def fit_minmax(features: pd.DataFrame) -> pd.DataFrame:
    """Per scaled column, the min and max over **train rows only**, so nothing leaks from val/test/holdout."""
    train = features.loc[features["split"] == "train", scaled_columns(features)]
    if train.empty:
        raise ValueError("features has no train rows to fit scaling on")
    scaling = pd.DataFrame({"min": train.min(), "max": train.max()}).astype(float)
    scaling.index.name = "feature"
    return scaling


def apply_minmax(features: pd.DataFrame, scaling: pd.DataFrame) -> pd.DataFrame:
    """`(x - min) / (max - min)` with the given train min/max. Not clipped: rows outside
    train's range (only possible outside train) fall below 0 or above 1. A column
    constant in train has range 1, so it is only shifted."""
    out = features.copy()
    cols = list(scaling.index)
    span = (scaling["max"] - scaling["min"]).replace(0, 1.0)
    out[cols] = (features[cols].astype(float) - scaling["min"]) / span
    return out


def ignore_tag(ignore_codes: Iterable[str]) -> str:
    """Short filename tag for the ignored-code set."""
    codes = sorted(set(ignore_codes))
    if codes == sorted(SURGERY_CODES):
        return "excl-surgery"
    if not codes:
        return "excl-none"
    digest = hashlib.sha1(",".join(codes).encode()).hexdigest()[:8]
    return f"excl-{len(codes)}codes-{digest}"


def features_filename(
    splits_file: Path | str,
    ignore_codes: Iterable[str] = SURGERY_CODES,
    min_train_patients: int = MIN_TRAIN_PATIENTS,
    include_demographics: bool = True,
    encoding: str = "count",
    scale: bool = True,
    include_specialties: bool = True,
) -> str:
    """`patient_windows_lookback4q_neg1x_seed0.csv`
    -> `features_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain11_demo_spec.csv`."""
    stem = Path(splits_file).stem.removeprefix("patient_windows_")
    kind = ENCODINGS[encoding] + ("_minmax" if scale else "")
    extras = ("_demo" if include_demographics else "") + ("_spec" if include_specialties else "")
    return f"features_{kind}_{stem}_{ignore_tag(ignore_codes)}_mintrain{min_train_patients}{extras}.csv"


def sidecar_path(features_path: Path, name: str) -> Path:
    """`<features file>_<name>.csv`: vocabulary, scaling, coverage or summary."""
    return features_path.with_name(f"{features_path.stem}_{name}.csv")


def vocabulary_path(features_path: Path) -> Path:
    return sidecar_path(features_path, "vocabulary")


def read_vocabulary(path: Path | str) -> list[str]:
    return pd.read_csv(path)["feature"].tolist()


def read_scaling(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="feature")


def read_splits(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"patient_token": str, "split": str, "label": "Int8"},
        parse_dates=["index_date", "start_date", "end_date"],
    )


def write_features(
    lookback_quarters: int,
    splits_file: Path | None = None,
    data_dir: Path = Path("data"),
    out_dir: Path = FEATURES_DIR,
    ignore_codes: Iterable[str] = SURGERY_CODES,
    sources: Iterable[str] = SOURCES,
    min_train_patients: int = MIN_TRAIN_PATIENTS,
    include_demographics: bool = True,
    encoding: str = "count",
    scale: bool = True,
    include_specialties: bool = True,
) -> Path:
    """Build the feature table for a splits file and write it to `out_dir`; returns its path.

    `splits_file` defaults to the default-parameter splits file for `lookback_quarters`,
    which is generated if it does not exist yet. Written next to the features:

        _vocabulary.csv  feature, train_patients (patients with the code in train)
        _scaling.csv     feature, min, max fitted on train (if `scale`)
        _coverage.csv    `feature_coverage` of the unscaled features
        _summary.csv     `coverage_summary` of the unscaled features
    """
    if splits_file is None:
        splits_file = SPLITS_DIR / splits_filename(lookback_quarters)
        if not splits_file.exists():
            splits_file = write_splits(lookback_quarters, data_dir=data_dir, out_dir=SPLITS_DIR)
    ignore_codes = list(ignore_codes)
    features = build_features(
        read_splits(splits_file),
        load_code_events(data_dir, sources),
        ignore_codes,
        min_train_patients,
        demographics=load_demographics(data_dir) if include_demographics else None,
        encoding=encoding,
        specialties=SPECIALTIES if include_specialties else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / features_filename(
        splits_file, ignore_codes, min_train_patients, include_demographics, encoding, scale, include_specialties
    )
    train_patients = (features.loc[features["split"] == "train", code_columns(features)] > 0).sum()
    train_patients.rename_axis("feature").rename("train_patients").reset_index().to_csv(
        vocabulary_path(path), index=False
    )
    feature_coverage(features).to_csv(sidecar_path(path, "coverage"))
    coverage_summary(features).to_csv(sidecar_path(path, "summary"))
    if scale:
        scaling = fit_minmax(features)
        scaling.to_csv(sidecar_path(path, "scaling"))
        features = apply_minmax(features, scaling)
    features.to_csv(path, index=False, float_format="%.6g")
    return path


def feature_coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Per feature: share of patients with the code in each split, plus counts and flags.

    Columns: `pct_<split>` (0-100) for train/val/test/holdout, `patients_total`,
    and `missing_in_train` (seen in val/test/holdout but never in train, so a
    model cannot learn a weight for it). Covers codes and sex; a patient "has"
    a feature if its value is > 0, so pass **unscaled** features.
    """
    present_cols = [c for c in features.columns if c not in ID_COLUMNS and c != "age"]
    by_split = (features[present_cols] > 0).astype(int).groupby(features["split"])
    pct = (by_split.mean().T * 100).reindex(columns=SPLIT_ORDER).add_prefix("pct_")
    counts = by_split.sum().T.reindex(columns=SPLIT_ORDER, fill_value=0)
    report = pct.assign(
        patients_total=counts.sum(axis=1),
        missing_in_train=(counts["train"] == 0) & (counts.drop(columns="train").sum(axis=1) > 0),
    )
    report.index.name = "feature"
    return report.sort_values("patients_total", ascending=False)


def coverage_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Per split: patients, code features present, mean distinct codes and code events
    per patient, share with no codes, and mean age if the features include
    demographics. Pass **unscaled** features."""
    codes = features[code_columns(features)]
    present = codes > 0
    split = features["split"]
    distinct = present.sum(axis=1).groupby(split)
    summary = pd.DataFrame(
        {
            "patients": distinct.size(),
            "features_present": present.groupby(split).max().sum(axis=1),
            "mean_codes_per_patient": distinct.mean().round(1),
            "mean_code_events_per_patient": codes.sum(axis=1).groupby(split).mean().round(1),
            "pct_patients_no_codes": (distinct.apply(lambda s: (s == 0).mean()) * 100).round(1),
        }
    )
    if "age" in features.columns:
        summary["mean_age"] = features.groupby("split")["age"].mean().round(1)
    return summary.reindex(SPLIT_ORDER)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build code-count features for a splits file.")
    parser.add_argument("--lookback-quarters", type=int, required=True)
    parser.add_argument("--splits-file", type=Path, help="defaults to the default splits file for the lookback")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument(
        "--ignore-codes", nargs="*", default=SURGERY_CODES, help="codes to drop (default: surgery codes)"
    )
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    parser.add_argument(
        "--min-train-patients",
        type=int,
        default=MIN_TRAIN_PATIENTS,
        help=f"keep features at least this many train patients have (default {MIN_TRAIN_PATIENTS})",
    )
    parser.add_argument("--no-demographics", action="store_true", help="leave out age and sex")
    parser.add_argument("--encoding", choices=list(ENCODINGS), default="count")
    parser.add_argument("--no-scale", action="store_true", help="skip min-max scaling")
    parser.add_argument("--no-specialties", action="store_true", help="leave out provider-specialty counts")
    args = parser.parse_args()

    path = write_features(
        args.lookback_quarters,
        args.splits_file,
        args.data_dir,
        args.out_dir,
        args.ignore_codes,
        args.sources,
        args.min_train_patients,
        include_demographics=not args.no_demographics,
        encoding=args.encoding,
        scale=not args.no_scale,
        include_specialties=not args.no_specialties,
    )
    n_features = len(pd.read_csv(path, nrows=0).columns) - len(ID_COLUMNS)
    print(f"wrote {path}  ({n_features} features)")
    print(pd.read_csv(sidecar_path(path, "summary"), index_col="split").to_string())
    coverage = pd.read_csv(sidecar_path(path, "coverage"))
    print(f"features missing in train: {int(coverage['missing_in_train'].sum())}")


if __name__ == "__main__":
    cli()
