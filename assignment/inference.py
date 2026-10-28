"""Inference: score the holdout and turn the scores into the ranked outreach list.

Runs on the holdout of the surgery splits (patients-table patients with active
coverage, recent data and no surgery evidence; see `assignment.splits`), using
the models recorded in `artifacts/manifest.json` by `assignment.pipeline`:

    1. surgery model    surgery risk for every holdout patient, from all their
                        history up to today. The model learned what patients
                        look like `gap_quarters` quarters before surgery, so this
                        flags surgery that far ahead or more. Positive if the
                        score is at least `surgery_threshold` (default 0.5; the
                        model was trained on a 50/50 sample).
    2. PT / WM models   enrollment propensity for the surgery-positive patients
                        only. Positive if the score is at least the model's
                        training positive rate, i.e. more likely than the
                        average training patient to enroll rather than refuse.
    3. rules            for every patient positive for PT and/or WM:
                        - coverage is active at the extract end
                        - no surgery evidence (any surgery code, or a `Spine
                          Surgery` encounter, any status)
                        - PT is dropped if already enrolled in PT, or with
                          PT-specialty claims before the index date (prior PT
                          care, which the PT model's cohort excludes)
                        - WM is dropped if already enrolled in WM
                        Patients left with no program are removed.

Features are rebuilt from the raw extract for each model with its saved
vocabulary and train min/max scaling, so this also scores a newer extract.

Scores are put on a real-world scale with a prior-shift correction: the models
are trained on artificial class mixes (surgery 1:1, enrollment mostly
enrollers), so each model's odds are rescaled from its training positive rate to
the population rate observed in the extract (`prior_correct`):

    surgery_risk     annual rate of new surgery among patients-table patients
    PT/WM_enroll_prob  share of outreached patients who enrolled in the program
                       within 90 days of the first attempt

The correction is monotonic, so thresholds, flags and the ranking are unchanged;
it only changes the numbers. It assumes the training negatives are representative
of the population, which they only roughly are.

Output, ordered by surgery score (highest first):

    targets         patient_token, program_recommended ("PT", "WM" or "PT+WM"), score
                    (the prior-corrected surgery risk)
    targets_detail  the same plus every raw and corrected model score, threshold
                    and rule outcome

    from assignment.inference import run_inference
    targets, detail = run_inference(Path("data"), Path("artifacts"))
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from assignment.enrollment_splits import load_program_tables, prior_specialty_patients
from assignment.features import (
    SPECIALTIES,
    apply_minmax,
    build_features,
    load_code_events,
    load_demographics,
    read_scaling,
    read_splits,
    read_vocabulary,
)
from assignment.splits import EXTRACT_END, HOLDOUT, _events, _window

PROGRAMS = ("PT", "WM")
SURGERY_THRESHOLD = 0.5
TARGET_COLUMNS = ["patient_token", "program_recommended", "score"]
ENROLLMENT_WINDOW_DAYS = 90


def load_manifest(artifacts_dir: Path) -> dict:
    path = artifacts_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run main.py train` first")
    return json.loads(path.read_text())


def train_positive_rate(entry: dict) -> float:
    report = json.loads((Path(entry["run_dir"]) / "metrics_val.json").read_text())
    return report["train"]["positives"] / report["train"]["patients"]


def prior_correct(scores: pd.Series, train_rate: float, true_rate: float) -> pd.Series:
    """Rescale probabilities from a model trained at `train_rate` positives to a population at `true_rate`.

    odds_true = odds_model * [true_rate / (1 - true_rate)] / [train_rate / (1 - train_rate)]
    """
    p = scores.clip(1e-6, 1 - 1e-6)
    odds = p / (1 - p) * (true_rate / (1 - true_rate)) / (train_rate / (1 - train_rate))
    return odds / (1 + odds)


def annual_surgery_rate(tables: dict[str, pd.DataFrame], extract_end: pd.Timestamp = EXTRACT_END) -> float:
    """Patients-table patients whose first surgery code falls in the extract, per patient per year."""
    events = _events(tables["encounters"], tables["claims"])
    patients = set(tables["patients"]["patient_token"])
    first_surgery = events[events["is_surgery_code"]].groupby("patient_token")["date"].min()
    n_surgery = first_surgery.index.isin(patients).sum()
    years = (extract_end - events["date"].min() + pd.Timedelta(days=1)).days / 365.25
    return n_surgery / len(patients) / years


def outreach_enrollment_rate(tables: dict[str, pd.DataFrame], program: str, days: int = ENROLLMENT_WINDOW_DAYS) -> float:
    """Share of patients outreached for `program` who enrolled in it within `days` of their first attempt."""
    outreach = tables["outreach"]
    first = outreach[outreach["program"] == program].groupby("patient_token")["timestamp"].min().dt.normalize()
    enrolled = tables["enrollments"]
    enrolled = enrolled[enrolled["program"] == program].merge(first.rename("first_attempt"), on="patient_token")
    gap = (enrolled["enrollment_date"] - enrolled["first_attempt"]).dt.days
    return enrolled.loc[gap.between(0, days), "patient_token"].nunique() / len(first)


def holdout_rows(manifest: dict) -> pd.DataFrame:
    splits = read_splits(manifest["surgery"]["splits_file"])
    return splits[splits["split"] == HOLDOUT].reset_index(drop=True)


def score_with_model(
    entry: dict, patients: pd.DataFrame, events: pd.DataFrame, demographics: pd.DataFrame
) -> pd.Series:
    """Score `patients` (`patient_token, index_date`, plus split/label) with one task's model.

    Features use the task's scoring window (all history for surgery, the lookback
    for enrollment) before each index date, with no gap, its ignored codes,
    saved vocabulary and train scaling; columns follow the model's feature list.
    """
    windows = _window(pd.Index(patients["patient_token"]), patients.set_index("patient_token")["index_date"], entry["lookback_quarters"])
    rows = windows.assign(split=HOLDOUT, label=pd.NA)
    features = build_features(
        rows,
        events,
        ignore_codes=entry["ignore_codes"],
        vocabulary=read_vocabulary(entry["vocabulary_file"]),
        demographics=demographics,
        specialties=SPECIALTIES,
    )
    features = apply_minmax(features, read_scaling(entry["scaling_file"]))
    with open(Path(entry["run_dir"]) / "model.pkl", "rb") as f:
        saved = pickle.load(f)
    missing = [c for c in saved["features"] if c not in features.columns]
    if missing:
        raise ValueError(f"{entry['task']}: rebuilt features lack model columns {missing[:3]}")
    return pd.Series(saved["model"].predict_proba(features[saved["features"]])[:, 1], index=features["patient_token"])


def eligibility(tokens: pd.Index, tables: dict[str, pd.DataFrame], index_date: pd.Timestamp, extract_end: pd.Timestamp) -> pd.DataFrame:
    """Per patient: covered at the extract end, any surgery evidence, enrolled in each program, prior PT care."""
    patients = tables["patients"].set_index("patient_token").reindex(tokens)
    covered = (patients["coverage_start"] <= extract_end) & (
        patients["coverage_end"].isna() | (patients["coverage_end"] >= extract_end)
    )
    events = _events(tables["encounters"], tables["claims"])
    surgery = set(events.loc[events["is_surgery_any"], "patient_token"])
    enrollments = tables["enrollments"]
    prior_pt = prior_specialty_patients(
        pd.DataFrame({"patient_token": tokens, "index_date": index_date}), tables["claims"], ["Physical Therapy"]
    )
    out = pd.DataFrame(index=tokens)
    out["covered"] = covered.fillna(False).astype(bool)
    out["has_surgery"] = out.index.isin(surgery)
    for program in PROGRAMS:
        enrolled = set(enrollments.loc[enrollments["program"] == program, "patient_token"])
        out[f"enrolled_{program}"] = out.index.isin(enrolled)
    out["prior_pt_care"] = out.index.isin(prior_pt)
    return out


def recommend(
    scores: pd.DataFrame,
    eligible: pd.DataFrame,
    surgery_threshold: float,
    program_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Apply thresholds and rules. `scores` has `surgery_score` and `<program>_score` columns
    (missing where not scored), indexed by patient_token; `eligible` comes from `eligibility`.

    Returns every scored patient, ordered by surgery score, with the positive /
    recommended flags and `program_recommended` ("" when no program survives the
    rules); the outreach list is the rows where it is not empty.
    """
    d = scores.join(eligible, how="left")
    d["surgery_positive"] = d["surgery_score"] >= surgery_threshold
    allowed = {
        "PT": d["covered"] & ~d["has_surgery"] & ~d["enrolled_PT"] & ~d["prior_pt_care"],
        "WM": d["covered"] & ~d["has_surgery"] & ~d["enrolled_WM"],
    }
    for program in PROGRAMS:
        positive = d["surgery_positive"] & (d[f"{program}_score"] >= program_thresholds[program]).fillna(False)
        d[f"{program}_positive"] = positive
        d[f"{program}_recommended"] = positive & allowed[program]
    d["program_recommended"] = [
        "+".join(p for p in PROGRAMS if row[f"{p}_recommended"]) for _, row in d.iterrows()
    ]
    d = d.sort_values("surgery_score", ascending=False, kind="stable")
    d.index.name = "patient_token"
    return d.reset_index().assign(score=lambda x: x["surgery_score"])


def run_inference(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    surgery_threshold: float = SURGERY_THRESHOLD,
    program_thresholds: dict[str, float] | None = None,
    extract_end: pd.Timestamp = EXTRACT_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the holdout and build the outreach list; returns (targets, detail)."""
    manifest = load_manifest(artifacts_dir)
    if program_thresholds is None:
        program_thresholds = {p: train_positive_rate(manifest[f"enroll-{p}"]) for p in PROGRAMS}

    holdout = holdout_rows(manifest)
    events = load_code_events(data_dir)
    demographics = load_demographics(data_dir)

    scores = pd.DataFrame({"surgery_score": score_with_model(manifest["surgery"], holdout, events, demographics)})
    positives = holdout[holdout["patient_token"].isin(scores.index[scores["surgery_score"] >= surgery_threshold])]
    for program in PROGRAMS:
        scores[f"{program}_score"] = np.nan
        if not positives.empty:
            program_scores = score_with_model(manifest[f"enroll-{program}"], positives, events, demographics)
            scores.loc[program_scores.index, f"{program}_score"] = program_scores

    index_date = extract_end + pd.Timedelta(days=1)
    tables = load_program_tables(data_dir)
    eligible = eligibility(scores.index, tables, index_date, extract_end)
    detail = recommend(scores, eligible, surgery_threshold, program_thresholds)

    # real-world scale: rescale each model from its training class mix to the population rate
    surgery_rate = annual_surgery_rate(tables, extract_end)
    detail["surgery_risk"] = prior_correct(detail["surgery_score"], train_positive_rate(manifest["surgery"]), surgery_rate)
    rates = {"surgery_population_rate": surgery_rate}
    for program in PROGRAMS:
        rate = outreach_enrollment_rate(tables, program)
        detail[f"{program}_enroll_prob"] = prior_correct(
            detail[f"{program}_score"], train_positive_rate(manifest[f"enroll-{program}"]), rate
        )
        rates[f"{program}_population_rate"] = rate
    detail["score"] = detail["surgery_risk"]
    detail = detail.assign(
        surgery_threshold=surgery_threshold, **{f"{p}_threshold": program_thresholds[p] for p in PROGRAMS}, **rates
    )
    targets = detail.loc[detail["program_recommended"] != "", TARGET_COLUMNS].reset_index(drop=True)
    return targets, detail


def funnel(detail: pd.DataFrame) -> pd.Series:
    """Patients remaining at each step, from the `detail` table of `run_inference`."""
    listed = detail["program_recommended"]
    return pd.Series(
        {
            "holdout scored": len(detail),
            "surgery positive": int(detail["surgery_positive"].sum()),
            "PT positive": int(detail["PT_positive"].sum()),
            "WM positive": int(detail["WM_positive"].sum()),
            "PT or WM positive": int((detail["PT_positive"] | detail["WM_positive"]).sum()),
            "dropped: not covered": int(((detail["PT_positive"] | detail["WM_positive"]) & ~detail["covered"]).sum()),
            "dropped: surgery evidence": int(((detail["PT_positive"] | detail["WM_positive"]) & detail["has_surgery"]).sum()),
            "PT removed: enrolled in PT": int((detail["PT_positive"] & detail["enrolled_PT"]).sum()),
            "PT removed: prior PT care": int((detail["PT_positive"] & detail["prior_pt_care"]).sum()),
            "WM removed: enrolled in WM": int((detail["WM_positive"] & detail["enrolled_WM"]).sum()),
            "on list": int((listed != "").sum()),
            "on list: PT only": int((listed == "PT").sum()),
            "on list: WM only": int((listed == "WM").sum()),
            "on list: PT+WM": int((listed == "PT+WM").sum()),
        }
    )
