from pathlib import Path

import pandas as pd
import pytest

from assignment.splits import (
    HISTORY_START,
    HOLDOUT,
    SURGERY_CODES,
    assign_split,
    build_splits,
    load_tables,
    split_bucket,
    splits_filename,
    write_splits,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELLED = ["train", "val", "test"]


def tok(i: int) -> str:
    """Hex token whose split bucket is i % 100."""
    return f"{i:032x}"


def pid(group: int, i: int) -> str:
    """Token for patient i of a synthetic group; i * 37 spreads groups across all split buckets."""
    return tok(group * 100_000 + i * 37)


def ids(group: int, n: int) -> list[str]:
    return [pid(group, i) for i in range(n)]


def tables():
    """Small synthetic extract.

    group 0 (60)  patients table, 3 years of claims, no surgery       -> negative pool
    group 1 (40)  patients table, surgery code in encounters          -> positives
    group 2 (10)  claims only, surgery code in claims                 -> positives (not in patients table)
    group 9 (5)   patients table, surgery code, no data before it     -> no history before the gap: excluded
    group 3 (5)   patients table, `Spine Surgery` row with no code    -> excluded everywhere
    group 4 (10)  patients table, 1 year of history, no surgery       -> holdout only (too little history)
    group 5 (10)  claims only, long history, no surgery               -> negative pool, never holdout
    group 6 (10)  patients table, long history ending before every
                  surgery date, no surgery                           -> no matchable index date, and no
                                                                        recent data so not holdout either
    group 7 (10)  patients table, 1 year of history, coverage ended   -> not holdout (inactive coverage)
    group 8 (10)  patients table, 1 year of history, last event
                  2026-03-31 (the day before the last quarter)       -> not holdout (no recent data)

    Everyone in the patients table has active coverage (no end date) except group 7.
    """
    patients = ids(0, 60) + ids(1, 40) + ids(3, 5) + ids(4, 10) + ids(6, 10) + ids(7, 10) + ids(8, 10) + ids(9, 5)
    claims, encounters = [], []
    for t in ids(6, 10):
        claims += [(t, "2022-06-01", "ICD10CM", "M54.50"), (t, "2024-12-01", "CPT", "99213")]
    for t in ids(0, 60) + ids(5, 10):
        claims += [(t, "2023-08-01", "ICD10CM", "M54.50"), (t, "2026-05-01", "CPT", "99213")]
    for t in ids(4, 10) + ids(7, 10):
        claims += [(t, "2025-09-01", "CPT", "99213"), (t, "2026-06-30", "CPT", "99213")]
    for t in ids(8, 10):
        claims += [(t, "2025-09-01", "CPT", "99213"), (t, "2026-03-31", "CPT", "99213")]
    for i, t in enumerate(ids(1, 40)):
        encounters += [
            (t, "2024-01-10", "LBP Encounter", "M54.50"),
            (t, "2025-03-15", "Spine Surgery", SURGERY_CODES[i % 4]),
            (t, "2025-06-01", "LBP Encounter", "M54.50"),
        ]
    for i, t in enumerate(ids(2, 10)):
        claims += [(t, "2023-09-01", "ICD10CM", "M54.50"), (t, "2025-02-02", "CPT", SURGERY_CODES[i % 4])]
    for i, t in enumerate(ids(9, 5)):
        claims += [(t, "2025-01-01", "ICD10CM", "M54.50"), (t, "2025-05-01", "CPT", SURGERY_CODES[i % 4])]
    for t in ids(3, 5):
        claims += [(t, "2023-08-01", "ICD10CM", "M54.50"), (t, "2026-05-01", "CPT", "99213")]
        encounters.append((t, "2025-01-01", "Spine Surgery", None))
    return {
        "patients": pd.DataFrame(
            {
                "patient_token": patients,
                "coverage_start": pd.Timestamp("2023-07-01"),
                "coverage_end": [pd.Timestamp("2026-03-01") if p in ids(7, 10) else pd.NaT for p in patients],
            }
        ),
        "encounters": pd.DataFrame(encounters, columns=["patient_token", "event_date", "event_category", "event_code"]).astype(
            {"event_date": "datetime64[ns]"}
        ),
        "claims": pd.DataFrame(claims, columns=["patient_token", "service_date", "code_system", "code"]).astype(
            {"service_date": "datetime64[ns]"}
        ),
    }


def surgery_evidence(t) -> set:
    enc, clm = t["encounters"], t["claims"]
    return set(enc.loc[enc.event_code.isin(SURGERY_CODES) | (enc.event_category == "Spine Surgery"), "patient_token"]) | set(
        clm.loc[clm.code.isin(SURGERY_CODES), "patient_token"]
    )


def assert_no_overlap(splits: pd.DataFrame) -> None:
    assert not splits.patient_token.duplicated().any(), "a patient appears in more than one row"
    groups = {s: set(splits.loc[splits.split == s, "patient_token"]) for s in [*LABELLED, HOLDOUT]}
    for a in groups:
        for b in groups:
            if a < b:
                assert not groups[a] & groups[b], f"{a} and {b} share patients"


@pytest.fixture
def synthetic():
    t = tables()
    return t, build_splits(**t, gap_quarters=4, neg_ratio=1.0, seed=0)


def test_split_bucket_and_assignment():
    assert [split_bucket(tok(i)) for i in (0, 69, 70, 84, 85, 99, 100, 170)] == [0, 69, 70, 84, 85, 99, 0, 70]
    assert [assign_split(tok(i)) for i in (0, 69, 70, 84, 85, 99)] == ["train", "train", "val", "val", "test", "test"]


def test_no_overlap_between_train_val_test(synthetic):
    _, splits = synthetic
    assert_no_overlap(splits)
    assert set(splits.split) == {*LABELLED, HOLDOUT}
    labelled = splits[splits.split.isin(LABELLED)]
    assert (labelled.split == labelled.patient_token.map(assign_split)).all()


def test_holdout_does_not_overlap_splits(synthetic):
    _, splits = synthetic
    holdout = set(splits.loc[splits.split == HOLDOUT, "patient_token"])
    labelled = set(splits.loc[splits.split.isin(LABELLED), "patient_token"])
    assert holdout and labelled
    assert not holdout & labelled


def test_holdout_only_from_patients_table_and_without_surgery(synthetic):
    t, splits = synthetic
    holdout = set(splits.loc[splits.split == HOLDOUT, "patient_token"])
    assert holdout <= set(t["patients"].patient_token)
    assert not holdout & surgery_evidence(t)
    assert splits.loc[splits.split == HOLDOUT, "label"].isna().all()


def test_positives_include_claims_only_patients(synthetic):
    _, splits = synthetic
    pos = set(splits.loc[splits.label == 1, "patient_token"])
    assert pos == set(ids(1, 40) + ids(2, 10))
    assert not set(splits.patient_token) & set(ids(9, 5))  # no data a year before surgery: neither positive nor holdout


def test_negatives_have_no_surgery_and_enough_history(synthetic):
    t, splits = synthetic
    neg = splits[splits.label == 0]
    assert len(neg) == (splits.label == 1).sum()
    assert not set(neg.patient_token) & surgery_evidence(t)
    assert set(neg.patient_token) <= set(ids(0, 60) + ids(5, 10))


def test_negative_index_dates_match_positives_and_sit_inside_observed_data(synthetic):
    t, splits = synthetic
    pos_dates = set(splits.loc[splits.label == 1, "index_date"])
    neg = splits[splits.label == 0].set_index("patient_token")
    assert set(neg.index_date) <= pos_dates
    assert len(set(neg.index_date)) == len(pos_dates)  # both positive dates are used
    dates = pd.concat([t["claims"].set_index("patient_token").service_date, t["encounters"].set_index("patient_token").event_date])
    span = dates.groupby(level=0).agg(["min", "max"]).loc[neg.index]
    assert (neg.index_date > span["min"] + pd.DateOffset(months=12)).all()  # data before the gap
    assert (neg.index_date <= span["max"] + pd.Timedelta(days=1)).all()
    assert not set(neg.index) & set(ids(6, 10))


def test_holdout_requires_active_coverage_and_recent_data(synthetic):
    t, splits = synthetic
    holdout = set(splits.loc[splits.split == HOLDOUT, "patient_token"])
    assert set(ids(4, 10)) <= holdout  # covered, event on the last day of the extract
    assert not holdout & set(ids(7, 10))  # coverage ended
    assert not holdout & set(ids(8, 10))  # last event one day before the last quarter
    assert not holdout & set(ids(6, 10))  # last event in 2024
    covered = t["patients"].loc[t["patients"].coverage_end.isna(), "patient_token"]
    assert holdout <= set(covered)


def test_holdout_recency_window_is_variable():
    splits = build_splits(**tables(), gap_quarters=4, holdout_recent_quarters=2)
    holdout = set(splits.loc[splits.split == HOLDOUT, "patient_token"])
    assert set(ids(8, 10)) <= holdout  # 2026-03-31 is inside the last two quarters


def test_ambiguous_surgery_patients_excluded_everywhere(synthetic):
    _, splits = synthetic
    assert not set(splits.patient_token) & set(ids(3, 5))


def test_windows_use_all_history_up_to_the_gap(synthetic):
    _, splits = synthetic
    labelled = splits[splits.split != HOLDOUT]
    assert (labelled.end_date == labelled.index_date - pd.DateOffset(months=12) - pd.Timedelta(days=1)).all()
    assert (splits.start_date == HISTORY_START).all()
    holdout = splits[splits.split == HOLDOUT]  # scored on everything up to the extract end: no gap
    assert (holdout.end_date == pd.Timestamp("2026-06-30")).all()
    pos = splits[splits.label == 1].set_index("patient_token")
    assert (pos.loc[pid(1, 0), "index_date"] == pd.Timestamp("2025-03-15"))
    assert (pos.loc[pid(2, 0), "index_date"] == pd.Timestamp("2025-02-02"))


def test_gap_is_variable():
    t = tables()
    for q in (1, 4, 6):
        s = build_splits(**t, gap_quarters=q)
        labelled = s[s.split != HOLDOUT]
        assert (labelled.end_date == labelled.index_date - pd.DateOffset(months=3 * q) - pd.Timedelta(days=1)).all()
    with pytest.raises(ValueError):
        build_splits(**t, gap_quarters=0)


def test_sampling_is_reproducible_and_ratio_respected():
    t = tables()
    a = build_splits(**t, gap_quarters=4, neg_ratio=0.5, seed=1)
    b = build_splits(**t, gap_quarters=4, neg_ratio=0.5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a.label == 0).sum() == round(0.5 * (a.label == 1).sum())
    assert_no_overlap(a)


def test_write_splits_filename(tmp_path, monkeypatch):
    monkeypatch.setattr("assignment.splits.load_tables", lambda data_dir: tables())
    path = write_splits(gap_quarters=6, out_dir=tmp_path)
    assert path.name == splits_filename(6) == "patient_windows_gap6q_neg1x_seed0.csv"
    written = pd.read_csv(path)
    assert list(written.columns) == ["patient_token", "split", "label", "index_date", "start_date", "end_date"]


@pytest.mark.skipif(not (DATA_DIR / "meridian_health_patients.csv").exists(), reason="extract not present")
def test_real_data_has_no_overlap():
    t = load_tables(DATA_DIR)
    splits = build_splits(**t, gap_quarters=4)
    assert_no_overlap(splits)
    holdout = set(splits.loc[splits.split == HOLDOUT, "patient_token"])
    assert holdout <= set(t["patients"].patient_token)
    assert holdout <= set(t["patients"].loc[t["patients"].coverage_end.isna(), "patient_token"])
    assert not holdout & surgery_evidence(t)
    assert not set(splits.loc[splits.label == 0, "patient_token"]) & surgery_evidence(t)
