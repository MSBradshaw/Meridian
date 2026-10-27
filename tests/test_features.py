from pathlib import Path

import pandas as pd
import pytest

from assignment.features import (
    DEMOGRAPHIC_COLUMNS,
    SPECIALTIES,
    apply_minmax,
    build_features,
    code_columns,
    coverage_summary,
    demographic_features,
    feature_coverage,
    features_filename,
    fit_minmax,
    ignore_tag,
    load_code_events,
    read_splits,
    read_scaling,
    read_vocabulary,
    sidecar_path,
    specialty_column,
    vocabulary_path,
    write_features,
)
from assignment.splits import SURGERY_CODES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def splits():
    return pd.DataFrame(
        {
            "patient_token": ["a", "b", "c", "d"],
            "split": ["train", "train", "test", "holdout"],
            "label": pd.array([1, 0, 0, pd.NA], dtype="Int8"),
            "index_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01", "2026-07-01"]),
            "start_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2025-07-01"]),
            "end_date": pd.to_datetime(["2024-12-31", "2024-12-31", "2024-12-31", "2026-06-30"]),
        }
    )


def events():
    rows = [
        ("a", "2024-01-01", "ICD10CM", "M54.50"),  # first day of window -> in
        ("a", "2024-12-31", "CPT", "72148"),  # last day of window -> in
        ("a", "2024-06-01", "CPT", "72148"),  # repeat -> still 1
        ("a", "2025-01-01", "CPT", "22630"),  # surgery on index date -> out (window and ignore list)
        ("a", "2024-06-01", "CPT", "63047"),  # surgery code in window -> ignored
        ("b", "2023-12-31", "CPT", "72100"),  # day before window -> out
        ("b", "2025-01-01", "NDC", "65862003900"),  # day after window -> out
        ("b", "2024-03-03", "HCPCS", "G0446"),
        ("d", "2026-06-30", "NDC", "65862003900"),
        ("z", "2024-06-01", "CPT", "99213"),  # patient not in splits -> dropped
    ]
    df = pd.DataFrame(rows, columns=["patient_token", "date", "code_system", "code"])
    return df.assign(date=pd.to_datetime(df.date))


def demographics():
    """Patients table extract; c is missing (e.g. claims-only), d has a blank sex."""
    return pd.DataFrame(
        {"patient_token": ["a", "b", "d", "x"], "sex": ["F", "M", None, "U"], "birth_year": pd.array([1950, 1960, 1940, 1970], dtype="Int64")}
    )


@pytest.fixture
def features():
    """Every observed code kept (threshold 0), so window and ignore rules can be checked directly."""
    return build_features(splits(), events(), min_train_patients=0).set_index("patient_token")


def test_one_row_per_patient_in_splits_order(features):
    assert list(features.index) == ["a", "b", "c", "d"]
    assert list(features.columns[:2]) == ["split", "label"]


def test_window_is_inclusive_and_nothing_outside_it_counts(features):
    assert features.loc["a", "ICD10CM_M54.50"] == 1
    assert features.loc["a", "CPT_72148"] == 2  # two rows in window, incl. the last day
    assert "CPT_72100" not in features.columns
    assert features.loc["b", "NDC_65862003900"] == 0
    assert features.loc["d", "NDC_65862003900"] == 1


def test_binary_encoding_and_patients_without_codes_are_all_zero(features):
    assert features.drop(columns=["split", "label"]).loc["c"].sum() == 0
    binary = build_features(splits(), events(), min_train_patients=0, encoding="binary").set_index("patient_token")
    assert set(binary.drop(columns=["split", "label"]).to_numpy().ravel()) <= {0, 1}
    assert binary.loc["a", "CPT_72148"] == 1
    with pytest.raises(ValueError, match="encoding"):
        build_features(splits(), events(), encoding="tfidf")


def test_surgery_codes_ignored_by_default(features):
    assert not [c for c in features.columns if c.split("_", 1)[-1] in SURGERY_CODES]


def test_ignore_codes_is_configurable():
    none_ignored = build_features(splits(), events(), ignore_codes=[], min_train_patients=0).set_index("patient_token")
    assert none_ignored.loc["a", "CPT_63047"] == 1
    assert "CPT_22630" not in none_ignored.columns  # still outside the window
    more = build_features(splits(), events(), ignore_codes=[*SURGERY_CODES, "72148"], min_train_patients=0)
    assert "CPT_72148" not in more.columns


def test_patients_outside_splits_are_dropped(features):
    assert "z" not in features.index
    assert "CPT_99213" not in features.columns


def test_filename_tracks_splits_file_and_ignore_list():
    splits_file = "artifacts/splits/patient_windows_lookback6q_neg1x_seed0.csv"
    assert features_filename(splits_file) == "features_count_minmax_lookback6q_neg1x_seed0_excl-surgery_mintrain11_demo_spec.csv"
    assert features_filename(splits_file, []) == "features_count_minmax_lookback6q_neg1x_seed0_excl-none_mintrain11_demo_spec.csv"
    assert features_filename(splits_file, include_specialties=False).endswith("_mintrain11_demo.csv")
    assert features_filename(splits_file, encoding="binary", scale=False).startswith("features_onehot_lookback6q_")
    assert features_filename(splits_file, min_train_patients=5).endswith("_excl-surgery_mintrain5_demo_spec.csv")
    assert features_filename(splits_file, include_demographics=False).endswith("_excl-surgery_mintrain11_spec.csv")
    custom = features_filename(splits_file, [*SURGERY_CODES, "72148"])
    assert custom.startswith("features_count_minmax_lookback6q_neg1x_seed0_excl-5codes-")
    assert ignore_tag(["72148", *SURGERY_CODES]) == ignore_tag([*SURGERY_CODES, "72148"])


def test_feature_coverage(features):
    report = feature_coverage(features.reset_index())
    assert list(report.columns) == ["pct_train", "pct_val", "pct_test", "pct_holdout", "patients_total", "missing_in_train"]
    assert report.loc["CPT_72148", "pct_train"] == 50.0
    assert report.loc["NDC_65862003900", "pct_holdout"] == 100.0
    assert report.loc["NDC_65862003900", "missing_in_train"]
    assert not report.loc["CPT_72148", "missing_in_train"]


def test_coverage_summary(features):
    summary = coverage_summary(features.reset_index())
    assert summary.loc["train", "patients"] == 2
    assert summary.loc["test", "pct_patients_no_codes"] == 100.0
    assert summary.loc["train", "features_present"] == 3
    assert summary.loc["train", "mean_code_events_per_patient"] == 2.0  # a: 3 events, b: 1


def test_write_features_round_trip(tmp_path, monkeypatch):
    splits_file = tmp_path / "patient_windows_lookback4q_neg1x_seed0.csv"
    splits().to_csv(splits_file, index=False, date_format="%Y-%m-%d")
    monkeypatch.setattr("assignment.features.load_code_events", lambda data_dir, sources: events())
    monkeypatch.setattr("assignment.features.load_demographics", lambda data_dir: demographics())
    path = write_features(4, splits_file=splits_file, out_dir=tmp_path / "features", min_train_patients=1)
    assert path.name == "features_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain1_demo_spec.csv"
    written = pd.read_csv(path, dtype={"patient_token": str})
    expected = build_features(
        read_splits(splits_file), events(), min_train_patients=1, demographics=demographics(), specialties=SPECIALTIES
    )
    vocab = pd.read_csv(vocabulary_path(path))
    assert list(vocab.columns) == ["feature", "train_patients"]
    assert read_vocabulary(vocabulary_path(path)) == list(expected.columns[3 + len(DEMOGRAPHIC_COLUMNS) + len(SPECIALTIES):])  # codes only
    assert dict(zip(vocab.feature, vocab.train_patients)) == {"CPT_72148": 1, "HCPCS_G0446": 1, "ICD10CM_M54.50": 1}
    assert list(written.columns) == list(expected.columns)
    scaling = read_scaling(sidecar_path(path, "scaling"))
    pd.testing.assert_frame_equal(scaling, fit_minmax(expected), check_dtype=False)
    scaled = apply_minmax(expected, scaling).drop(columns=["patient_token", "split", "label"]).to_numpy()
    assert abs(written.drop(columns=["patient_token", "split", "label"]).to_numpy() - scaled).max() < 1e-5
    assert sidecar_path(path, "coverage").exists() and sidecar_path(path, "summary").exists()
    unscaled = write_features(4, splits_file=splits_file, out_dir=tmp_path / "raw", min_train_patients=1, scale=False)
    assert not sidecar_path(unscaled, "scaling").exists()
    assert (pd.read_csv(unscaled).CPT_72148.tolist()) == [2, 0, 0, 0]


def test_minmax_is_fitted_on_train_only():
    f = build_features(splits(), events(), min_train_patients=0, demographics=demographics())
    scaling = fit_minmax(f)
    assert "sex_F" not in scaling.index  # already 0/1
    assert scaling.loc["CPT_72148"].tolist() == [0.0, 2.0]
    assert scaling.loc["age"].tolist() == [66.0, 76.0]  # train patients a (76) and b (66) only

    changed = f.copy()
    others = changed.split != "train"
    changed.loc[others, "CPT_72148"] = 50
    changed.loc[others, "age"] = 120.0
    pd.testing.assert_frame_equal(fit_minmax(changed), scaling)


def test_apply_minmax_maps_train_to_unit_range_and_does_not_clip_others():
    f = build_features(splits(), events(), min_train_patients=0, demographics=demographics())
    scaled = apply_minmax(f, fit_minmax(f)).set_index("patient_token")
    train = scaled[scaled.split == "train"]
    cols = list(fit_minmax(f).index)
    assert train[cols].min().min() >= 0 and train[cols].max().max() <= 1
    assert scaled.loc["a", "CPT_72148"] == 1.0
    assert scaled.loc["a", "age"] == 1.0 and scaled.loc["b", "age"] == 0.0
    assert scaled.loc["d", "age"] == pytest.approx(2.0)  # 86 with train range 66-76: outside, not clipped
    assert scaled.loc["d", "NDC_65862003900"] == 1.0  # constant (0) in train -> range 1, only shifted
    assert (scaled[["sex_F", "sex_M", "sex_U"]] == f.set_index("patient_token")[["sex_F", "sex_M", "sex_U"]]).all().all()


def test_demographic_features():
    demo = demographic_features(splits(), demographics())
    assert list(demo.columns) == DEMOGRAPHIC_COLUMNS
    assert list(demo.age) == [76.0, 66.0, 71.0, 86.0]  # c imputed with the table median (1955 -> 71)
    assert list(demo.sex_F) == [1, 0, 0, 0]
    assert list(demo.sex_M) == [0, 1, 0, 0]
    assert demo.loc[2:3, ["sex_F", "sex_M", "sex_U"]].sum().sum() == 0  # missing patient / blank sex


def test_demographics_placed_after_ids_and_kept_regardless_of_vocabulary():
    f = build_features(splits(), events(), min_train_patients=99, demographics=demographics())
    assert list(f.columns) == ["patient_token", "split", "label", *DEMOGRAPHIC_COLUMNS]
    f = build_features(splits(), events(), vocabulary=["CPT_72148"], demographics=demographics())
    assert list(f.columns[3:]) == [*DEMOGRAPHIC_COLUMNS, "CPT_72148"]


def test_coverage_excludes_age_and_summary_reports_it():
    f = build_features(splits(), events(), min_train_patients=0, demographics=demographics())
    assert "age" not in feature_coverage(f).index
    assert "sex_F" in feature_coverage(f).index
    summary = coverage_summary(f)
    assert summary.loc["train", "mean_age"] == 71.0
    assert summary.loc["train", "mean_codes_per_patient"] == 1.5  # sex and age not counted as codes


def test_vocabulary_is_chosen_on_train_only():
    f = build_features(splits(), events(), min_train_patients=1)
    assert list(f.columns[3:]) == ["CPT_72148", "HCPCS_G0446", "ICD10CM_M54.50"]
    assert "NDC_65862003900" not in f.columns  # only the holdout patient has it in-window


def test_min_train_patients_threshold():
    many = splits().iloc[[0]].loc[[0] * 12].assign(patient_token=[f"t{i}" for i in range(12)], split="train")
    rows = [(f"t{i}", "2024-05-05", "CPT", "72148") for i in range(11)] + [
        (f"t{i}", "2024-05-05", "CPT", "72100") for i in range(10)
    ]
    ev = pd.DataFrame(rows, columns=["patient_token", "date", "code_system", "code"]).assign(
        date=lambda d: pd.to_datetime(d.date)
    )
    default = build_features(many, ev)
    assert list(default.columns[3:]) == ["CPT_72148"]  # 11 train patients kept, 10 dropped
    assert list(build_features(many, ev, min_train_patients=10).columns[3:]) == ["CPT_72100", "CPT_72148"]


def test_existing_vocabulary_is_reused_as_is():
    f = build_features(splits(), events(), vocabulary=["NDC_65862003900", "CPT_99999"]).set_index("patient_token")
    assert list(f.columns[2:]) == ["CPT_99999", "NDC_65862003900"]
    assert f["CPT_99999"].sum() == 0
    assert f.loc["d", "NDC_65862003900"] == 1


def test_vocabulary_needs_train_patients():
    with pytest.raises(ValueError, match="no train patients"):
        build_features(splits().query("split != 'train'"), events())


@pytest.mark.skipif(not (DATA_DIR / "meridian_health_patients.csv").exists(), reason="extract not present")
def test_real_events_load_all_four_code_systems():
    events = load_code_events(DATA_DIR)
    assert set(events.code_system.dropna()) == {"ICD10CM", "CPT", "HCPCS", "NDC"}
    assert events[["patient_token", "date"]].notna().all().all()
    assert set(events.provider_specialty.dropna()) <= set(SPECIALTIES)


def specialty_events():
    rows = [
        ("a", "2024-02-01", "CPT", "72148", "Radiology"),
        ("a", "2024-03-01", "ICD10CM", "M54.50", "Orthopedic Surgery"),
        ("a", "2024-03-01", "CPT", "99213", "Orthopedic Surgery"),  # two lines, same visit -> 2
        ("a", "2024-04-01", None, None, "Physical Therapy"),  # no code: still counts for the specialty
        ("a", "2024-05-01", "CPT", "63047", "Orthopedic Surgery"),  # ignored surgery code: counts for neither
        ("a", "2025-02-01", "CPT", "99213", "Radiology"),  # after the window
        ("a", "2024-06-01", "NDC", "65862003900", None),  # pharmacy line, no specialty
        ("b", "2024-06-01", "CPT", "99213", "Primary Care"),
        ("d", "2026-06-30", "CPT", "99213", "Pain Management"),
    ]
    df = pd.DataFrame(rows, columns=["patient_token", "date", "code_system", "code", "provider_specialty"])
    return df.assign(date=pd.to_datetime(df.date))


def test_specialty_columns_count_claim_lines_in_window():
    f = build_features(splits(), specialty_events(), min_train_patients=0, specialties=SPECIALTIES).set_index("patient_token")
    spec = [specialty_column(s) for s in SPECIALTIES]
    assert list(f.columns[2:2 + len(spec)]) == spec  # after the ID columns, before codes
    assert f.loc["a", "specialty_orthopedic_surgery"] == 2
    assert f.loc["a", "specialty_radiology"] == 1
    assert f.loc["a", "specialty_physical_therapy"] == 1
    assert f.loc["b", "specialty_primary_care"] == 1
    assert f.loc["d", "specialty_pain_management"] == 1
    assert f.loc["c", spec].sum() == 0
    assert f["specialty_neurosurgery"].sum() == 0  # documented but absent: column kept, all zero
    assert "CPT_63047" not in f.columns


def test_specialties_are_scaled_on_train_and_not_codes():
    f = build_features(splits(), specialty_events(), min_train_patients=0, specialties=SPECIALTIES)
    scaling = fit_minmax(f)
    assert scaling.loc["specialty_orthopedic_surgery"].tolist() == [0.0, 2.0]
    assert "specialty_orthopedic_surgery" not in code_columns(f)
    summary = coverage_summary(f)
    assert summary.loc["train", "mean_code_events_per_patient"] == 2.5  # a: 72148, M54.50, 99213, NDC; b: 99213


def test_specialties_optional():
    f = build_features(splits(), events(), min_train_patients=0)
    assert not [c for c in f.columns if c.startswith("specialty_")]
