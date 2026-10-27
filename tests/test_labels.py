from pathlib import Path

import pandas as pd
import pytest

from assignment.labels import (
    enrollment_labels,
    label_summary,
    labels_filename,
    load_enrollments,
    task_from_labels_file,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def splits():
    """Index 2025-01-01 for everyone except the late patient and the holdout."""
    tokens = ["new", "late_in", "before", "after", "prior", "other_prog", "none", "censored", "hold"]
    index = ["2025-01-01"] * 7 + ["2026-01-01", "2026-07-01"]
    split = ["train"] * 7 + ["val", "holdout"]
    return pd.DataFrame(
        {"patient_token": tokens, "split": split, "index_date": pd.to_datetime(index)}
    )


def enrollments():
    rows = [
        ("new", "PT", "2025-01-01"),  # on the index date -> in horizon
        ("late_in", "PT", "2025-12-31"),  # last day of a 4-quarter horizon
        ("after", "PT", "2026-01-01"),  # first day after the horizon -> 0
        ("before", "PT", "2024-12-31"),  # day before index -> prior enrollee, excluded
        ("prior", "PT", "2024-03-01"),  # prior enrollee who enrolls again in horizon: still excluded
        ("prior", "PT", "2025-02-01"),
        ("other_prog", "WM", "2025-02-01"),  # WM only -> PT label 0
        ("censored", "PT", "2026-02-01"),
        ("hold", "PT", "2026-08-01"),
    ]
    df = pd.DataFrame(rows, columns=["patient_token", "program", "enrollment_date"])
    return df.assign(enrollment_date=pd.to_datetime(df.enrollment_date))


def labels(program="PT", horizon=4):
    return enrollment_labels(splits(), enrollments(), program, horizon).set_index("patient_token").label


def test_horizon_is_index_date_inclusive_and_end_exclusive():
    pt = labels()
    assert pt["new"] == 1 and pt["late_in"] == 1
    assert pt["after"] == 0
    assert pt["none"] == 0 and pt["other_prog"] == 0


def test_prior_enrollees_in_that_program_are_excluded():
    pt = labels()
    assert pd.isna(pt["before"]) and pd.isna(pt["prior"])
    wm = labels("WM")
    assert wm["before"] == 0  # prior PT enrollment does not exclude from WM
    assert wm["other_prog"] == 1


def test_censored_follow_up_and_holdout_are_unlabelled():
    pt = labels()
    assert pd.isna(pt["censored"])  # 2026-01-01 + 4 quarters runs past 2026-06-30
    assert pd.isna(pt["hold"])
    assert labels(horizon=1)["censored"] == 1  # 1-quarter horizon (to 2026-03-31) is fully observed; enrolled 2026-02-01


def test_shorter_horizon():
    pt = labels(horizon=1)
    assert pt["new"] == 1
    assert pt["late_in"] == 0  # 2025-12-31 is outside 1 quarter


def test_one_row_per_splits_patient_in_order():
    out = enrollment_labels(splits(), enrollments(), "PT")
    assert list(out.patient_token) == list(splits().patient_token)
    assert str(out.label.dtype) == "Int8"


def test_bad_arguments():
    with pytest.raises(ValueError, match="program"):
        enrollment_labels(splits(), enrollments(), "XX")
    with pytest.raises(ValueError, match="horizon"):
        enrollment_labels(splits(), enrollments(), "PT", 0)


def test_names():
    name = labels_filename("artifacts/splits/patient_windows_lookback4q_neg1x_seed0.csv", "WM", 1)
    assert name == "labels_enroll-WM_h1q_lookback4q_neg1x_seed0.csv"
    assert task_from_labels_file(f"artifacts/labels/{name}") == "enroll-WM_h1q"


def test_label_summary():
    summary = label_summary(splits(), enrollment_labels(splits(), enrollments(), "PT"))
    assert summary.loc["train"].tolist() == [7, 5, 2, 2, 0.4]
    assert summary.loc["holdout", "labelled"] == 0


@pytest.mark.skipif(not (DATA_DIR / "meridian_health_enrollments.csv").exists(), reason="extract not present")
def test_real_enrollments_load():
    e = load_enrollments(DATA_DIR)
    assert set(e.program) == {"PT", "WM"}
    assert e.enrollment_date.notna().all()
