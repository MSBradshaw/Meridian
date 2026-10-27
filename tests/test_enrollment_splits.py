import pandas as pd
import pytest

from assignment.enrollment_splits import DEFAULT_IGNORE_CODES, build_enrollment_splits, enrollment_splits_filename
from assignment.features import ignore_tag
from assignment.splits import HOLDOUT


def tok(i: int) -> str:
    return f"{i:032x}"


# named patients -> tokens spread over split buckets
P = {name: tok(i * 37) for i, name in enumerate([
    "pos_nosurg", "pos_presurg", "post_surg_enroll", "neg", "neg_refused_then_enrolled", "neg_after_surg",
    "neg_other_prog", "contacted_no_refusal", "early_pos", "late_pos", "plain",
    "pos_prior_pt", "neg_prior_pt", "pos_pt_after", "hold_prior_pt",
])}


def tables():
    patients = pd.DataFrame({"patient_token": list(P.values()), "coverage_start": pd.Timestamp("2023-07-01"), "coverage_end": pd.NaT})
    claims = pd.DataFrame(
        [(t, "2026-05-01", "CPT", "99213", "Primary Care") for t in P.values()]  # everyone recently active
        + [
            (P["pos_presurg"], "2025-09-01", "CPT", "63030", "Orthopedic Surgery"),
            (P["post_surg_enroll"], "2024-09-01", "CPT", "22630", "Orthopedic Surgery"),
            (P["neg_after_surg"], "2024-08-01", "CPT", "22630", "Orthopedic Surgery"),
            (P["pos_prior_pt"], "2024-10-01", "CPT", "97110", "Physical Therapy"),  # before 2025-03-01 enrollment
            (P["neg_prior_pt"], "2024-01-01", "CPT", "97140", "Physical Therapy"),  # before 2024-09-01 refusal
            (P["pos_pt_after"], "2025-04-01", "CPT", "97530", "Physical Therapy"),  # program sessions after enrollment
            (P["hold_prior_pt"], "2025-11-01", "CPT", "97110", "Physical Therapy"),  # before the holdout index date
        ],
        columns=["patient_token", "service_date", "code_system", "code", "provider_specialty"],
    ).astype({"service_date": "datetime64[ns]"})
    encounters = pd.DataFrame(columns=["patient_token", "event_date", "event_category", "event_code"]).astype(
        {"event_date": "datetime64[ns]"}
    )
    enrollments = pd.DataFrame(
        [
            (P["pos_nosurg"], "PT", "2025-03-10"),
            (P["pos_nosurg"], "PT", "2025-09-10"),  # later repeat: index stays the first
            (P["pos_presurg"], "PT", "2025-05-05"),  # before 2025-09-01 surgery
            (P["post_surg_enroll"], "PT", "2025-01-01"),  # after 2024-09-01 surgery
            (P["neg_refused_then_enrolled"], "PT", "2025-06-01"),
            (P["early_pos"], "PT", "2023-09-01"),  # before any outreach period
            (P["late_pos"], "PT", "2026-05-01"),  # after the last refusal
            (P["neg_other_prog"], "WM", "2025-01-01"),
            (P["pos_prior_pt"], "PT", "2025-03-01"),
            (P["pos_pt_after"], "PT", "2025-03-15"),
            (P["pos_prior_pt"], "WM", "2025-05-01"),
        ],
        columns=["patient_token", "program", "enrollment_date"],
    ).astype({"enrollment_date": "datetime64[ns]"})
    outreach = pd.DataFrame(
        [
            (P["neg"], "2024-08-15 10:00", "PT", "No Answer"),
            (P["neg"], "2024-08-20 11:30", "PT", "Declined"),
            (P["neg"], "2025-01-05 09:00", "PT", "Opted Out"),
            (P["neg_refused_then_enrolled"], "2025-02-01 10:00", "PT", "Declined"),
            (P["neg_after_surg"], "2025-01-01 10:00", "PT", "Unsubscribed"),
            (P["neg_other_prog"], "2025-02-01 10:00", "WM", "Declined"),
            (P["contacted_no_refusal"], "2025-02-01 10:00", "PT", "Voicemail"),
            (P["plain"], "2025-12-01 10:00", "PT", "Opted Out"),
            (P["neg_prior_pt"], "2024-09-01 10:00", "PT", "Declined"),
        ],
        columns=["patient_token", "timestamp", "program", "attempt_result"],
    ).astype({"timestamp": "datetime64[ns]"})
    return {"patients": patients, "encounters": encounters, "claims": claims, "enrollments": enrollments, "outreach": outreach}


def build(program="PT", **kwargs):
    return build_enrollment_splits(**tables(), program=program, lookback_quarters=4, **kwargs).set_index("patient_token")


def test_positives_enrolled_before_surgery_or_never_had_surgery():
    s = build()
    assert s.loc[P["pos_nosurg"], "label"] == 1
    assert s.loc[P["pos_nosurg"], "index_date"] == pd.Timestamp("2025-03-10")
    assert s.loc[P["pos_presurg"], "label"] == 1
    assert P["post_surg_enroll"] not in s.index  # enrolled after surgery: not labelled, and has surgery so not holdout


def test_negatives_refused_never_enrolled_before_surgery():
    s = build()
    assert s.loc[P["neg"], "label"] == 0
    assert s.loc[P["neg"], "index_date"] == pd.Timestamp("2024-08-20")  # first refusal, not first contact
    assert s.loc[P["neg_refused_then_enrolled"], "label"] == 1  # enrolled -> a positive, never a negative
    assert P["neg_after_surg"] not in s.index
    assert s.loc[P["neg_other_prog"], "split"] == HOLDOUT  # refused WM, so not a PT negative
    assert s.loc[P["contacted_no_refusal"], "split"] == HOLDOUT  # voicemail is not a refusal


def test_program_specific():
    s = build("WM")
    assert s.loc[P["neg_other_prog"], "label"] == 1  # enrolled in WM 2025-01-01, before the 2025-02-01 refusal of WM
    assert s.loc[P["neg"], "split"] == HOLDOUT


def test_match_period_limits_positives_to_negative_index_range():
    s = build()
    assert s.loc[P["early_pos"], "split"] == HOLDOUT  # 2023-09 is before the first refusal (2024-08-20)
    assert s.loc[P["late_pos"], "split"] == HOLDOUT  # 2026-05 is after the last refusal (2025-12-01)
    unmatched = build(match_period=False)
    assert unmatched.loc[P["early_pos"], "label"] == 1 and unmatched.loc[P["late_pos"], "label"] == 1


def test_windows_and_no_overlap():
    s = build().reset_index()
    assert not s.patient_token.duplicated().any()
    assert (s.end_date == s.index_date - pd.Timedelta(days=1)).all()
    assert (s.start_date == s.index_date - pd.DateOffset(months=12)).all()
    labelled = s[s.label.notna()]
    assert set(labelled.split) <= {"train", "val", "test"}
    assert not set(labelled.patient_token) & set(s.loc[s.split == HOLDOUT, "patient_token"])


def test_bad_arguments_and_filename():
    with pytest.raises(ValueError, match="program"):
        build("XX")
    assert enrollment_splits_filename("WM", 6) == "patient_windows_enroll-WM_lookback6q.csv"
    assert enrollment_splits_filename("PT", 4) == "patient_windows_enroll-PT_noprior-physical_therapy_lookback4q.csv"
    assert enrollment_splits_filename("PT", 4, match_period=False, exclude_prior_specialties=()) == (
        "patient_windows_enroll-PT_allperiod_lookback4q.csv"
    )
    assert enrollment_splits_filename("WM", 4, exclude_prior_specialties=["Physical Therapy", "Chiropractic"]) == (
        "patient_windows_enroll-WM_noprior-chiropractic-physical_therapy_lookback4q.csv"
    )


def test_prior_physical_therapy_excluded_from_pt_cohort_by_default():
    s = build()
    for name in ["pos_prior_pt", "neg_prior_pt", "hold_prior_pt"]:
        assert P[name] not in s.index  # PT claims before the index date: out of every split
    assert s.loc[P["pos_pt_after"], "label"] == 1  # PT claims only after enrolling: kept
    kept = build(exclude_prior_specialties=())
    assert kept.loc[P["pos_prior_pt"], "label"] == 1 and kept.loc[P["neg_prior_pt"], "label"] == 0
    assert kept.loc[P["hold_prior_pt"], "split"] == HOLDOUT


def test_wm_keeps_prior_pt_by_default():
    s = build("WM")
    assert s.loc[P["pos_prior_pt"], "label"] == 1  # prior PT is not excluded from WM
    assert P["pos_prior_pt"] not in build("WM", exclude_prior_specialties=["Physical Therapy"]).index


def test_default_ignore_codes():
    assert set(DEFAULT_IGNORE_CODES["PT"]) == {"22630", "22633", "63030", "63047", "97001", "97110"}
    assert set(DEFAULT_IGNORE_CODES["WM"]) == {"22630", "22633", "63030", "63047"}
    assert ignore_tag(DEFAULT_IGNORE_CODES["PT"]) == "excl-surgery-consult-ptref"
    assert ignore_tag(reversed(DEFAULT_IGNORE_CODES["PT"])) == "excl-surgery-consult-ptref"
    assert ignore_tag(DEFAULT_IGNORE_CODES["WM"]) == "excl-surgery-consult"
