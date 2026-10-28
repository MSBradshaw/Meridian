import numpy as np
import pandas as pd

from assignment.inference import TARGET_COLUMNS, eligibility, funnel, recommend
from assignment.utils import validate_targets

THRESHOLDS = {"PT": 0.8, "WM": 0.7}


def scores():
    """surgery_score, PT_score, WM_score per patient (PT/WM only scored for surgery positives)."""
    rows = {
        "both": (0.90, 0.85, 0.75),
        "pt_only": (0.80, 0.90, 0.10),
        "wm_only": (0.70, 0.10, 0.90),
        "neither": (0.60, 0.10, 0.10),
        "low_surgery": (0.40, np.nan, np.nan),  # not surgery positive: never scored for programs
        "edge": (0.50, 0.80, 0.70),  # exactly at every threshold -> positive
        "uncovered": (0.95, 0.90, 0.90),
        "operated": (0.94, 0.90, 0.90),
        "enrolled_pt": (0.93, 0.90, 0.90),
        "prior_pt": (0.92, 0.90, 0.10),
        "enrolled_wm": (0.91, 0.10, 0.90),
    }
    return pd.DataFrame(rows, index=["surgery_score", "PT_score", "WM_score"]).T


def eligible(**overrides):
    e = pd.DataFrame(False, index=scores().index, columns=["has_surgery", "enrolled_PT", "enrolled_WM", "prior_pt_care"])
    e.insert(0, "covered", True)
    e.loc["uncovered", "covered"] = False
    e.loc["operated", "has_surgery"] = True
    e.loc["enrolled_pt", "enrolled_PT"] = True
    e.loc["prior_pt", "prior_pt_care"] = True
    e.loc["enrolled_wm", "enrolled_WM"] = True
    return e


def result():
    return recommend(scores(), eligible(), 0.5, THRESHOLDS).set_index("patient_token")


def test_program_assignment_and_thresholds():
    r = result().program_recommended
    assert r["both"] == "PT+WM"
    assert r["pt_only"] == "PT"
    assert r["wm_only"] == "WM"
    assert r["edge"] == "PT+WM"  # thresholds are inclusive
    assert r["neither"] == "" and r["low_surgery"] == ""


def test_rules_remove_patients_or_programs():
    r = result().program_recommended
    assert r["uncovered"] == ""  # coverage not active
    assert r["operated"] == ""  # surgery evidence
    assert r["enrolled_pt"] == "WM"  # already in PT: only WM left
    assert r["prior_pt"] == ""  # prior PT care removes PT, WM was not positive
    assert r["enrolled_wm"] == ""  # already in WM, PT was not positive


def test_order_and_score_is_surgery_score():
    r = result()
    assert r.surgery_score.is_monotonic_decreasing
    assert (r.score == r.surgery_score).all()


def test_targets_format_validates():
    d = recommend(scores(), eligible(), 0.5, THRESHOLDS)
    targets = d.loc[d.program_recommended != "", TARGET_COLUMNS].reset_index(drop=True)
    validate_targets(targets)
    assert list(targets.patient_token) == ["enrolled_pt", "both", "pt_only", "wm_only", "edge"]


def test_funnel_counts():
    f = funnel(recommend(scores(), eligible(), 0.5, THRESHOLDS))
    assert f["holdout scored"] == 11
    assert f["surgery positive"] == 10
    assert f["on list"] == 5
    assert f["on list: PT+WM"] == 2 and f["on list: PT only"] == 1 and f["on list: WM only"] == 2
    assert f["dropped: not covered"] == 1 and f["dropped: surgery evidence"] == 1
    assert f["PT removed: enrolled in PT"] == 1 and f["PT removed: prior PT care"] == 1


def test_eligibility_from_tables():
    tokens = pd.Index(["a", "b", "c", "d", "e"])
    tables = {
        "patients": pd.DataFrame({
            "patient_token": ["a", "b", "c", "d", "e"],
            "coverage_start": pd.to_datetime(["2023-07-01"] * 5),
            "coverage_end": pd.to_datetime([None, "2026-03-01", None, None, None]),
        }),
        "encounters": pd.DataFrame({
            "patient_token": ["c", "d"], "event_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "event_category": ["Spine Surgery", "Imaging"], "event_code": [None, "72100"],
        }),
        "claims": pd.DataFrame({
            "patient_token": ["d", "e"], "service_date": pd.to_datetime(["2025-05-01", "2025-05-01"]),
            "code_system": ["CPT", "CPT"], "code": ["97110", "99213"], "provider_specialty": ["Physical Therapy", "Primary Care"],
        }),
        "enrollments": pd.DataFrame({
            "patient_token": ["e"], "program": ["WM"], "enrollment_date": pd.to_datetime(["2024-01-01"]),
        }),
    }
    e = eligibility(tokens, tables, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-06-30"))
    assert e.covered.tolist() == [True, False, True, True, True]
    assert e.has_surgery.tolist() == [False, False, True, False, False]  # Spine Surgery row with no code counts
    assert e.prior_pt_care.tolist() == [False, False, False, True, False]
    assert e.enrolled_WM.tolist() == [False, False, False, False, True]
    assert not e.enrolled_PT.any()
