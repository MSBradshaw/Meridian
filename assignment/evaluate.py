"""Score a trained model on the held-out test split.

`assignment.train` never reads test rows; this is the one place they are used,
once per trained model, for a final unbiased estimate. Only `split == "test"`
rows are read from the features file.

Writes next to the model, in its run directory:

    metrics_test.json      the same metrics as metrics_val.json, on test
    calibration_test.csv   predicted vs observed rate by decile of test score
    predictions_test.csv   patient_token, label, score

    from assignment.evaluate import evaluate_test
    metrics = evaluate_test(run_dir, features_file)
"""

import json
import pickle
from pathlib import Path

import pandas as pd

from assignment.features import forbidden_columns
from assignment.train import calibration_table, evaluate

TEST = "test"


def read_test_rows(features_file: Path | str) -> pd.DataFrame:
    """The labelled test rows of a features file, and nothing else."""
    features = pd.read_csv(features_file, dtype={"patient_token": str, "split": str, "label": "Int8"})
    forbidden = forbidden_columns(features.columns)
    if forbidden:
        raise ValueError(f"{features_file} has forbidden feature columns {forbidden}")
    test = features[(features["split"] == TEST) & features["label"].notna()].reset_index(drop=True)
    if test.empty:
        raise ValueError(f"no labelled test rows in {features_file}")
    return test


def load_model(run_dir: Path | str) -> tuple[object, list[str]]:
    with open(Path(run_dir) / "model.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["features"]


def evaluate_test(run_dir: Path | str, features_file: Path | str) -> dict:
    """Score the test split with the model in `run_dir`, write the test reports, return the metrics."""
    run_dir = Path(run_dir)
    model, columns = load_model(run_dir)
    test = read_test_rows(features_file)
    missing = [c for c in columns if c not in test.columns]
    if missing:
        raise ValueError(f"features file lacks {len(missing)} model columns, e.g. {missing[:3]}")

    y = test["label"].astype(int).to_numpy()
    score = model.predict_proba(test[columns])[:, 1]
    metrics = evaluate(y, score)
    report = {"features_file": str(features_file), "split": TEST, **metrics}
    (run_dir / "metrics_test.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    calibration_table(y, score).to_csv(run_dir / "calibration_test.csv", index=False)
    pd.DataFrame({"patient_token": test["patient_token"], "label": y, "score": score}).to_csv(
        run_dir / "predictions_test.csv", index=False
    )
    return metrics
