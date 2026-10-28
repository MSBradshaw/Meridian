import json

import pandas as pd
import pytest

from assignment.evaluate import evaluate_test, read_test_rows
from assignment.train import train
from test_train import features, write


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_evaluate_test_scores_only_test_rows(tmp_path, model):
    df = features()
    path = write(df, tmp_path)
    run_dir = train(path, out_dir=tmp_path, model=model)
    metrics = evaluate_test(run_dir, path)

    test = df[df.split == "test"]
    assert metrics["patients"] == len(test)
    assert metrics["positives"] == int(test.label.sum())
    assert metrics["roc_auc"] > 0.7  # the signal feature is learnable
    preds = pd.read_csv(run_dir / "predictions_test.csv")
    assert set(preds.patient_token) == set(test.patient_token)
    assert preds.score.between(0, 1).all()
    report = json.loads((run_dir / "metrics_test.json").read_text())
    assert report["split"] == "test" and report["patients"] == len(test)
    assert pd.read_csv(run_dir / "calibration_test.csv").patients.sum() == len(test)


def test_train_val_and_holdout_rows_do_not_affect_test_scores(tmp_path):
    df = features()
    path = write(df, tmp_path)
    run_dir = train(path, out_dir=tmp_path)
    a = evaluate_test(run_dir, path)
    scrambled = df.copy()
    other = scrambled.split != "test"
    scrambled.loc[other, ["f_noise", "f_signal"]] = 999
    b = evaluate_test(run_dir, write(scrambled, tmp_path, name="features_scrambled.csv"))
    assert a == b


def test_read_test_rows_guards(tmp_path):
    no_test = write(features().query("split != 'test'"), tmp_path, name="features_no_test.csv")
    with pytest.raises(ValueError, match="no labelled test rows"):
        read_test_rows(no_test)
    forbidden = write(features().assign(CPT_99244=0), tmp_path, name="features_forbidden.csv")
    with pytest.raises(ValueError, match="forbidden"):
        read_test_rows(forbidden)
