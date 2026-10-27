import json
import pickle

import numpy as np
import pandas as pd
import pytest

from assignment.train import (
    calibration_table,
    read_features,
    run_name,
    top_fraction_metrics,
    train,
)


def features(seed=0, n=400):
    """Synthetic features: f_signal drives the label, f_noise does not."""
    rng = np.random.default_rng(seed)
    split = rng.choice(["train", "val", "test", "holdout"], size=n, p=[0.55, 0.15, 0.15, 0.15])
    signal = rng.integers(0, 2, n)
    noise = rng.integers(0, 2, n)
    label = (rng.random(n) < np.where(signal == 1, 0.8, 0.2)).astype(int)
    df = pd.DataFrame(
        {
            "patient_token": [f"p{i}" for i in range(n)],
            "split": split,
            "label": pd.array(label, dtype="Int8"),
            "f_noise": noise,
            "f_signal": signal,
        }
    )
    df.loc[df.split == "holdout", "label"] = pd.NA
    return df


def write(df, tmp_path, name="features_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain11_demo.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


PROBE = pd.DataFrame({"f_noise": [0, 1, 0, 1], "f_signal": [0, 0, 1, 1]})


def outputs(run_dir):
    """The fitted model's predictions on a fixed probe grid, and the metrics report."""
    with open(run_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    return model.predict_proba(PROBE)[:, 1], json.loads((run_dir / "metrics_val.json").read_text())


def test_read_features_drops_test_and_holdout(tmp_path):
    kept = read_features(write(features(), tmp_path))
    assert set(kept.split) == {"train", "val"}


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_test_and_holdout_rows_do_not_affect_the_model(tmp_path, model):
    base = features()
    coef_a, metrics_a = outputs(train(write(base, tmp_path), out_dir=tmp_path / "a", model=model))

    scrambled = base.copy()
    other = scrambled.split.isin(["test", "holdout"])
    scrambled.loc[other, ["f_noise", "f_signal"]] = 999  # nonsense values
    scrambled.loc[other, "label"] = pd.NA  # and no labels
    coef_b, metrics_b = outputs(train(write(scrambled, tmp_path), out_dir=tmp_path / "b", model=model))

    np.testing.assert_array_equal(coef_a, coef_b)
    metrics_a.pop("features_file"), metrics_b.pop("features_file")
    assert metrics_a == metrics_b


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_metrics_report_is_on_val(tmp_path, model):
    df = features()
    run_dir = train(write(df, tmp_path), out_dir=tmp_path, model=model)
    report = json.loads((run_dir / "metrics_val.json").read_text())
    assert report["val"]["patients"] == (df.split == "val").sum()
    assert report["train"]["patients"] == (df.split == "train").sum()
    assert report["n_features"] == 2
    for key in ["roc_auc", "average_precision", "brier", "log_loss", "precision_top_10%", "lift_top_10%"]:
        assert key in report["val"]
    assert report["val"]["roc_auc"] > 0.7  # the signal feature is learnable


def test_run_outputs(tmp_path):
    run_dir = train(write(features(), tmp_path), out_dir=tmp_path)
    assert run_dir.name == "logreg_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain11_demo"
    assert {p.name for p in run_dir.iterdir()} == {"model.pkl", "metrics_val.json", "calibration_val.csv", "coefficients.csv"}
    coef = pd.read_csv(run_dir / "coefficients.csv")
    assert coef.feature.iloc[0] == "f_signal"  # largest weight first
    calib = pd.read_csv(run_dir / "calibration_val.csv")
    assert calib.patients.sum() == (features().split == "val").sum()


def test_xgboost_run_outputs(tmp_path):
    run_dir = train(write(features(), tmp_path), out_dir=tmp_path, model="xgboost")
    assert run_dir.name == "xgboost_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain11_demo"
    assert {p.name for p in run_dir.iterdir()} == {"model.pkl", "metrics_val.json", "calibration_val.csv", "feature_importance.csv"}
    importance = pd.read_csv(run_dir / "feature_importance.csv")
    assert importance.feature.iloc[0] == "f_signal"
    report = json.loads((run_dir / "metrics_val.json").read_text())
    assert report["model"] == "XGBClassifier" and report["params"]["max_depth"] == 3


def test_unknown_model_fails(tmp_path):
    with pytest.raises(ValueError, match="model must be one of"):
        train(write(features(), tmp_path), out_dir=tmp_path, model="forest")


def test_missing_val_or_labels_fail_loudly(tmp_path):
    no_val = features().query("split != 'val'")
    with pytest.raises(ValueError, match="no val rows"):
        train(write(no_val, tmp_path), out_dir=tmp_path)
    unlabelled = features()
    unlabelled.loc[unlabelled.split == "train", "label"] = pd.NA
    with pytest.raises(ValueError, match="missing labels"):
        train(write(unlabelled, tmp_path), out_dir=tmp_path)


def test_top_fraction_metrics():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.linspace(1, 0, 10)  # first patient scores highest
    m = top_fraction_metrics(y, score, 0.2)
    assert m["precision_top_20%"] == 1.0
    assert m["recall_top_20%"] == pytest.approx(2 / 3)
    assert m["lift_top_20%"] == pytest.approx(1 / 0.3)


def test_calibration_table_bins():
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    table = calibration_table(y, np.linspace(0, 1, 10), bins=2)
    assert list(table.patients) == [5, 5]
    assert list(table.observed_rate) == [0.2, 0.8]


def test_run_name():
    assert run_name("x/features_count_minmax_lookback8q_neg1x_seed0_excl-none_mintrain11_demo.csv") == (
        "logreg_count_minmax_lookback8q_neg1x_seed0_excl-none_mintrain11_demo"
    )
    assert run_name("features_onehot_lookback4q_neg1x_seed0_excl-surgery_mintrain11.csv") == (
        "logreg_onehot_lookback4q_neg1x_seed0_excl-surgery_mintrain11"
    )


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_labels_file_replaces_label_and_drops_unlabelled(tmp_path, model):
    df = features()
    rng = np.random.default_rng(1)
    task = pd.DataFrame({"patient_token": df.patient_token, "label": pd.array(rng.integers(0, 2, len(df)), dtype="Int8")})
    task.loc[df.index[::10], "label"] = pd.NA  # excluded for this task
    labels_file = tmp_path / "labels_enroll-PT_h4q_lookback4q_neg1x_seed0.csv"
    task.to_csv(labels_file, index=False)

    run_dir = train(write(df, tmp_path), out_dir=tmp_path, model=model, labels_file=labels_file)
    assert run_dir.name == f"{model}_enroll-PT_h4q_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain11_demo"
    report = json.loads((run_dir / "metrics_val.json").read_text())
    assert report["task"] == "enroll-PT_h4q"
    kept = df.assign(label=task.label)[df.split.isin(["train", "val"]) & task.label.notna()]
    assert report["train"]["patients"] == (kept.split == "train").sum()
    assert report["val"]["patients"] == (kept.split == "val").sum()
    assert report["val"]["positives"] == int(kept.loc[kept.split == "val", "label"].sum())
    dropped = df[df.split.isin(["train", "val"]) & task.label.isna()].split.value_counts()
    assert report["excluded_unlabelled"] == {s: int(dropped.get(s, 0)) for s in ["train", "val"]}


def test_surgery_task_is_the_default(tmp_path):
    report = json.loads((train(write(features(), tmp_path), out_dir=tmp_path) / "metrics_val.json").read_text())
    assert report["task"] == "surgery" and report["labels_file"] is None and report["excluded_unlabelled"] == {}


def test_training_refuses_forbidden_feature_columns(tmp_path):
    with pytest.raises(ValueError, match="forbidden feature columns"):
        train(write(features().assign(CPT_99204=0), tmp_path), out_dir=tmp_path)
