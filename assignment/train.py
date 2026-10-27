"""Train the surgery-risk model on a features file and report validation metrics.

Input is a features file from `assignment.features` (`patient_token, split,
label`, then 0/1 feature columns). Rows are separated on `split`:

    train      fits the model
    val        scores it; the metrics in the report come from here
    test       never read: dropped before anything else happens
    holdout    never read: dropped before anything else happens

The task is the surgery label carried in the features file, unless
`labels_file` (from `assignment.labels`, e.g. program enrollment) is given: its
`label` replaces the features' one, and train/val patients it leaves unlabelled
(excluded for that task) are dropped and counted in the report.

Two models, both with fixed (untuned) settings:

    logreg     L2-regularized logistic regression (default): simple,
               interpretable, usually well calibrated
    xgboost    gradient-boosted trees: shallow, slow learning rate, row and
               column subsampling; can pick up interactions logreg cannot

Outputs go to `artifacts/models/<run name>/`, where the run name is the
features file name with `features_` replaced by `<model>_` (`logreg_`, `xgboost_`),
preceded by the task when a labels file is used (`xgboost_enroll-PT_h4q_count_...`):

    model.pkl          fitted model and its feature list
    metrics_val.json   validation metrics (plus train AUC as an overfitting check)
    calibration_val.csv  predicted vs observed surgery rate by decile of score
    coefficients.csv   logreg: feature weights, largest magnitude first
    feature_importance.csv  xgboost: gain importance per feature, largest first

As a module:

    from assignment.train import train
    run_dir = train("artifacts/features/features_count_minmax_lookback4q_neg1x_seed0_excl-surgery-consult_mintrain11_demo_spec.csv")

As a script:

    uv run python -m assignment.train --features-file artifacts/features/<file>.csv [--model xgboost]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from assignment.features import forbidden_columns
from assignment.labels import read_labels, task_from_labels_file

ID_COLUMNS = ["patient_token", "split", "label"]
USED_SPLITS = ("train", "val")
MODELS_DIR = Path("artifacts/models")

C = 1.0  # logreg inverse L2 strength; fixed, not tuned
XGBOOST_PARAMS = {  # fixed, not tuned: shallow trees, small steps, subsampling against overfitting
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 0,
    "n_jobs": 1,  # deterministic results
}
MODEL_TYPES = ("logreg", "xgboost")
TOP_FRACTIONS = (0.05, 0.10, 0.20)
CALIBRATION_BINS = 10


def read_features(path: Path | str, labels_file: Path | str | None = None) -> pd.DataFrame:
    """Read a features file, keeping only train and val rows. Test and holdout are never returned.

    With `labels_file`, its `label` replaces the features' surgery label (patients
    missing from it get no label).
    """
    features = pd.read_csv(path, dtype={"patient_token": str, "split": str, "label": "Int8"})
    forbidden = forbidden_columns(features.columns)
    if forbidden:
        raise ValueError(f"{path} has forbidden feature columns {forbidden}; rebuild it with assignment.features")
    features = features[features["split"].isin(USED_SPLITS)].reset_index(drop=True)
    if labels_file is not None:
        labels = read_labels(labels_file).set_index("patient_token")["label"]
        features["label"] = features["patient_token"].map(labels).astype("Int8")
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ID_COLUMNS]


def split_xy(features: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows = features[features["split"] == split]
    if rows.empty:
        raise ValueError(f"no {split} rows in features")
    if rows["label"].isna().any():
        raise ValueError(f"{split} rows have missing labels")
    return rows[feature_columns(features)], rows["label"].astype(int).to_numpy()


def fit_model(X: pd.DataFrame, y: np.ndarray, model: str = "logreg", c: float = C):
    if model == "logreg":
        return LogisticRegression(C=c, max_iter=5000).fit(X, y)
    if model == "xgboost":
        return XGBClassifier(**XGBOOST_PARAMS).fit(X, y)
    raise ValueError(f"model must be one of {MODEL_TYPES}")


def model_description(model: str, c: float) -> dict:
    if model == "logreg":
        return {"model": "LogisticRegression (L2)", "params": {"C": c}}
    return {"model": "XGBClassifier", "params": XGBOOST_PARAMS}


def feature_weights(fitted, columns: pd.Index) -> tuple[str, pd.DataFrame]:
    """(file name, table) of per-feature weights: logreg coefficients or xgboost gain importance."""
    if isinstance(fitted, LogisticRegression):
        table = pd.DataFrame({"feature": columns, "coefficient": fitted.coef_[0]})
        return "coefficients.csv", table.sort_values("coefficient", key=abs, ascending=False)
    gain = fitted.get_booster().get_score(importance_type="gain")
    table = pd.DataFrame({"feature": columns, "gain": [gain.get(c, 0.0) for c in columns]})
    return "feature_importance.csv", table.sort_values("gain", ascending=False)


def top_fraction_metrics(y: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    """Precision, recall and lift among the highest-scoring `fraction` of patients."""
    k = max(1, int(round(fraction * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    precision = y[top].mean()
    pct = f"{fraction:.0%}"
    return {
        f"precision_top_{pct}": precision,
        f"recall_top_{pct}": y[top].sum() / y.sum(),
        f"lift_top_{pct}": precision / y.mean(),
    }


def calibration_table(y: np.ndarray, score: np.ndarray, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Mean predicted vs observed positive rate by quantile bin of score (bin 1 = lowest scores)."""
    bin_id = pd.qcut(pd.Series(score).rank(method="first"), bins, labels=range(1, bins + 1))
    frame = pd.DataFrame({"bin": bin_id, "score": score, "label": y})
    return (
        frame.groupby("bin", observed=True)
        .agg(patients=("label", "size"), mean_predicted=("score", "mean"), observed_rate=("label", "mean"))
        .reset_index()
    )


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    metrics = {
        "patients": len(y),
        "positives": int(y.sum()),
        "positive_rate": y.mean(),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "brier": brier_score_loss(y, score),
        "log_loss": log_loss(y, score),
    }
    for fraction in TOP_FRACTIONS:
        metrics |= top_fraction_metrics(y, score, fraction)
    return metrics


def run_name(features_file: Path | str, model: str = "logreg", labels_file: Path | str | None = None) -> str:
    """`features_count_minmax_lookback4q_...` -> `logreg_count_minmax_lookback4q_...`, or
    `logreg_enroll-PT_h4q_count_minmax_lookback4q_...` with a labels file."""
    task = f"{task_from_labels_file(labels_file)}_" if labels_file is not None else ""
    return f"{model}_{task}" + Path(features_file).stem.removeprefix("features_")


def train(
    features_file: Path | str,
    out_dir: Path = MODELS_DIR,
    c: float = C,
    model: str = "logreg",
    labels_file: Path | str | None = None,
) -> Path:
    """Fit on train, evaluate on val, write the model and reports; returns the run directory."""
    features = read_features(features_file, labels_file)
    excluded = {}
    if labels_file is not None:
        unlabelled = features["label"].isna()
        excluded = features.loc[unlabelled, "split"].value_counts().reindex(USED_SPLITS, fill_value=0).to_dict()
        features = features[~unlabelled]
    X_train, y_train = split_xy(features, "train")
    X_val, y_val = split_xy(features, "val")

    fitted = fit_model(X_train, y_train, model, c)
    val_score = fitted.predict_proba(X_val)[:, 1]
    train_score = fitted.predict_proba(X_train)[:, 1]

    report = {
        "features_file": str(features_file),
        "labels_file": str(labels_file) if labels_file is not None else None,
        "task": task_from_labels_file(labels_file) if labels_file is not None else "surgery",
        "excluded_unlabelled": excluded,
        **model_description(model, c),
        "n_features": X_train.shape[1],
        "train": {"patients": len(y_train), "positives": int(y_train.sum()), "roc_auc": roc_auc_score(y_train, train_score)},
        "val": evaluate(y_val, val_score),
    }

    run_dir = out_dir / run_name(features_file, model, labels_file)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": fitted, "features": list(X_train.columns)}, f)
    (run_dir / "metrics_val.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    calibration_table(y_val, val_score).to_csv(run_dir / "calibration_val.csv", index=False)
    weights_file, weights = feature_weights(fitted, X_train.columns)
    weights.to_csv(run_dir / weights_file, index=False)
    return run_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description="Train on train, report metrics on val.")
    parser.add_argument("--features-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--model", choices=MODEL_TYPES, default="logreg")
    parser.add_argument("--labels-file", type=Path, help="task labels from assignment.labels (default: surgery)")
    parser.add_argument("--C", type=float, default=C, dest="c", help=f"logreg inverse L2 strength (default {C})")
    args = parser.parse_args()

    run_dir = train(args.features_file, args.out_dir, args.c, args.model, args.labels_file)
    report = json.loads((run_dir / "metrics_val.json").read_text())
    print(f"wrote {run_dir}")
    print(f"train AUC {report['train']['roc_auc']:.3f}")
    print("validation:")
    for key, value in report["val"].items():
        print(f"  {key:22s} {value:.4f}" if isinstance(value, float) else f"  {key:22s} {value}")


if __name__ == "__main__":
    cli()
