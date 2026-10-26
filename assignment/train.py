"""Train the surgery-risk model on a features file and report validation metrics.

Input is a features file from `assignment.features` (`patient_token, split,
label`, then 0/1 feature columns). Rows are separated on `split`:

    train      fits the model
    val        scores it; the metrics in the report come from here
    test       never read: dropped before anything else happens
    holdout    never read: dropped before anything else happens

The model is L2-regularized logistic regression: simple, interpretable, and
usually well calibrated on binary features.

Outputs go to `artifacts/models/<run name>/`, where the run name is the
features file name with `features_onehot_` replaced by `logreg_`:

    model.pkl          fitted model and its feature list
    metrics_val.json   validation metrics (plus train AUC as an overfitting check)
    calibration_val.csv  predicted vs observed surgery rate by decile of score
    coefficients.csv   feature weights, largest magnitude first

As a module:

    from assignment.train import train
    run_dir = train("artifacts/features/features_onehot_lookback4q_neg1x_seed0_excl-surgery_mintrain11.csv")

As a script:

    uv run python -m assignment.train --features-file artifacts/features/<file>.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ID_COLUMNS = ["patient_token", "split", "label"]
USED_SPLITS = ("train", "val")
MODELS_DIR = Path("artifacts/models")

C = 1.0  # inverse L2 strength; fixed, not tuned
TOP_FRACTIONS = (0.05, 0.10, 0.20)
CALIBRATION_BINS = 10


def read_features(path: Path | str) -> pd.DataFrame:
    """Read a features file, keeping only train and val rows. Test and holdout are never returned."""
    features = pd.read_csv(path, dtype={"patient_token": str, "split": str, "label": "Int8"})
    return features[features["split"].isin(USED_SPLITS)].reset_index(drop=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ID_COLUMNS]


def split_xy(features: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows = features[features["split"] == split]
    if rows.empty:
        raise ValueError(f"no {split} rows in features")
    if rows["label"].isna().any():
        raise ValueError(f"{split} rows have missing labels")
    return rows[feature_columns(features)], rows["label"].astype(int).to_numpy()


def fit_model(X: pd.DataFrame, y: np.ndarray, c: float = C) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000).fit(X, y)


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


def run_name(features_file: Path | str) -> str:
    return "logreg_" + Path(features_file).stem.removeprefix("features_onehot_")


def train(features_file: Path | str, out_dir: Path = MODELS_DIR, c: float = C) -> Path:
    """Fit on train, evaluate on val, write the model and reports; returns the run directory."""
    features = read_features(features_file)
    X_train, y_train = split_xy(features, "train")
    X_val, y_val = split_xy(features, "val")

    model = fit_model(X_train, y_train, c)
    val_score = model.predict_proba(X_val)[:, 1]
    train_score = model.predict_proba(X_train)[:, 1]

    report = {
        "features_file": str(features_file),
        "model": "LogisticRegression (L2)",
        "C": c,
        "n_features": X_train.shape[1],
        "train": {"patients": len(y_train), "positives": int(y_train.sum()), "roc_auc": roc_auc_score(y_train, train_score)},
        "val": evaluate(y_val, val_score),
    }

    run_dir = out_dir / run_name(features_file)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": list(X_train.columns)}, f)
    (run_dir / "metrics_val.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    calibration_table(y_val, val_score).to_csv(run_dir / "calibration_val.csv", index=False)
    coefficients = pd.DataFrame({"feature": X_train.columns, "coefficient": model.coef_[0]})
    coefficients.sort_values("coefficient", key=abs, ascending=False).to_csv(run_dir / "coefficients.csv", index=False)
    return run_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description="Train on train, report metrics on val.")
    parser.add_argument("--features-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--C", type=float, default=C, dest="c", help=f"inverse L2 strength (default {C})")
    args = parser.parse_args()

    run_dir = train(args.features_file, args.out_dir, args.c)
    report = json.loads((run_dir / "metrics_val.json").read_text())
    print(f"wrote {run_dir}")
    print(f"train AUC {report['train']['roc_auc']:.3f}")
    print("validation:")
    for key, value in report["val"].items():
        print(f"  {key:22s} {value:.4f}" if isinstance(value, float) else f"  {key:22s} {value}")


if __name__ == "__main__":
    cli()
