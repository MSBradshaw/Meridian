"""End-to-end training: the three models behind the outreach list, from the raw extract.

Needs only the five extract files in `data_dir`. For each task it builds the
splits, the features and the model, scores the validation split (in
`assignment.train`) and the held-out test split (in `assignment.evaluate`):

    surgery     will this patient have lumbar surgery at least `gap_quarters`
                quarters from now? (`assignment.splits`)
    enroll-PT   enrolls in physical therapy vs refuses PT outreach, patients
                with no prior PT care (`assignment.enrollment_splits`)
    enroll-WM   enrolls in weight management vs refuses WM outreach

Defaults reproduce the agreed configuration (see DECISIONS.md): surgery
features from all history up to 4 quarters before surgery, enrollment features
from the 4 quarters before enrollment / refusal, code counts + provider specialties + age/sex, min-max scaled on train,
codes seen in more than 10 train patients, surgery and surgical consult codes
never used as features (plus PT referral codes for the PT model), XGBoost.

Everything goes under `artifacts_dir`:

    splits/  features/  models/<run>/   per-task outputs
    manifest.json                       which files belong to which task (for scoring)
    training_summary.csv                validation and test metrics per task

    from assignment.pipeline import train_all
    summary = train_all(Path("data"), Path("artifacts"))
"""

import json
from pathlib import Path

import pandas as pd

from assignment.enrollment_splits import DEFAULT_IGNORE_CODES, write_enrollment_splits
from assignment.evaluate import evaluate_test
from assignment.features import read_vocabulary, sidecar_path, vocabulary_path, write_features
from assignment.splits import SURGERY_CODES, write_splits
from assignment.train import train

TASKS = ("surgery", "enroll-PT", "enroll-WM")
GAP_QUARTERS = 4  # surgery: features stop this many quarters before surgery
LOOKBACK_QUARTERS = 4  # enrollment: features from this many quarters before the index date
MODEL = "xgboost"
SUMMARY_METRICS = ["patients", "positives", "positive_rate", "roc_auc", "average_precision", "brier", "lift_top_10%"]


def task_splits(
    task: str, gap_quarters: int, lookback_quarters: int, data_dir: Path, splits_dir: Path
) -> tuple[Path, tuple[str, ...]]:
    """Build the splits file for `task`; returns it and the codes to ignore as features."""
    if task == "surgery":
        return write_splits(gap_quarters, data_dir=data_dir, out_dir=splits_dir), tuple(SURGERY_CODES)
    program = task.removeprefix("enroll-")
    path = write_enrollment_splits(lookback_quarters, program, data_dir=data_dir, out_dir=splits_dir)
    return path, tuple(DEFAULT_IGNORE_CODES[program])


def train_task(
    task: str,
    data_dir: Path,
    artifacts_dir: Path,
    gap_quarters: int = GAP_QUARTERS,
    lookback_quarters: int = LOOKBACK_QUARTERS,
    model: str = MODEL,
) -> dict:
    """Splits -> features -> model -> val and test metrics for one task. Returns its manifest entry."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    splits_file, ignore_codes = task_splits(task, gap_quarters, lookback_quarters, data_dir, artifacts_dir / "splits")
    features_file = write_features(
        gap_quarters,
        splits_file=splits_file,
        data_dir=data_dir,
        out_dir=artifacts_dir / "features",
        ignore_codes=ignore_codes,
    )
    run_dir = train(features_file, out_dir=artifacts_dir / "models", model=model)
    test_metrics = evaluate_test(run_dir, features_file)
    val_report = json.loads((run_dir / "metrics_val.json").read_text())
    val_metrics = val_report["val"]
    return {
        "task": task,
        # the window used at scoring time: all history (None) for surgery, else the lookback
        "lookback_quarters": None if task == "surgery" else lookback_quarters,
        "gap_quarters": gap_quarters if task == "surgery" else 0,
        "model": model,
        "ignore_codes": list(ignore_codes),
        "splits_file": str(splits_file),
        "features_file": str(features_file),
        "vocabulary_file": str(vocabulary_path(features_file)),
        "scaling_file": str(sidecar_path(features_file, "scaling")),
        "run_dir": str(run_dir),
        "n_features": val_report["n_features"],
        "n_code_features": len(read_vocabulary(vocabulary_path(features_file))),
        "val": {k: val_metrics[k] for k in SUMMARY_METRICS},
        "test": {k: test_metrics[k] for k in SUMMARY_METRICS},
    }


def train_all(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    gap_quarters: int = GAP_QUARTERS,
    lookback_quarters: int = LOOKBACK_QUARTERS,
    model: str = MODEL,
    tasks: tuple[str, ...] = TASKS,
) -> pd.DataFrame:
    """Train every task, write `manifest.json` and `training_summary.csv`; returns the summary table."""
    entries = [train_task(task, data_dir, artifacts_dir, gap_quarters, lookback_quarters, model) for task in tasks]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "manifest.json").write_text(json.dumps({e["task"]: e for e in entries}, indent=2, default=float) + "\n")

    rows = []
    for e in entries:
        for split in ("val", "test"):
            rows.append({"task": e["task"], "split": split, **e[split]})
    summary = pd.DataFrame(rows)
    summary.to_csv(artifacts_dir / "training_summary.csv", index=False)
    return summary
