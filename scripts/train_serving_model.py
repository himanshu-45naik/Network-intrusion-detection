"""Trains the single model that the inference service serves.

The ZenML pipeline in `pipelines/training_pipeline.py` is the experimentation
surface: it trains eight model/target combinations so they can be compared in
MLflow. This script is the promotion step. It trains one multiclass Random
Forest on the attack-type target and writes a self-contained bundle to
`artifacts/` that `serving/` loads at startup.

The binary verdict (attack vs benign) is derived from the multiclass prediction
rather than trained separately, so one model answers both questions and the
served artifact stays small.

Transformations are imported from `src/` so the serving model is built by the
same code as the pipeline runs.

Usage:
    python scripts/train_serving_model.py --sample-size 0.1
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import time
import zipfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_handling import (
    DownCasting,
    DropDuplicateValues,
    Handler,
    ReplaceFeatureNames,
    ReplaceInfinteValues,
)
from src.dropfeatures import DropOneValueFeature
from src.sample_data import Sampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET = "Attack Type"
BENIGN_LABEL = "BENIGN"
ARTIFACT_DIR = "artifacts"
RARE_CLASS_THRESHOLD = 100
DRIFT_REFERENCE_BINS = 10


def load_raw_frame(data_path: str) -> pd.DataFrame:
    """Reads the CICIDS2017 CSVs from a zip or an already-extracted directory.

    Args:
        data_path (str): Path to the dataset zip or a directory holding the CSVs.

    Returns:
        pd.DataFrame: Every CSV concatenated into one frame.
    """
    if data_path.endswith(".zip"):
        logger.info(f"Extracting {data_path}")
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")
        search_root = "extracted_data"
    else:
        search_root = data_path

    csv_files = sorted(glob.glob(os.path.join(search_root, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {search_root}")

    logger.info(f"Reading {len(csv_files)} CSV files")
    return pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the split-independent cleaning the ZenML pipeline applies.

    Column names are stripped, raw labels are collapsed into attack families,
    duplicate rows are dropped, infinities become NaN and numeric columns are
    downcast. No statistic is fitted here, so this is safe pre-split.

    Args:
        df (pd.DataFrame): Raw concatenated frame.

    Returns:
        pd.DataFrame: Cleaned frame carrying the `Attack Type` target.
    """
    df = Handler(ReplaceFeatureNames()).execute_strategy(df, df.columns)
    df = Handler(DropDuplicateValues()).execute_strategy(df, features=None)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    inf_cols = [col for col in numeric_cols if df[col].isin([np.inf, -np.inf]).any()]
    if inf_cols:
        df = Handler(ReplaceInfinteValues()).execute_strategy(df, inf_cols)
        logger.info(f"Replaced infinities with NaN in {len(inf_cols)} columns")

    df = Handler(DownCasting()).execute_strategy(df, features=None)

    # Rows whose raw label is outside the attack map become NaN targets.
    unmapped = int(df[TARGET].isna().sum())
    if unmapped:
        logger.warning(f"Dropping {unmapped} rows with an unmapped label")
        df = df[df[TARGET].notna()]

    return df


def merge_rare_classes(labels: pd.Series, threshold: int = RARE_CLASS_THRESHOLD) -> pd.Series:
    """Folds classes with too few examples into a single `Other Attack` class.

    Heartbleed and Infiltration have a few dozen rows in CICIDS2017. Kept
    separate they produce a class the model cannot learn and a metric that
    swings on single predictions.

    Args:
        labels (pd.Series): Attack-type labels.
        threshold (int): Minimum rows for a class to stay on its own.

    Returns:
        pd.Series: Labels with rare classes merged.
    """
    counts = labels.value_counts()
    rare = counts[counts < threshold].index.tolist()
    if not rare:
        return labels.astype(str)

    logger.info(f"Merging rare classes into 'Other Attack': {rare}")
    return labels.astype(str).where(~labels.astype(str).isin(rare), "Other Attack")


def build_reference_stats(X: pd.DataFrame) -> dict:
    """Captures the training distribution used for drift detection at serve time.

    Stores decile edges plus mean/std per feature. The service bins live
    requests against these edges and reports a Population Stability Index, so
    drift is computed without shipping the training data.

    Args:
        X (pd.DataFrame): Training features.

    Returns:
        dict: Per-feature reference statistics.
    """
    quantiles = np.linspace(0, 1, DRIFT_REFERENCE_BINS + 1)
    stats = {}
    for col in X.columns:
        series = X[col].dropna()
        if series.empty:
            continue
        edges = np.unique(np.quantile(series, quantiles))
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "bin_edges": [float(e) for e in edges],
        }
    return stats


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Scores the trained pipeline on multiclass and derived-binary terms.

    Args:
        pipeline (Pipeline): Fitted inference pipeline.
        X_test (pd.DataFrame): Held-out features.
        y_test (pd.Series): Held-out attack-type labels.

    Returns:
        dict: Metrics plus the per-class report and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "multiclass": {
            "accuracy": float((y_pred == y_test.to_numpy()).mean()),
            "precision_weighted": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
    }

    # Binary view: anything that is not BENIGN is an attack. This is the number
    # a SOC cares about, so it is reported separately from the macro average.
    y_true_attack = (y_test.to_numpy() != BENIGN_LABEL).astype(int)
    y_pred_attack = (y_pred != BENIGN_LABEL).astype(int)
    benign_index = list(pipeline.classes_).index(BENIGN_LABEL)
    attack_proba = 1.0 - pipeline.predict_proba(X_test)[:, benign_index]

    metrics["binary"] = {
        "precision": float(precision_score(y_true_attack, y_pred_attack, zero_division=0)),
        "recall": float(recall_score(y_true_attack, y_pred_attack, zero_division=0)),
        "f1": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_attack, attack_proba)),
        "false_positive_rate": float(
            ((y_pred_attack == 1) & (y_true_attack == 0)).sum() / max((y_true_attack == 0).sum(), 1)
        ),
    }

    labels = sorted(set(y_test) | set(y_pred))
    metrics["per_class"] = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    metrics["confusion_matrix"] = {
        "labels": labels,
        "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }
    return metrics


def git_sha() -> str:
    """Returns the current commit sha, or `unknown` outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_to_mlflow(metrics: dict, params: dict, pipeline: Pipeline) -> None:
    """Logs the promoted model to MLflow, treating an unreachable server as non-fatal.

    Training must not fail because a local tracking server is down, so every
    MLflow error is caught and logged.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URL")
    if not tracking_uri:
        logger.info("MLFLOW_TRACKING_URL not set, skipping MLflow logging")
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "intrusion_detection"))
        with mlflow.start_run(run_name="serving_candidate"):
            mlflow.log_params(params)
            mlflow.log_metrics({f"multiclass_{k}": v for k, v in metrics["multiclass"].items()})
            mlflow.log_metrics({f"binary_{k}": v for k, v in metrics["binary"].items()})
            mlflow.sklearn.log_model(pipeline, "nids_detector")
        logger.info("Logged promoted model to MLflow")
    except Exception as err:  # noqa: BLE001 - tracking is best-effort
        logger.warning(f"MLflow logging failed, continuing: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", "extracted_data"),
        help="Dataset zip or directory of extracted CSVs.",
    )
    parser.add_argument(
        "--sample-size",
        type=float,
        default=0.1,
        help="Stratified fraction of the dataset to train on.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-samples-export",
        type=int,
        default=12,
        help="Number of held-out rows exported for the demo UI.",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    raw_df = load_raw_frame(args.data_path)
    logger.info(f"Raw rows: {len(raw_df):,}")

    df = clean_frame(raw_df)
    logger.info(f"Rows after cleaning: {len(df):,}")

    df = Sampler.execute_sampling(df, args.sample_size, args.random_state).copy()
    df[TARGET] = merge_rare_classes(df[TARGET])

    # Constant columns carry no signal and would otherwise have to be sent in
    # every inference request.
    df = DropOneValueFeature().drop_features(df, df.nunique())

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    logger.info(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")

    # Imputation lives inside the pipeline so the saved artifact accepts raw
    # feature values and no fill statistic has to be recomputed at serve time.
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_samples=0.9,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    logger.info("Fitting pipeline")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating")
    metrics = evaluate(pipeline, X_test, y_test)
    logger.info(
        "binary recall=%.4f precision=%.4f fpr=%.5f | multiclass f1_macro=%.4f",
        metrics["binary"]["recall"],
        metrics["binary"]["precision"],
        metrics["binary"]["false_positive_rate"],
        metrics["multiclass"]["f1_macro"],
    )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    importances = pipeline.named_steps["rf"].feature_importances_
    top_features = (
        pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(15)
    )

    bundle = {
        "pipeline": pipeline,
        "feature_names": list(X_train.columns),
        "classes": [str(c) for c in pipeline.classes_],
        "benign_label": BENIGN_LABEL,
        "metrics": metrics,
        "reference_stats": build_reference_stats(X_train),
        "top_features": {k: float(v) for k, v in top_features.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "params": {
            "sample_size": args.sample_size,
            "test_size": args.test_size,
            "n_estimators": args.n_estimators,
            "random_state": args.random_state,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "rare_class_threshold": RARE_CLASS_THRESHOLD,
        },
        "dataset": "CICIDS2017 (MachineLearningCVE)",
    }

    bundle_path = os.path.join(ARTIFACT_DIR, "nids_model.joblib")
    joblib.dump(bundle, bundle_path, compress=3)
    logger.info(f"Wrote {bundle_path} ({os.path.getsize(bundle_path) / 1e6:.1f} MB)")

    # Model card: everything in the bundle except the estimator, so the service
    # and the README can read metadata without loading the model.
    card = {k: v for k, v in bundle.items() if k not in {"pipeline", "reference_stats"}}
    with open(os.path.join(ARTIFACT_DIR, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)

    # Held-out rows for the demo UI, one per class where available.
    samples = []
    for label in bundle["classes"]:
        matching = X_test[y_test == label]
        if matching.empty:
            continue
        row = matching.iloc[0]
        samples.append(
            {
                "true_label": label,
                "features": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
            }
        )
    samples = samples[: args.n_samples_export]
    with open(os.path.join(ARTIFACT_DIR, "samples.json"), "w") as f:
        json.dump(samples, f)
    logger.info(f"Exported {len(samples)} demo samples")

    log_to_mlflow(metrics, bundle["params"], pipeline)

    logger.info(f"Done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
