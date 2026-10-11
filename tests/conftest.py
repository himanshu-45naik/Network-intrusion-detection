"""Test fixtures for the inference service.

A tiny synthetic bundle is built here rather than loading the real 4 MB model,
so the API tests run in CI without the CICIDS2017 dataset and finish in
milliseconds. The bundle has the same shape as the real one, which is what the
service actually depends on.
"""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = [f"Feature {i}" for i in range(6)]
CLASSES = ["BENIGN", "DDoS", "Port Scan"]


def _build_bundle(path: str, samples_path: str) -> None:
    """Fits a 5-tree forest on separable synthetic data and writes a bundle."""
    rng = np.random.default_rng(0)
    n = 90
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    # Shift each class apart so the model is confidently right on the fixtures.
    y = pd.Series([CLASSES[i % 3] for i in range(n)])
    for idx, label in enumerate(CLASSES):
        X.loc[y == label, FEATURES[0]] += idx * 8

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]
    )
    pipeline.fit(X, y)

    reference_stats = {
        col: {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "bin_edges": [float(e) for e in np.unique(np.quantile(X[col], np.linspace(0, 1, 6)))],
        }
        for col in FEATURES
    }

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": FEATURES,
            "classes": [str(c) for c in pipeline.classes_],
            "benign_label": "BENIGN",
            "metrics": {
                "binary": {"recall": 1.0, "precision": 1.0, "false_positive_rate": 0.0},
                "multiclass": {"f1_macro": 1.0, "accuracy": 1.0},
            },
            "reference_stats": reference_stats,
            "top_features": {FEATURES[0]: 1.0},
            "trained_at": "2026-01-01T00:00:00+00:00",
            "git_sha": "testsha",
            "params": {"train_rows": n},
            "dataset": "synthetic",
        },
        path,
    )

    samples = [
        {
            "true_label": label,
            "features": {col: float(val) for col, val in X.loc[y == label].iloc[0].items()},
        }
        for label in CLASSES
    ]
    with open(samples_path, "w") as f:
        json.dump(samples, f)


# `serving.config` reads the environment at import time, and test modules import
# it at collection time, so the bundle is built and the environment set here at
# conftest import rather than inside a fixture.
_TMPDIR = tempfile.mkdtemp()
MODEL_PATH = os.path.join(_TMPDIR, "model.joblib")
SAMPLES_PATH = os.path.join(_TMPDIR, "samples.json")
_build_bundle(MODEL_PATH, SAMPLES_PATH)

os.environ["MODEL_PATH"] = MODEL_PATH
os.environ["SAMPLES_PATH"] = SAMPLES_PATH
os.environ["DRIFT_MIN_ROWS"] = "5"
os.environ["MAX_BATCH_SIZE"] = "10"


@pytest.fixture(scope="session")
def client():
    """A TestClient over the app, wired to the synthetic bundle."""
    from fastapi.testclient import TestClient

    from serving.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_features(client):
    """The feature map of the first demo sample."""
    return client.get("/samples").json()["samples"][0]["features"]
