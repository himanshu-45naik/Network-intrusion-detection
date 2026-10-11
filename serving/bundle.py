"""Loads the trained bundle and turns request payloads into predictions.

Everything the service needs at runtime — the fitted sklearn pipeline, the
expected feature order, the class names and the training reference statistics —
is in one joblib file, so the container has no dependency on MLflow or ZenML.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the bundle is loaded."""


class UnknownFeaturesError(ValueError):
    """Raised when a payload carries none of the expected features."""


class ModelBundle:
    """Wraps the fitted pipeline with the metadata needed to serve it."""

    def __init__(self, raw: Dict[str, Any]):
        self.pipeline = raw["pipeline"]
        self.feature_names: List[str] = raw["feature_names"]
        self.classes: List[str] = raw["classes"]
        self.benign_label: str = raw.get("benign_label", "BENIGN")
        self.metrics: Dict[str, Any] = raw.get("metrics", {})
        self.reference_stats: Dict[str, Any] = raw.get("reference_stats", {})
        self.top_features: Dict[str, float] = raw.get("top_features", {})
        self.trained_at: str = raw.get("trained_at", "unknown")
        self.git_sha: str = raw.get("git_sha", "unknown")
        self.params: Dict[str, Any] = raw.get("params", {})
        self.dataset: str = raw.get("dataset", "unknown")

        if self.benign_label not in self.classes:
            raise ValueError(
                f"Benign label {self.benign_label!r} missing from classes {self.classes}"
            )
        self._benign_index = self.classes.index(self.benign_label)

    @property
    def version(self) -> str:
        """Short human-readable version string, used in responses and metrics."""
        return f"{self.git_sha}-{self.trained_at[:10]}"

    def to_frame(self, payloads: List[Dict[str, float]]) -> pd.DataFrame:
        """Aligns request payloads to the column order the pipeline was fitted on.

        Missing features become NaN and are filled by the pipeline's imputer;
        unexpected keys are dropped. This keeps a client that sends a superset
        of features (a full CICFlowMeter row, say) working without changes.

        Args:
            payloads (List[Dict[str, float]]): One dict of feature values per flow.

        Returns:
            pd.DataFrame: Frame with exactly the fitted feature columns.

        Raises:
            UnknownFeaturesError: If no payload key matches a known feature.
        """
        frame = pd.DataFrame(payloads)
        frame.columns = [str(c).strip() for c in frame.columns]

        known = [c for c in frame.columns if c in self.feature_names]
        if not known:
            raise UnknownFeaturesError(
                "Payload contains none of the expected features. "
                f"Expected names such as {self.feature_names[:5]}"
            )

        aligned = frame.reindex(columns=self.feature_names)
        # Infinities would survive the imputer and reach the estimator as inf.
        return aligned.replace([np.inf, -np.inf], np.nan).astype("float64")

    def predict(self, frame: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Scores a frame and applies the attack threshold.

        The multiclass head gives the attack family; the binary verdict is
        `1 - P(BENIGN) >= threshold`, so raising the threshold trades recall for
        precision without retraining.

        Args:
            frame (pd.DataFrame): Aligned feature frame.
            threshold (float): Decision threshold on P(attack).

        Returns:
            List[Dict[str, Any]]: One prediction record per row.
        """
        proba = self.pipeline.predict_proba(frame)
        attack_proba = 1.0 - proba[:, self._benign_index]

        results = []
        for row_proba, p_attack in zip(proba, attack_proba):
            is_attack = bool(p_attack >= threshold)
            if is_attack:
                # Pick the most likely attack family, excluding BENIGN, so the
                # label never contradicts the verdict.
                attack_scores = np.delete(row_proba, self._benign_index)
                attack_labels = [c for c in self.classes if c != self.benign_label]
                attack_type = attack_labels[int(np.argmax(attack_scores))]
            else:
                attack_type = self.benign_label

            results.append(
                {
                    "verdict": "attack" if is_attack else "benign",
                    "attack_type": attack_type,
                    "attack_probability": float(p_attack),
                    "class_probabilities": {
                        cls: float(p) for cls, p in zip(self.classes, row_proba)
                    },
                }
            )
        return results

    def card(self) -> Dict[str, Any]:
        """Returns served-model metadata, without the estimator itself."""
        return {
            "version": self.version,
            "dataset": self.dataset,
            "trained_at": self.trained_at,
            "git_sha": self.git_sha,
            "classes": self.classes,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "top_features": self.top_features,
            "metrics": self.metrics,
            "params": self.params,
        }


def load_bundle(path: str) -> ModelBundle:
    """Loads a bundle from disk.

    Args:
        path (str): Path to the joblib bundle.

    Returns:
        ModelBundle: The loaded bundle.

    Raises:
        FileNotFoundError: If the bundle is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run scripts/train_serving_model.py first."
        )
    logger.info(f"Loading model bundle from {path}")
    return ModelBundle(joblib.load(path))


def load_samples(path: str) -> List[Dict[str, Any]]:
    """Loads the held-out demo rows, returning an empty list if absent."""
    if not os.path.exists(path):
        logger.warning(f"No demo samples at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def optional_load(path: str) -> Optional[ModelBundle]:
    """Loads a bundle, logging and swallowing failure.

    The service starts even without a model so `/health` can report the problem
    instead of the container crash-looping before anything is observable.
    """
    try:
        return load_bundle(path)
    except Exception as err:  # noqa: BLE001 - startup must stay observable
        logger.error(f"Model load failed: {err}")
        return None
