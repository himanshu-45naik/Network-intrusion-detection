"""Service configuration, read from the environment at import time."""

import os

# Path to the bundle written by scripts/train_serving_model.py.
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/nids_model.joblib")
SAMPLES_PATH = os.getenv("SAMPLES_PATH", "artifacts/samples.json")

# Default decision threshold on P(attack). 0.5 matches argmax on a balanced
# problem; a SOC drowning in false positives raises it, one that cannot miss an
# intrusion lowers it. Overridable per request.
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# Rows kept in the live feature histogram used for drift scoring.
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "5000"))

# Minimum live rows before a PSI number is reported at all.
DRIFT_MIN_ROWS = int(os.getenv("DRIFT_MIN_ROWS", "200"))

# PSI above this is conventionally read as a material distribution shift.
DRIFT_ALERT_PSI = float(os.getenv("DRIFT_ALERT_PSI", "0.2"))

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))
