"""In-process request metrics and feature-drift scoring.

Deliberately dependency-free: counters live in memory and are rendered in the
Prometheus text exposition format by `render_prometheus`. One replica's view is
enough for a single-container deployment, and a scrape every 15s is what makes
the numbers useful rather than where they are stored.

Drift uses the Population Stability Index against the decile edges captured at
training time. Only bin counts are kept, so memory stays flat no matter how many
requests arrive.
"""

import threading
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Latency bucket upper bounds in seconds, matching Prometheus histogram style.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Monitor:
    """Thread-safe counters, latency histogram and drift window for one replica."""

    def __init__(self, reference_stats: Dict[str, Any], window: int, min_rows: int):
        self._lock = threading.Lock()
        self._started = time.time()

        self.requests_total = Counter()
        self.predictions_total = Counter()
        self.errors_total = Counter()
        self.flows_scored = 0
        self.latency_sum = 0.0
        self.latency_count = 0
        self.latency_buckets = {b: 0 for b in LATENCY_BUCKETS}
        self.recent_latencies: deque = deque(maxlen=1000)

        self._reference = reference_stats or {}
        self._window = window
        self._min_rows = min_rows
        # Per-feature live bin counts, plus the ids of the rows contributing to
        # them so the window can be trimmed to the most recent `window` rows.
        self._live_bins: Dict[str, np.ndarray] = {}
        self._row_bins: deque = deque(maxlen=window)
        self._live_rows = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started

    def record_request(self, endpoint: str, status: int, duration: float) -> None:
        """Records one HTTP request's outcome and latency."""
        with self._lock:
            self.requests_total[(endpoint, status)] += 1
            self.latency_sum += duration
            self.latency_count += 1
            self.recent_latencies.append(duration)
            for bound in LATENCY_BUCKETS:
                if duration <= bound:
                    self.latency_buckets[bound] += 1

    def record_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Counts scored flows by predicted class and verdict."""
        with self._lock:
            self.flows_scored += len(predictions)
            for pred in predictions:
                self.predictions_total[pred["attack_type"]] += 1
                self.predictions_total[f"verdict:{pred['verdict']}"] += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.errors_total[kind] += 1

    def observe_features(self, frame: pd.DataFrame) -> None:
        """Adds a scored frame to the live drift window.

        Each value is binned against the training decile edges; only the counts
        are kept. Rolling the window subtracts the bin vector of the row that
        falls out, which keeps the update O(features) per row.

        Args:
            frame (pd.DataFrame): The aligned frame that was just scored.
        """
        if not self._reference:
            return

        features = [c for c in frame.columns if c in self._reference]
        if not features:
            return

        binned = {}
        for col in features:
            edges = self._reference[col]["bin_edges"]
            if len(edges) < 2:
                continue
            # Interior edges only: np.digitize then yields bin indices in
            # [0, len(edges)-1], matching the reference bin count.
            binned[col] = np.digitize(frame[col].to_numpy(dtype=float), edges[1:-1])

        with self._lock:
            for col, indices in binned.items():
                n_bins = max(len(self._reference[col]["bin_edges"]) - 1, 1)
                if col not in self._live_bins:
                    self._live_bins[col] = np.zeros(n_bins, dtype=np.int64)

            for row in range(len(frame)):
                row_bins = {col: int(indices[row]) for col, indices in binned.items()}
                if len(self._row_bins) == self._window and self._row_bins:
                    evicted = self._row_bins[0]
                    for col, idx in evicted.items():
                        if col in self._live_bins and self._live_bins[col][idx] > 0:
                            self._live_bins[col][idx] -= 1
                    self._live_rows -= 1
                self._row_bins.append(row_bins)
                for col, idx in row_bins.items():
                    bins = self._live_bins[col]
                    if 0 <= idx < len(bins):
                        bins[idx] += 1
                self._live_rows += 1

    def drift_report(self, top_n: int = 10) -> Dict[str, Any]:
        """Computes PSI per feature over the live window.

        PSI compares the live share of each bin with the training share. Below
        0.1 is conventionally read as stable, 0.1-0.2 as moderate and above 0.2
        as a material shift worth retraining on.

        Args:
            top_n (int): How many of the worst features to list.

        Returns:
            dict: Window size, worst features and whether the alert fires.
        """
        with self._lock:
            live_rows = self._live_rows
            live_bins = {k: v.copy() for k, v in self._live_bins.items()}

        if live_rows < self._min_rows:
            return {
                "status": "warming_up",
                "rows_observed": live_rows,
                "rows_required": self._min_rows,
                "features": [],
            }

        scores = []
        for col, counts in live_bins.items():
            n_bins = len(counts)
            if n_bins < 2 or counts.sum() == 0:
                continue
            live_share = counts / counts.sum()
            # The training reference is quantile-based, so each bin holds an
            # equal share by construction.
            expected_share = np.full(n_bins, 1.0 / n_bins)
            # Floor both sides: a zero bin otherwise makes the log term infinite.
            eps = 1e-6
            live_share = np.clip(live_share, eps, None)
            expected_share = np.clip(expected_share, eps, None)
            psi = float(np.sum((live_share - expected_share) * np.log(live_share / expected_share)))
            scores.append({"feature": col, "psi": round(psi, 4)})

        scores.sort(key=lambda s: s["psi"], reverse=True)
        max_psi = scores[0]["psi"] if scores else 0.0
        return {
            "status": "ok",
            "rows_observed": live_rows,
            "max_psi": max_psi,
            "features": scores[:top_n],
        }

    def snapshot(self, model_version: Optional[str]) -> Dict[str, Any]:
        """Returns a JSON-friendly summary of the counters."""
        with self._lock:
            latencies = sorted(self.recent_latencies)
            p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            return {
                "model_version": model_version,
                "uptime_seconds": round(self.uptime_seconds, 1),
                "requests_total": sum(self.requests_total.values()),
                "flows_scored_total": self.flows_scored,
                "errors_total": dict(self.errors_total),
                "predictions_total": dict(self.predictions_total),
                "latency_ms": {"p50": round(p50 * 1000, 2), "p95": round(p95 * 1000, 2)},
            }

    def render_prometheus(self, model_version: Optional[str]) -> str:
        """Renders the counters in the Prometheus text exposition format."""
        version = model_version or "none"
        with self._lock:
            lines = [
                "# HELP nids_build_info Served model version.",
                "# TYPE nids_build_info gauge",
                f'nids_build_info{{model_version="{version}"}} 1',
                "# HELP nids_uptime_seconds Seconds since the service started.",
                "# TYPE nids_uptime_seconds gauge",
                f"nids_uptime_seconds {self.uptime_seconds:.1f}",
                "# HELP nids_requests_total HTTP requests by endpoint and status.",
                "# TYPE nids_requests_total counter",
            ]
            for (endpoint, status), count in sorted(self.requests_total.items()):
                lines.append(
                    f'nids_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
                )

            lines += [
                "# HELP nids_flows_scored_total Network flows scored.",
                "# TYPE nids_flows_scored_total counter",
                f"nids_flows_scored_total {self.flows_scored}",
                "# HELP nids_predictions_total Predictions by class and verdict.",
                "# TYPE nids_predictions_total counter",
            ]
            for label, count in sorted(self.predictions_total.items()):
                lines.append(f'nids_predictions_total{{class="{label}"}} {count}')

            lines += [
                "# HELP nids_errors_total Errors by kind.",
                "# TYPE nids_errors_total counter",
            ]
            for kind, count in sorted(self.errors_total.items()):
                lines.append(f'nids_errors_total{{kind="{kind}"}} {count}')

            lines += [
                "# HELP nids_request_latency_seconds Request latency.",
                "# TYPE nids_request_latency_seconds histogram",
            ]
            for bound in LATENCY_BUCKETS:
                lines.append(
                    f'nids_request_latency_seconds_bucket{{le="{bound}"}} '
                    f"{self.latency_buckets[bound]}"
                )
            lines.append(f'nids_request_latency_seconds_bucket{{le="+Inf"}} {self.latency_count}')
            lines.append(f"nids_request_latency_seconds_sum {self.latency_sum:.6f}")
            lines.append(f"nids_request_latency_seconds_count {self.latency_count}")

        drift = self.drift_report(top_n=5)
        lines += [
            "# HELP nids_feature_drift_psi Population Stability Index vs training data.",
            "# TYPE nids_feature_drift_psi gauge",
        ]
        if drift["status"] == "ok":
            lines.append(f"nids_feature_drift_psi_max {drift['max_psi']}")
            for item in drift["features"]:
                feature = item["feature"].replace('"', "")
                lines.append(f'nids_feature_drift_psi{{feature="{feature}"}} {item["psi"]}')

        return "\n".join(lines) + "\n"
