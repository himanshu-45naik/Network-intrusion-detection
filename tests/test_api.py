"""Contract tests for the inference service.

These cover the behaviour a caller depends on: the probe endpoints, the
prediction contract, the threshold, and the failure modes that would otherwise
return a confident answer from a malformed request.
"""

import pytest

from serving import config


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("testsha")


def test_ready_returns_200_when_model_loaded(client):
    assert client.get("/ready").status_code == 200


def test_model_card_exposes_metrics_and_features(client):
    card = client.get("/model").json()
    assert card["n_features"] == len(card["feature_names"])
    assert "BENIGN" in card["classes"]
    assert card["metrics"]["binary"]["recall"] == pytest.approx(1.0)


def test_predict_returns_benign_for_a_benign_flow(client, sample_features):
    body = client.post("/predict", json={"features": sample_features}).json()
    assert body["verdict"] == "benign"
    assert body["attack_type"] == "BENIGN"
    assert body["attack_probability"] < 0.5
    assert body["missing_features"] == []
    assert body["unknown_features"] == []
    # Probabilities are a full distribution over the known classes.
    assert pytest.approx(sum(body["class_probabilities"].values()), abs=1e-6) == 1.0


def test_predict_classifies_an_attack_flow(client):
    samples = client.get("/samples").json()["samples"]
    attack = next(s for s in samples if s["true_label"] != "BENIGN")
    body = client.post("/predict", json={"features": attack["features"]}).json()
    assert body["verdict"] == "attack"
    assert body["attack_type"] == attack["true_label"]


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_verdict_follows_the_threshold_rule(client, threshold):
    """The verdict is exactly `P(attack) >= threshold` at every threshold."""
    samples = client.get("/samples").json()["samples"]
    attack = next(s for s in samples if s["true_label"] != "BENIGN")
    body = client.post(
        "/predict", json={"features": attack["features"], "threshold": threshold}
    ).json()
    assert body["threshold"] == threshold
    expected = "attack" if body["attack_probability"] >= threshold else "benign"
    assert body["verdict"] == expected


def test_low_threshold_flags_a_benign_flow(client, sample_features):
    body = client.post("/predict", json={"features": sample_features, "threshold": 0.0}).json()
    assert body["verdict"] == "attack"
    # The reported family must never be BENIGN once the verdict is attack.
    assert body["attack_type"] != "BENIGN"


def test_partial_payload_is_imputed_and_reported(client, sample_features):
    first_key = next(iter(sample_features))
    body = client.post(
        "/predict", json={"features": {first_key: sample_features[first_key]}}
    ).json()
    assert len(body["missing_features"]) == body_feature_count(client) - 1
    assert body["verdict"] in {"attack", "benign"}


def body_feature_count(client) -> int:
    return client.get("/model").json()["n_features"]


def test_unknown_feature_names_are_listed(client, sample_features):
    payload = dict(sample_features)
    payload["Not A Feature"] = 1.0
    body = client.post("/predict", json={"features": payload}).json()
    assert body["unknown_features"] == ["Not A Feature"]


def test_payload_with_no_known_feature_is_rejected(client):
    res = client.post("/predict", json={"features": {"nope": 1.0, "also nope": 2.0}})
    assert res.status_code == 422
    assert "expected features" in res.json()["detail"].lower()


def test_threshold_outside_zero_one_is_rejected(client, sample_features):
    res = client.post("/predict", json={"features": sample_features, "threshold": 1.5})
    assert res.status_code == 422


def test_batch_scores_every_flow(client):
    samples = client.get("/samples").json()["samples"]
    flows = [s["features"] for s in samples]
    body = client.post("/predict/batch", json={"flows": flows}).json()
    assert body["count"] == len(flows)
    assert body["attacks_detected"] == sum(1 for s in samples if s["true_label"] != "BENIGN")
    predicted = [p["attack_type"] for p in body["predictions"]]
    assert predicted == [s["true_label"] for s in samples]


def test_oversized_batch_is_rejected(client, sample_features):
    flows = [sample_features] * (config.MAX_BATCH_SIZE + 1)
    res = client.post("/predict/batch", json={"flows": flows})
    assert res.status_code == 413


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"flows": []}).status_code == 422


def test_drift_warms_up_then_reports_psi(client, sample_features):
    """Drift must stay silent until it has enough rows, then produce a PSI."""
    flows = [sample_features] * config.DRIFT_MIN_ROWS
    client.post("/predict/batch", json={"flows": flows})
    body = client.get("/drift").json()
    assert body["status"] == "ok"
    assert body["rows_observed"] >= config.DRIFT_MIN_ROWS
    # Identical rows collapse into one bin, which is maximal drift.
    assert body["max_psi"] > config.DRIFT_ALERT_PSI
    assert body["alert"] is True


def test_stats_counts_requests_and_flows(client):
    body = client.get("/stats").json()
    assert body["requests_total"] > 0
    assert body["flows_scored_total"] > 0
    assert body["predictions_total"]


def test_metrics_endpoint_is_prometheus_text(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    body = res.text
    assert "# TYPE nids_requests_total counter" in body
    assert "nids_flows_scored_total" in body
    assert 'nids_build_info{model_version="testsha-2026-01-01"} 1' in body


def test_requests_carry_a_latency_header(client):
    res = client.get("/health")
    assert float(res.headers["X-Process-Time-Ms"]) >= 0


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]
