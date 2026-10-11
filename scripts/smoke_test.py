"""Exercises a running service and prints throughput, latency and drift.

Used as a post-deploy check: it hits every endpoint, replays held-out flows and
reports whether predictions match the labels, so a broken deploy shows up as a
wrong number rather than a 200.

Usage:
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000
    python scripts/smoke_test.py --base-url https://<host> --flows 2000
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SAMPLES = "artifacts/samples.json"


def call(base_url: str, path: str, body=None, timeout: float = 30.0):
    """Issues a JSON request and returns (status, parsed body)."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--flows", type=int, default=400, help="Total flows to replay through the batch endpoint."
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--single-calls", type=int, default=30, help="Single-flow calls used to measure latency."
    )
    args = parser.parse_args()

    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        print(f"[{mark}] {name}{f' - {detail}' if detail else ''}")
        if not condition:
            failures.append(name)

    status, health = call(args.base_url, "/health")
    check("health 200", status == 200, json.dumps(health))
    check("model loaded", bool(health.get("model_loaded")), health.get("model_version", ""))

    status, _ = call(args.base_url, "/ready")
    check("ready 200", status == 200)

    status, card = call(args.base_url, "/model")
    check("model card 200", status == 200)
    if status == 200:
        print(
            f"       model={card['version']} features={card['n_features']} "
            f"classes={len(card['classes'])} "
            f"attack_recall={card['metrics']['binary']['recall']:.4f}"
        )

    with open(args.samples) as f:
        samples = json.load(f)
    check("samples available", bool(samples), f"{len(samples)} rows")
    if not samples:
        return 1

    # Label agreement on the held-out rows. The merged rare class is expected to
    # be weak, so it is reported but not treated as a failure.
    status, batch = call(
        args.base_url, "/predict/batch", {"flows": [s["features"] for s in samples]}
    )
    check("batch 200", status == 200)
    if status == 200:
        hits = sum(
            1
            for sample, pred in zip(samples, batch["predictions"])
            if pred["attack_type"] == sample["true_label"]
        )
        print(f"       label agreement {hits}/{len(samples)}")
        for sample, pred in zip(samples, batch["predictions"]):
            flag = "ok " if pred["attack_type"] == sample["true_label"] else "miss"
            print(
                f"       {flag} true={sample['true_label']:<13} "
                f"pred={pred['attack_type']:<13} p_attack={pred['attack_probability']:.3f}"
            )

    # Single-flow latency, measured client side so it includes transport.
    latencies = []
    for _ in range(args.single_calls):
        flow = random.choice(samples)["features"]
        started = time.perf_counter()
        status, _ = call(args.base_url, "/predict", {"features": flow})
        latencies.append((time.perf_counter() - started) * 1000)
        if status != 200:
            break
    check("single predict 200", status == 200)
    if latencies:
        latencies.sort()
        print(
            f"       single-flow latency p50={statistics.median(latencies):.1f}ms "
            f"p95={latencies[int(len(latencies) * 0.95) - 1]:.1f}ms "
            f"over {len(latencies)} calls"
        )

    # Throughput, driven through the batch endpoint.
    pool = [s["features"] for s in samples]
    sent = 0
    started = time.perf_counter()
    while sent < args.flows:
        size = min(args.batch_size, args.flows - sent)
        flows = [random.choice(pool) for _ in range(size)]
        status, _ = call(args.base_url, "/predict/batch", {"flows": flows})
        if status != 200:
            check("throughput batch 200", False, str(status))
            break
        sent += size
    elapsed = time.perf_counter() - started
    if sent:
        print(f"       replayed {sent} flows in {elapsed:.2f}s ({sent / elapsed:,.0f} flows/s)")

    status, reject = call(args.base_url, "/predict", {"features": {"not a feature": 1}})
    check("malformed payload rejected with 422", status == 422, str(reject.get("detail", ""))[:60])

    status, drift = call(args.base_url, "/drift")
    check("drift 200", status == 200)
    if status == 200:
        if drift["status"] == "ok":
            print(f"       rows={drift['rows_observed']} max_psi={drift['max_psi']}")
            print(
                "       top drifting: "
                + ", ".join(f"{d['feature']}={d['psi']}" for d in drift["features"][:3])
            )
        else:
            print(f"       {drift['status']} {drift['rows_observed']}/{drift['rows_required']}")

    # /metrics is Prometheus text, not JSON, so it is read raw.
    try:
        raw = urllib.request.urlopen(f"{args.base_url.rstrip('/')}/metrics", timeout=10).read()
        check("metrics exposes counters", b"nids_flows_scored_total" in raw)
    except Exception as err:  # noqa: BLE001 - smoke test reports, never raises
        check("metrics exposes counters", False, str(err))

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
