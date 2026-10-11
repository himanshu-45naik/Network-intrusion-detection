# Deployment

How this project runs once it is deployed: what the live system is, what each
piece does, the numbers it actually produces, and the procedures to deploy,
verify, roll back and retrain it.

The repository holds two separate things. `pipelines/` is the **experimentation**
side — a ZenML pipeline that trains eight model/target combinations and logs
them to MLflow so they can be compared. `serving/` is the **production** side —
one promoted model behind a FastAPI service. Everything below is about the
second.

---

## 1. What is deployed

```
                       ┌──────────────────────────────────────┐
  analyst / client     │        Inference container           │
  ─────────────────▶   │                                      │
   HTTPS               │  uvicorn ──▶ FastAPI (serving/app.py)│
                       │                 │                    │
                       │                 ├─ ModelBundle       │
                       │                 │  • sklearn Pipeline │
                       │                 │    (median imputer  │
                       │                 │     → RandomForest) │
                       │                 │  • 70 feature names │
                       │                 │  • 8 class labels   │
                       │                 │  • held-out metrics │
                       │                 │  • training deciles │
                       │                 │                    │
                       │                 └─ Monitor            │
                       │                    • request counters │
                       │                    • latency histogram│
                       │                    • rolling PSI      │
                       │                                       │
                       │  loaded once at startup from          │
                       │  artifacts/nids_model.joblib (4.2 MB) │
                       └───────────────┬───────────────────────┘
                                       │ GET /metrics
                                       ▼
                               Prometheus  ──▶  alerting
```

**One container, one process, no external dependency at request time.** No
database, no cache, no model server, no network call. The model is a 4.2 MB
joblib bundle baked into the image and loaded once in the FastAPI lifespan
handler, so a request is CPU work and nothing else. That is why the service can
run on a free-tier box and still answer in single-digit milliseconds.

The deliberate consequence of that design: the container is **stateless and
disposable**. Scaling means more replicas, rollback means deploying a previous
image tag, and there is nothing to migrate.

### Why the model bundle is a single file

The bundle carries everything the service needs: the fitted pipeline, the
expected feature names and order, the class labels, the held-out metrics, and
the per-feature decile edges used for drift. Because metadata travels with the
estimator, the runtime image needs neither MLflow nor ZenML — `requirements-serving.txt`
is 7 packages against 15 for training, which is both a smaller image and a
smaller attack surface on the internet-facing component.

The bundle is committed to the repository. For a project of this size that
makes a deploy reproducible from a single commit sha, at the cost of 4.2 MB of
git history. At a size where that stops being true, the same bundle would be
pushed to an artifact registry or the MLflow model registry and pulled by digest
at image build time; nothing in `serving/` would change, only `MODEL_PATH`.

---

## 2. The served model

| | |
|---|---|
| Algorithm | `RandomForestClassifier`, 120 trees, `max_features="sqrt"`, `class_weight="balanced_subsample"` |
| Pipeline | `SimpleImputer(median)` → `RandomForestClassifier` |
| Dataset | CICIDS2017 (`MachineLearningCVE`), 2,830,743 raw flows |
| After de-duplication | ~2.52 M flows |
| Trained on | 10% stratified sample = 252,232 flows → 189,174 train / 63,058 test |
| Features | 70 (8 constant columns dropped) |
| Classes | BENIGN, Bot, Brute Force, DDoS, DoS, Port Scan, Web Attack, Other Attack |
| Artifact | `artifacts/nids_model.joblib`, 4.2 MB |
| Training time | ~32 s end to end on a laptop |

### Held-out performance

Attack vs benign — the number a SOC actually operates on:

| Metric | Value |
|---|---|
| Recall (attacks caught) | **0.9919** |
| Precision | **0.9966** |
| F1 | 0.9943 |
| ROC AUC | 0.9998 |
| False positive rate | **0.00069** |

Attack family (multiclass):

| Metric | Value |
|---|---|
| Accuracy | 0.9980 |
| Weighted F1 | 0.9980 |
| Macro F1 | **0.8313** |

Per class F1: BENIGN 0.999, DDoS 0.999, DoS 0.996, Port Scan 0.989, Brute Force
0.987, Web Attack 0.940, Bot 0.741, **Other Attack 0.000**.

The gap between 0.998 weighted and 0.831 macro F1 is one class. `Other Attack`
is Infiltration and Heartbleed merged — together about 47 flows in the full
dataset, a handful of rows after a 10% sample, and the model never predicts it.
Macro F1 is reported rather than hidden because it is the honest signal: this
model detects the attack families that have volume and cannot detect the two
that do not. Fixing that is a data problem (targeted oversampling or keeping
those classes at 100% sample rate), not a model problem.

The false positive rate matters more than the headline accuracy. At 83% benign
traffic and 0.00069 FPR, a million flows a day produces roughly 570 false
alerts — the difference between an alert stream someone reads and one they mute.

### The decision threshold

`/predict` returns `verdict = "attack"` when `1 - P(BENIGN) >= threshold`, with
`threshold` configurable per request and defaulted by `DEFAULT_THRESHOLD`. The
multiclass head gives the family; the binary verdict is derived from it, so one
model answers both questions and the two answers can never contradict each
other.

Exposing the threshold at request time rather than baking it in is the point:
moving it trades recall against precision with no retraining, which is the knob
an operator actually wants during an incident.

---

## 3. API contract

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Browser demo: pick a held-out flow, move the threshold, see the verdict |
| GET | `/docs` | OpenAPI / Swagger UI |
| GET | `/health` | Liveness. Returns 200 with `status: degraded` if the model failed to load |
| GET | `/ready` | Readiness. 503 until the model is loaded |
| GET | `/model` | Served model card: metrics, feature list, class list, training params |
| GET | `/samples` | Held-out flows, one per class, for the demo UI |
| POST | `/predict` | Score one flow |
| POST | `/predict/batch` | Score up to `MAX_BATCH_SIZE` flows in one vectorised pass |
| GET | `/drift` | Feature drift (PSI) of live traffic vs the training distribution |
| GET | `/stats` | JSON counters |
| GET | `/metrics` | Prometheus exposition |

### Request

```bash
curl -X POST https://<host>/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"Destination Port": 80, "Flow Duration": 1385514, ...},
       "threshold": 0.5}'
```

### Response

```json
{
  "verdict": "attack",
  "attack_type": "DDoS",
  "attack_probability": 1.0,
  "class_probabilities": {"BENIGN": 0.0, "DDoS": 1.0, "...": 0.0},
  "model_version": "200e687c-2026-10-11",
  "threshold": 0.5,
  "latency_ms": 20.73,
  "unknown_features": [],
  "missing_features": []
}
```

Features go in as a name/value map, not a fixed field list. CICFlowMeter emits
~78 columns and the model keeps 70, so a map lets a client send its full row
without tracking which columns were dropped during training.

`unknown_features` and `missing_features` exist because of a specific failure
mode. The payload is reindexed to the fitted column order and anything absent is
imputed with the training median — which means a client that sends the *wrong
column names* gets a perfectly confident prediction computed from an entirely
imputed row. Returning both lists makes that visible instead of silent. A
payload with no recognised feature at all is rejected with 422 rather than
scored.

Every response carries `model_version` (`<git sha>-<train date>`) and an
`X-Process-Time-Ms` header, so a prediction can be traced back to the commit
that produced the model.

---

## 4. Measured behaviour

From `scripts/smoke_test.py` against the container on a laptop, single uvicorn
worker:

| | |
|---|---|
| Single-flow latency | p50 **16.6 ms**, p95 16.8 ms (client-side, includes HTTP) |
| Batch throughput | **7,000–10,000 flows/s** (400-flow batches) |
| Batch of 320 flows | 19.7 ms server-side |
| Cold start | ~1 s to load the bundle and bind the port |
| Resident memory | ~250 MB |

The 600× gap between single-flow and batched throughput is per-call overhead,
not forest inference: a 70-feature DataFrame and one `predict_proba` call cost
about the same whether the frame has 1 row or 400. Anything scoring a live flow
feed should batch, which is why `/predict/batch` exists and why the Dockerfile
runs one worker and scales by replica — each worker would hold its own copy of
the forest.

---

## 5. Observability

### Metrics

`/metrics` serves the Prometheus text format with no client library — the
counters live in a thread-safe in-process `Monitor` and are rendered on scrape:

- `nids_build_info{model_version}` — which model this replica is serving
- `nids_requests_total{endpoint,status}` — traffic and error rate by route
- `nids_flows_scored_total` — flows, not requests, which is the real unit of work
- `nids_predictions_total{class}` — prediction mix, by class and by verdict
- `nids_request_latency_seconds` — histogram with buckets from 5 ms to 2.5 s
- `nids_errors_total{kind}` — `model_not_loaded`, `unknown_features`, `batch_too_large`
- `nids_feature_drift_psi{feature}` and `nids_feature_drift_psi_max`

The prediction mix is the cheapest early warning available. This model should
call ~83% of real traffic benign; a replica suddenly flagging 40% of flows as
attacks is either under attack or broken, and either way someone should look.

### Drift

Drift is a Population Stability Index against the decile edges captured from the
training set and stored in the bundle. Incoming values are binned against those
edges and only the bin counts are kept, so memory is flat no matter how much
traffic arrives, and the training data never has to ship with the model.

PSI is read conventionally: `< 0.1` stable, `0.1–0.2` moderate, `> 0.2` a
material shift. `/drift` reports the worst features and sets `alert: true` past
`DRIFT_ALERT_PSI`; below `DRIFT_MIN_ROWS` observed rows it returns
`status: "warming_up"` rather than a number computed from too little traffic.

This is feature drift, not accuracy. Production traffic has no labels, so the
service can detect that the input distribution has moved away from what the
model was trained on — which is the signal available without a labelling
pipeline, and is enough to decide to retrain.

### Logs

Structured on stdout, collected by the platform. No request payloads are
logged: a flow record is network telemetry about real hosts.

---

## 6. Deploying

All three paths build the same `Dockerfile`: Python 3.12 slim, dependencies in
their own layer, non-root `appuser` (uid 10001), `HEALTHCHECK` against `/ready`,
dataset and training stack excluded via `.dockerignore`.

### Local, with Prometheus

```bash
docker compose -f deploy/docker-compose.yml up --build
# API        http://localhost:8000
# Demo UI    http://localhost:8000
# Prometheus http://localhost:9090
```

### Hugging Face Spaces (free, no card)

A Space builds from its own git repo and needs YAML frontmatter in `README.md`,
so the deploy script assembles a minimal Space repo in a temp directory — only
`serving/`, the three artifact files, `requirements-serving.txt` and a
port-7860 Dockerfile. The dataset and training code never leave the repository.

The script creates the Space if it does not exist, then force-pushes the staged
repo, so the same command is both the first deploy and every redeploy.

```bash
export HF_TOKEN=hf_xxx           # write token from huggingface.co/settings/tokens
./deploy/push_to_space.sh <hf-username> nids-api
# → https://<hf-username>-nids-api.hf.space
```

The build takes a few minutes the first time, then the Space serves the demo UI
at `/`, OpenAPI at `/docs` and the full API. Free CPU Spaces do not sleep on a
per-request basis, so there is no cold start between demos.

### Render (free tier)

`render.yaml` is a blueprint: connect the repo in the Render dashboard and it
builds the Dockerfile, health-checks `/ready` and redeploys on push to `main`.
The free plan sleeps after 15 minutes idle, so the first request after idle pays
a cold start.

### Verifying a deploy

```bash
python scripts/smoke_test.py --base-url https://<host> --flows 2000
```

The smoke test hits every endpoint, replays the held-out flows and checks the
predictions against their labels, measures single-flow latency and batch
throughput, confirms a malformed payload is rejected with 422, and reads the
drift and metrics endpoints. It exits non-zero on any failure, so it works as a
post-deploy gate. A deploy that returns 200 on everything but predicts the wrong
class still fails it.

Expected output includes `label agreement 7/8` — the miss is `Other Attack`,
per §2.

---

## 7. CI/CD

`.github/workflows/ci.yml`, on every push and pull request:

1. **lint** — `ruff check` and `ruff format --check`
2. **test** — 24 pytest contract tests against the API
3. **image** — build the deployment image, boot the container, poll `/ready`,
   then run the smoke test against it

The test job installs only `requirements-serving.txt`: the API tests build their
own 5-tree synthetic model with the same bundle shape, so the service contract
is verified in seconds without the 235 MB dataset or the ZenML stack in CI.

The image job is the one that matters for deployment. It builds the exact
artifact that ships and exercises it over HTTP, so a broken Dockerfile, a
missing model bundle or a model that loads but predicts wrongly fails in CI
rather than on the deployed URL.

---

## 8. Configuration

All via environment variables, read in `serving/config.py`:

| Variable | Default | Effect |
|---|---|---|
| `MODEL_PATH` | `artifacts/nids_model.joblib` | Bundle to serve |
| `SAMPLES_PATH` | `artifacts/samples.json` | Demo rows |
| `DEFAULT_THRESHOLD` | `0.5` | Default decision threshold on P(attack) |
| `DRIFT_WINDOW` | `5000` | Rows in the rolling drift window |
| `DRIFT_MIN_ROWS` | `200` | Rows before PSI is reported |
| `DRIFT_ALERT_PSI` | `0.2` | PSI at which `/drift` alerts |
| `MAX_BATCH_SIZE` | `500` | Flows per batch request |
| `PORT` | `8000` | Listen port |

---

## 9. Runbook

**Service returns 503 on `/predict`.** The bundle failed to load. `/health`
returns 200 with `model_loaded: false` — this is deliberate, so the reason is
observable instead of the container crash-looping before anything can be
scraped. Check `MODEL_PATH` and the startup logs; `nids_errors_total{kind="model_not_loaded"}`
will be climbing.

**Rollback.** Deploy the previous image tag, or the previous commit on the
platforms that build from git. The service is stateless, so there is nothing
else to undo. Confirm with `/model` — `version` carries the commit sha.

**`/drift` alerting.** The live feature distribution has moved. Look at which
features are drifting: a few features with high PSI is usually a client or
feature-extractor change, broad drift is a genuine traffic shift. Retrain on
recent data and promote.

**Prediction mix looks wrong.** Compare `nids_predictions_total` against the
~83% benign baseline. A sharp move with flat drift suggests a real traffic
change; a move together with drift suggests the input pipeline changed.

**Latency climbing.** Check batch sizes — many single-flow calls cost ~100× more
per flow than batched ones. Scale by adding replicas, not uvicorn workers.

---

## 10. Retraining and promotion

```
CICIDS2017 / captured flows
        │
        ├──▶ ZenML pipeline (run_pipeline.py) ──▶ MLflow: 8 candidates compared
        │
        └──▶ scripts/train_serving_model.py ──▶ artifacts/nids_model.joblib
                                                        │
                                                   git commit
                                                        │
                                                 CI: lint → test → image+smoke
                                                        │
                                                     deploy
```

`scripts/train_serving_model.py` imports its transformations from `src/`, the
same modules the ZenML pipeline steps call, so the promoted model is built by
the same cleaning code that produced the comparison runs rather than a
re-implementation that could drift from it.

Promotion is a commit: the bundle, its model card and the metrics in it land
together, and `model_version` on every prediction points back to that commit.
MLflow logging is attempted and treated as non-fatal — a training run must not
fail because a tracking server is unreachable.

---

## 11. Honest limits

Worth being direct about, since these are the first things a reviewer should
ask:

- **No authentication.** The deployed demo is open. Anything real needs an API
  key or mTLS in front of `/predict`, and `/metrics` should not be public.
- **No rate limiting.** A single caller can saturate the one worker.
- **Drift is feature drift, not accuracy drift.** Production flows are
  unlabelled, so the service cannot measure its own recall after deploy. That
  needs analyst feedback flowing back as labels.
- **Metrics are per-replica and in-memory.** They reset on restart and do not
  aggregate across replicas. Correct for one container; a multi-replica
  deployment needs Prometheus to do the aggregation, which it does by design.
- **Trained on a 10% sample.** Chosen so the whole train-and-promote loop runs
  in ~30 seconds. The sample is stratified and class proportions match the full
  dataset to within 1%, but the two rare attack classes are the visible cost.
- **CICIDS2017 is from 2017.** A model trained on it detects 2017 attack
  patterns. Deployed on live traffic it would need retraining on captured flows;
  the drift endpoint is what would tell you how far off the training
  distribution that traffic is.
- **The feature contract is CICFlowMeter's.** The service scores flow records,
  not packets. Something upstream has to produce those 70 features.

---

## 12. Reproducing from scratch

```bash
pip install -r requirements.txt            # training stack
cp config/.env.example config/.env         # set DATA_PATH to the CICIDS2017 zip
python scripts/train_serving_model.py --sample-size 0.1

pip install -r requirements-serving.txt    # serving only
uvicorn serving.app:app --port 8000
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```
