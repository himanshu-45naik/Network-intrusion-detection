"""FastAPI inference service for the network intrusion detector.

Serves the bundle written by `scripts/train_serving_model.py`: one multiclass
Random Forest that answers both "is this flow an attack" and "which family".

Endpoints
    GET  /                 Browser demo UI
    GET  /health           Liveness plus loaded-model version
    GET  /ready            Readiness: 503 until the model is loaded
    GET  /model            Served model card (metrics, features, params)
    POST /predict          Score one flow
    POST /predict/batch    Score up to MAX_BATCH_SIZE flows
    GET  /drift            Feature drift (PSI) over the live request window
    GET  /stats            JSON counters
    GET  /metrics          Prometheus exposition

Run locally:
    uvicorn serving.app:app --reload --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles

from serving import config
from serving.bundle import ModelBundle, UnknownFeaturesError, load_samples, optional_load
from serving.monitoring import Monitor
from serving.schemas import (
    BatchRequest,
    BatchResponse,
    DriftResponse,
    FlowRequest,
    HealthResponse,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = "serving/static"

state: Dict[str, Any] = {"bundle": None, "monitor": None, "samples": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model once at startup rather than per request."""
    bundle = optional_load(config.MODEL_PATH)
    state["bundle"] = bundle
    state["samples"] = load_samples(config.SAMPLES_PATH)
    state["monitor"] = Monitor(
        reference_stats=bundle.reference_stats if bundle else {},
        window=config.DRIFT_WINDOW,
        min_rows=config.DRIFT_MIN_ROWS,
    )
    if bundle:
        logger.info(f"Serving model {bundle.version} with {len(bundle.feature_names)} features")
    else:
        logger.error("Started without a model: /predict will return 503")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Network Intrusion Detection API",
    description=(
        "Random Forest intrusion detector trained on CICIDS2017. Classifies a network "
        "flow as benign or as an attack family, and exposes drift and latency metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def require_bundle() -> ModelBundle:
    """Returns the loaded bundle or fails the request with 503."""
    bundle = state["bundle"]
    if bundle is None:
        state["monitor"].record_error("model_not_loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return bundle


def monitor() -> Monitor:
    return state["monitor"]


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    """Times every request and files it under the matched route path."""
    started = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - started
    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)
    monitor().record_request(endpoint, response.status_code, duration)
    response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.2f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe. Returns 200 even without a model, so the reason is visible."""
    bundle = state["bundle"]
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
        uptime_seconds=round(monitor().uptime_seconds, 1),
    )


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe. 503 until the model is loaded, so no traffic arrives early."""
    if state["bundle"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": state["bundle"].version}


@app.get("/model", tags=["model"])
def model_card():
    """Returns what is actually deployed: metrics, feature list, training params."""
    return require_bundle().card()


@app.get("/samples", tags=["model"])
def samples():
    """Held-out flows, one per class, used by the demo UI."""
    return {"count": len(state["samples"]), "samples": state["samples"]}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: FlowRequest) -> PredictResponse:
    """Scores a single network flow.

    Unknown keys are ignored and missing features are imputed, so both lists are
    returned in the response: a client silently sending the wrong column names
    would otherwise get a confident prediction from an all-imputed row.
    """
    bundle = require_bundle()
    threshold = config.DEFAULT_THRESHOLD if payload.threshold is None else payload.threshold

    started = time.perf_counter()
    try:
        frame = bundle.to_frame([payload.features])
    except UnknownFeaturesError as err:
        monitor().record_error("unknown_features")
        raise HTTPException(status_code=422, detail=str(err)) from err

    predictions = bundle.predict(frame, threshold)
    latency = (time.perf_counter() - started) * 1000

    monitor().record_predictions(predictions)
    monitor().observe_features(frame)

    sent = {str(k).strip() for k in payload.features}
    expected = set(bundle.feature_names)
    return PredictResponse(
        **predictions[0],
        model_version=bundle.version,
        threshold=threshold,
        latency_ms=round(latency, 2),
        unknown_features=sorted(sent - expected),
        missing_features=sorted(expected - sent),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(payload: BatchRequest) -> BatchResponse:
    """Scores many flows in one vectorised pass.

    Batching matters here: per-call overhead dominates a Random Forest on a
    single 70-feature row, so one request of 500 flows is far cheaper than 500
    requests of one.
    """
    bundle = require_bundle()
    if len(payload.flows) > config.MAX_BATCH_SIZE:
        monitor().record_error("batch_too_large")
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(payload.flows)} exceeds limit {config.MAX_BATCH_SIZE}",
        )
    threshold = config.DEFAULT_THRESHOLD if payload.threshold is None else payload.threshold

    started = time.perf_counter()
    try:
        frame = bundle.to_frame(payload.flows)
    except UnknownFeaturesError as err:
        monitor().record_error("unknown_features")
        raise HTTPException(status_code=422, detail=str(err)) from err

    predictions = bundle.predict(frame, threshold)
    latency = (time.perf_counter() - started) * 1000

    monitor().record_predictions(predictions)
    monitor().observe_features(frame)

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        attacks_detected=sum(1 for p in predictions if p["verdict"] == "attack"),
        model_version=bundle.version,
        threshold=threshold,
        latency_ms=round(latency, 2),
    )


@app.get("/drift", response_model=DriftResponse, tags=["ops"])
def drift() -> DriftResponse:
    """Reports feature drift of live traffic against the training distribution."""
    report = monitor().drift_report()
    max_psi = report.get("max_psi")
    return DriftResponse(
        status=report["status"],
        rows_observed=report["rows_observed"],
        rows_required=report.get("rows_required"),
        max_psi=max_psi,
        alert=bool(max_psi is not None and max_psi >= config.DRIFT_ALERT_PSI),
        alert_threshold=config.DRIFT_ALERT_PSI,
        features=report["features"],
    )


@app.get("/stats", tags=["ops"])
def stats():
    """JSON counters, for a quick look without a Prometheus scrape."""
    bundle = state["bundle"]
    return monitor().snapshot(bundle.version if bundle else None)


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> PlainTextResponse:
    """Prometheus exposition of request, prediction and drift metrics."""
    bundle = state["bundle"]
    return PlainTextResponse(
        monitor().render_prometheus(bundle.version if bundle else None),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/", include_in_schema=False)
def index():
    """Serves the demo UI."""
    return FileResponse(f"{STATIC_DIR}/index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
