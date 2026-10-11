# Inference image for the NIDS service. Training is not done in here: the model
# bundle is built by scripts/train_serving_model.py and copied in, so the image
# needs neither the 235 MB dataset nor the ZenML/MLflow stack.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=artifacts/nids_model.joblib

WORKDIR /app

# Dependencies first so a code change does not invalidate the layer.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY serving ./serving
COPY artifacts/nids_model.joblib artifacts/samples.json artifacts/model_card.json ./artifacts/

# Run unprivileged: nothing in the service needs root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Reads $PORT rather than hardcoding 8000: hosts like Render inject their own
# port, and a healthcheck pinned to the wrong one reports a healthy service as
# unhealthy.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/ready\").status==200 else 1)"

# Single worker by default: the Random Forest holds ~100 MB resident and each
# worker loads its own copy. Scale with replicas, not workers.
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
