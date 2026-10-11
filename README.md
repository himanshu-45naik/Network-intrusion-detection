# Network Intrusion Detection System (NIDS)

A machine learning pipeline for detecting and classifying network intrusions using ZenML pipelines and MLflow tracking.

## Overview

This repository contains a complete ML pipeline for network intrusion detection. The system processes network traffic data, applies various data transformations, and trains machine learning models to detect different types of network attacks. The pipeline is built using the ZenML framework and includes MLflow integration for experiment tracking and model management.

It has two sides:

- **Experimentation** (`pipelines/`, `steps/`, `src/`, `models/`): a ZenML pipeline that trains eight model/target combinations and logs them to MLflow for comparison.
- **Production** (`serving/`, `scripts/`, `Dockerfile`): one promoted model behind a FastAPI service, with a browser demo, Prometheus metrics and feature-drift detection.

## Deployed service

**Live: https://nids-api-bibt.onrender.com** — [demo UI](https://nids-api-bibt.onrender.com/) ·
[OpenAPI](https://nids-api-bibt.onrender.com/docs) ·
[model card](https://nids-api-bibt.onrender.com/model) ·
[metrics](https://nids-api-bibt.onrender.com/metrics)

Free instance, so it sleeps after 15 minutes idle and the first request after that
takes ~1 minute while the container boots.

One multiclass XGBoost model trained on CICIDS2017 is promoted to a self-contained
0.6 MB bundle and served over FastAPI. It answers both "is this flow an attack"
and "which attack family" from a single model.

Held-out performance, attack vs benign: **recall 0.9977, precision 0.9960, false
positive rate 0.00082**. Multiclass: accuracy 0.9988, weighted F1 0.9988, macro
F1 0.8416 (one rare merged class is undetectable by every candidate — see
[the comparison notebook](analysis/model_comparison.ipynb)).

Chosen over Random Forest, LightGBM, Logistic Regression and One-Class SVM in
[`analysis/model_comparison.ipynb`](analysis/model_comparison.ipynb), which trains all
five on one split and diagnoses why LightGBM needs `class_weight` to be competitive.

```bash
# train the promoted model (--model also accepts rf-tuned, lgbm)
python scripts/train_serving_model.py --model xgb --sample-size 0.1

# serve it
uvicorn serving.app:app --port 8000      # demo UI at /, OpenAPI at /docs

# or the container, with Prometheus alongside
docker compose -f deploy/docker-compose.yml up --build

# verify a running instance end to end
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Browser demo |
| GET | `/health`, `/ready` | Liveness and readiness probes |
| GET | `/model` | Served model card: metrics, features, training params |
| POST | `/predict` | Score one flow |
| POST | `/predict/batch` | Score up to 500 flows in one pass |
| GET | `/drift` | Feature drift (PSI) vs the training distribution |
| GET | `/metrics` | Prometheus exposition |

**[DEPLOYMENT.md](DEPLOYMENT.md)** covers the deployed architecture, the measured
latency and throughput, the observability and drift setup, the deploy paths
(Hugging Face Spaces, Render, Docker Compose), the CI/CD pipeline, a runbook and
the known limits.

## Installation

1. **Clone the repository**

   ```
   git clone https://github.com/himanshu-45naik/Network-intrusion-detection.git
   cd Network-intrusion-detection
   ```

2. **Create and activate a virtual environment**

   ```
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install the required packages**

   ```
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with certain packages on Windows, try installing them one by one:
   
   ```
   pip install zenml mlflow scikit-learn pandas numpy imblearn matplotlib seaborn click
   ```

4. **Set up ZenML**

   ```
   zenml login --"local"
   ```

5. **Configure environment**

   Copy the example config and adjust if needed:

   ```
   cp config/.env.example config/.env
   ```

   Place the CICIDS2017 zip (MachineLearningCSV.zip) at the path set in `DATA_PATH`
   (default: `data/MachineLearningCSV.zip`). The `data/` directory and `config/.env`
   are gitignored.

## Project Structure

- **`run_pipeline.py`**: Entry point to run the ML pipeline
- **`src/`**: Core functionality
  - Data ingestion and handling
  - Feature engineering and extraction
  - Data preprocessing and transformation
- **`steps/`**: ZenML pipeline steps
- **`models/`**: Model implementations (Logistic Regression, SVM, Random Forest, XGBoost, etc.)
- **`pipelines/`**: Definition of ZenML pipelines
- **`scripts/`**: `train_serving_model.py` (promote one model to a serving bundle) and `smoke_test.py` (post-deploy verification)
- **`serving/`**: FastAPI inference service, request metrics and drift scoring
- **`artifacts/`**: The promoted model bundle, its model card and demo samples
- **`deploy/`**: Docker Compose stack with Prometheus, and the Hugging Face Spaces deploy script
- **`tests/`**: Contract tests for the inference API

## Running the Pipeline

1. **Start MLflow server** (for tracking experiments)

   Open a new command prompt and run:
   ```
   mlflow ui --host 127.0.0.1 --port 5000
   ```

2. **Execute the pipeline**

   In your original terminal with the activated environment:
   ```
   python run_pipeline.py
   ```

## Configuring the Pipeline

The pipeline can be configured in multiple ways:

1. **Data Handling Configuration**
   - Missing value strategies: mean, median, mode, constant
   - Handling of infinite values
   - Feature name sanitization

2. **Feature Engineering Options**
   - Standard scaling
   - Min-max scaling
   - Log transformation
   - One-hot encoding
   - Handling of one-value features
   - Binary or multiclass encoding

3. **Feature Extraction with PCA**
   - Dimensionality reduction

4. **Duplicate Handling**
   - Drop duplicate values

5. **Model Selection**
   - Logistic Regression
   - One-Class SVM
   - Random Forest (binary/multiclass)
   - XGBoost (binary/multiclass)
   - LightGBM (binary/multiclass)

## Working with Imbalanced Data

The pipeline includes SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced data:

```python
# To apply SMOTE:
from steps.oversampling_data_step import sampling_data

x_resampled, y_resampled = sampling_data(x_train, y_train)
```

## Tracking Experiments with MLflow

MLflow is used to track experiments, metrics, and models. You can access the MLflow UI at http://127.0.0.1:5000 after starting the server.

The following metrics are tracked:
- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC (for binary classification)
- Confusion matrix
- Classification report


## Data Requirements

The pipeline expects a zip file containing CSV data with network traffic features and labeled attacks. The attack types include:

- BENIGN (normal traffic)
- DDoS (Distributed Denial of Service)
- DoS (various types: Hulk, GoldenEye, Slowloris, Slowhttptest)
- Port Scan
- Brute Force (FTP-Patator, SSH-Patator)
- Web Attacks (Brute Force, XSS, SQL Injection)
- Bot
- Infiltration
- Heartbleed

## Contact
- email: himanshucnaik45@gmail.com
