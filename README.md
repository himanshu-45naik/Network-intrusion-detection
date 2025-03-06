# Network Intrusion Detection System (NIDS)

A machine learning pipeline for detecting and classifying network intrusions using ZenML pipelines and MLflow tracking.

## Overview

This repository contains a complete ML pipeline for network intrusion detection. The system processes network traffic data, applies various data transformations, and trains machine learning models to detect different types of network attacks. The pipeline is built using the ZenML framework and includes MLflow integration for experiment tracking and model management.


## Installation

1. **Clone the repository**

   ```
   git clone https://github.com/yourusername/nids-ml-pipeline.git
   cd nids-ml-pipeline
   ```

2. **Create and activate a virtual environment**

   ```
   python -m venv venv
   venv\Scripts\activate
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
   zenml init
   ```

## Project Structure

- **`run_pipeline.py`**: Entry point to run the ML pipeline
- **`src/`**: Core functionality
  - Data ingestion and handling
  - Feature engineering and extraction
  - Data preprocessing and transformation
- **`steps/`**: ZenML pipeline steps
- **`models/`**: Model implementations (Logistic Regression, SVM, Random Forest, XGBoost, etc.)
- **`pipelines/`**: Definition of ZenML pipelines

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
