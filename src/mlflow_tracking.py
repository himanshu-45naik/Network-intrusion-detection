import os
import mlflow
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLFlowTracking(ABC):
    @abstractmethod
    def mlflow_log_model(
        model: Pipeline,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tracking_uri: str,
        experiment_name: str,
    ) -> str:
        """Performs tracking of model based on the model metrics

        Args:
            model (Pipeline): The model which has to be tracked.
            model_name (str): The name of th model to be tracked.
            X_test (pd.DataFrame): The test data with which the model performance is judged.
            y_test (pd.Series): The test data with which the model performance is judged.
            tracking_uri (str): The tracking uri of mlflow ui.
            experiment_name (str): The name of the experiment.

        Returns:
            str: The runid of the model is returned.
        """
        pass


class ModelTracking(MLFlowTracking):
    def mlflow_log_model(
        model: Pipeline,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "intrusion_detection",
    ) -> str:
        """
        Log model, parameters, metrics and artifacts to MLflow.

        Args:
            model: Trained sklearn Pipeline to log
            model_name: Name of the model (e.g., "pipeline_randomforest")
            X_test: Test features used for model evaluation
            y_test: Test target values
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name

        Returns:
            MLflow run ID
        """
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")

            # Log pipeline parameters (flattening nested parameters)
            pipeline_params = {}
            for step_name, step_obj in model.named_steps.items():
                if hasattr(step_obj, "get_params"):
                    step_params = step_obj.get_params()
                    # Prefix parameters with step name for clarity
                    for param_name, param_value in step_params.items():
                        pipeline_params[f"{step_name}__{param_name}"] = param_value

            mlflow.log_params(pipeline_params)

            # Calculate and log metrics
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }
            mlflow.log_metrics(metrics)

            # Log the entire pipeline as a model
            mlflow.sklearn.log_model(model, model_name)

            # Log confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Save and log confusion matrix
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            # Clean up local file
            if os.path.exists(cm_path):
                os.remove(cm_path)

            # For binary classification, log ROC curve
            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                plt.figure(figsize=(10, 8))
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                plt.plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f"ROC curve (area = {roc_auc:.2f})",
                )
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {model_name}")
                plt.legend(loc="lower right")

                # Save and log ROC curve
                roc_path = "roc_curve.png"
                plt.savefig(roc_path)
                mlflow.log_artifact(roc_path)
                plt.close()

                # Clean up local file
                if os.path.exists(roc_path):
                    os.remove(roc_path)

                # Log ROC AUC as a metric
                mlflow.log_metric("roc_auc", roc_auc)

            # Log feature list - use original feature names
            mlflow.log_param("input_features", str(list(X_test.columns)))

            # Log pipeline steps
            mlflow.log_param(
                "pipeline_steps", str([step_name for step_name, _ in model.steps])
            )

            # Log classification report as an artifact
            report = classification_report(y_test, y_pred)
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            # Clean up local file
            if os.path.exists(report_path):
                os.remove(report_path)

            # Log input data sample as an artifact
            sample_data = X_test.head(5).to_csv(index=False)
            sample_path = "sample_data.csv"
            with open(sample_path, "w") as f:
                f.write(sample_data)
            mlflow.log_artifact(sample_path)

            # Clean up local file
            if os.path.exists(sample_path):
                os.remove(sample_path)

            logger.info(
                f"Successfully logged pipeline model, metrics, and artifacts to MLflow run: {run_id}"
            )

            return run_id


class ModelTracker:
    @staticmethod
    def model_tracker(
        self,
        model: Pipeline,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tracking_uri: str,
        experiment_name: str,
    ):
        tracker = ModelTracking()
        tracker.mlflow_log_model(
            model, model_name, X_test, y_test, tracking_uri = " http://127.0.0.1:5000",
            experiment_name = "intrusion_detection")


if __name__ == "__main__":
    pass
