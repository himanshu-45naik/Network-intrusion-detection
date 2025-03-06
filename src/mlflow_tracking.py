import os
import mlflow
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Union
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
        self,
        model: Union[Pipeline, Any],
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tracking_uri: str,
        experiment_name: str,
    ) -> str:
        """Performs tracking of model based on the model metrics

        Args:
            model (Pipeline): The model which has to be tracked.
            model_name (str): The name of the model to be tracked.
            X_test (pd.DataFrame): The test data with which the model performance is judged.
            y_test (pd.Series): The test data with which the model performance is judged.
            tracking_uri (str): The tracking uri of mlflow ui.
            experiment_name (str): The name of the experiment.

        Returns:
            str: The runid of the model is returned.
        """
        pass


class ModelTracking(MLFlowTracking):
    def _safe_param_convert(self, param_value: Any) -> str:
        """
        Convert parameter values to a safe string representation
        
        Args:
            param_value (Any): Input parameter value
        
        Returns:
            str: Converted parameter value
        """
        try:
            # Handle common non-serializable types
            if param_value is None:
                return "None"
            if isinstance(param_value, (int, float, str, bool)):
                return str(param_value)
            return str(type(param_value).__name__)
        except Exception as e:
            logger.warning(f"Could not convert parameter: {e}")
            return "UNCONVERTIBLE_PARAM"

    def mlflow_log_model(
        self,
        model: Union[Pipeline, Any],
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tracking_uri: str,
        experiment_name: str,
    ) -> str:
        """
        Log model, parameters, metrics and artifacts to MLflow.
        """
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{model_name}_run") as run:
                run_id = run.info.run_id
                logger.info(f"Started MLflow run: {run_id}")

                # Log pipeline parameters (flattening nested parameters)
                pipeline_params = {}

                if isinstance(model, Pipeline):
                    # Handle Pipeline case
                    for step_name, step_obj in model.named_steps.items():
                        try:
                            if hasattr(step_obj, "get_params"):
                                step_params = step_obj.get_params()
                                for param_name, param_value in step_params.items():
                                    # Create fully qualified parameter name
                                    full_param_name = f"{step_name}__{param_name}"
                                    # Convert parameter to safe string
                                    pipeline_params[full_param_name] = self._safe_param_convert(param_value)
                        except Exception as step_err:
                            logger.warning(f"Error processing step {step_name}: {step_err}")
                else:
                    # Handle single model case
                    logger.warning(f"Model {model_name} is not a Pipeline. Logging only model parameters.")
                    if hasattr(model, "get_params"):
                        pipeline_params = {
                            k: self._safe_param_convert(v) 
                            for k, v in model.get_params().items()
                        }

                # Safely log parameters
                try:
                    mlflow.log_params(pipeline_params)
                except Exception as param_log_err:
                    logger.error(f"Failed to log parameters: {param_log_err}")

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

                # Confusion Matrix logging
                try:
                    plt.figure(figsize=(10, 8))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.title(f"Confusion Matrix - {model_name}")
                    plt.ylabel("True Label")
                    plt.xlabel("Predicted Label")

                    cm_path = "confusion_matrix.png"
                    plt.savefig(cm_path)
                    mlflow.log_artifact(cm_path)
                    plt.close()

                    # Clean up
                    if os.path.exists(cm_path):
                        os.remove(cm_path)
                except Exception as cm_err:
                    logger.error(f"Error logging confusion matrix: {cm_err}")

                # ROC Curve for binary classification
                try:
                    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                        plt.figure(figsize=(10, 8))
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)

                        plt.plot(
                            fpr, tpr,
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

                        roc_path = "roc_curve.png"
                        plt.savefig(roc_path)
                        mlflow.log_artifact(roc_path)
                        plt.close()

                        # Clean up
                        if os.path.exists(roc_path):
                            os.remove(roc_path)

                        # Log ROC AUC
                        mlflow.log_metric("roc_auc", roc_auc)
                except Exception as roc_err:
                    logger.error(f"Error logging ROC curve: {roc_err}")

                # Log additional metadata
                try:
                    mlflow.log_param("input_features", str(list(X_test.columns)))
                    mlflow.log_param(
                        "pipeline_steps", 
                        str([step_name for step_name, _ in model.steps]) 
                        if isinstance(model, Pipeline) else "Single Model"
                    )

                    # Classification report
                    report = classification_report(y_test, y_pred)
                    report_path = "classification_report.txt"
                    with open(report_path, "w") as f:
                        f.write(report)
                    mlflow.log_artifact(report_path)

                    # Clean up report file
                    if os.path.exists(report_path):
                        os.remove(report_path)

                    # Sample data
                    sample_data = X_test.head(5).to_csv(index=False)
                    sample_path = "sample_data.csv"
                    with open(sample_path, "w") as f:
                        f.write(sample_data)
                    mlflow.log_artifact(sample_path)

                    # Clean up sample data file
                    if os.path.exists(sample_path):
                        os.remove(sample_path)

                except Exception as metadata_err:
                    logger.error(f"Error logging metadata: {metadata_err}")

                logger.info(
                    f"Successfully logged pipeline model, metrics, and artifacts to MLflow run: {run_id}"
                )

                return run_id

        except Exception as main_err:
            logger.error(f"Critical error in MLflow logging: {main_err}")
            raise


class ModelTracker:
    @staticmethod
    def model_tracker(
        model: Union[Pipeline, Any],
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tracking_uri: str,
        experiment_name: str,
    ):
        tracker = ModelTracking()
        return tracker.mlflow_log_model(
            model, model_name, X_test, y_test, tracking_uri,
            experiment_name)


if __name__ == "__main__":
    pass