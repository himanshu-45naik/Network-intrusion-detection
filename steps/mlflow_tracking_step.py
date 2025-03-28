from zenml import step
import pandas as pd
from sklearn.pipeline import Pipeline
from src.mlflow_tracking import  ModelTracker

@step
def mlflow_tracker(model: Pipeline, model_name: str, X_test:pd.DataFrame, y_test: pd.Series, tracking_uri: str, experiment_name: str) -> str:
    """Implements the mlflow tracking.

    Args:
        model (Pipeline): The model to be tracked.
        model_name (str): The name of the model to be tracked.
        X_test (pd.DataFrame): The test data on which the model's performance is tracked.
        y_test (pd.Series): The test data on which the model's performance is tracked.

    Returns:
        str: The runid is returned as string.
    """
    if model_name == "logisticregression":
        tracker = ModelTracker()
    elif model_name == "oc_svm":
        tracker = ModelTracker()
    elif model_name == "rf_binary":
        tracker = ModelTracker()
    elif model_name == "rf_multiclass":
        tracker = ModelTracker()
    elif model_name == "xgb_binary":
        tracker = ModelTracker()
    elif model_name == "xgb_multiclass":
        tracker = ModelTracker()
    else:
        raise ValueError("Undefined Model.")
    
    run_id = tracker.model_tracker(model, model_name, X_test, y_test, tracking_uri, experiment_name)
    
    return run_id    
    