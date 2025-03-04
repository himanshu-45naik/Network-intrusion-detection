from zenml import step
import pandas as pd
from sklearn.pipeline import Pipeline
from src.mlflow_tracking import  ModelTracker

@step
def mlflow_tracker(model: Pipeline, model_name: str, X_test:pd.DataFrame, y_test: pd.Series, tracking_uri, experiment_name) -> str:
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
    if model_name == "oc-svm":
        tracker = ModelTracker()
    if model_name == "rf_binary":
        tracker = ModelTracker()
    if model_name == "xgb_bianry":
        tracker = ModelTracker()
    if model_name == "xgb_multiclass":
        tracker = ModelTracker()
    
    tracker.model_tracker(model, model_name, X_test, y_test, tracking_uri, experiment_name)
        
    