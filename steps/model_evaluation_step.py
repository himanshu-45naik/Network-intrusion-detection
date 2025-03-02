from src.model_evaluation import ModelEvaluator, LogisticRegressionEvaluation
from zenml import step
import pandas as pd

@step
def model_evaluation(X_test: pd.DataFrame, y_test: pd.Series, model, model_name: str):
    """Performs evaluation of a particular model.

    Args:
        X_test (pd.DataFrame): The test data.
        y_test (pd.Series): The test data,
        model (str): The model to be evaluated.
    """
    
    if model_name == "logisticregression":
        evaluator = ModelEvaluator(LogisticRegressionEvaluation())
    else:
        raise ValueError("Invalid Model.")
    
    return evaluator.execute_strategy(X_test, y_test, model)