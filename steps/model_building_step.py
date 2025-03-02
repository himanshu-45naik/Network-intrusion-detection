from zenml import step
import pandas as pd
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer
from src.model_building import ModelBuilder, LogisticRegressionModel
from sklearn.linear_model import LogisticRegression


@step(output_materializers={"output": SklearnMaterializer})
def model_building(X_train: pd.DataFrame, y_train: pd.Series, Model: str)-> LogisticRegression:
    """Builds and trains the model

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training data.
        Model (str): The model to be trained
    """
    if Model == "logisticregression":
        model = ModelBuilder(LogisticRegressionModel())
    if Model == "svm":
        model = ModelBuilder()
    if Model == "rf":
        model = ModelBuilder()

    best_model = model.execute_strategy(X_train, y_train)

    return best_model
