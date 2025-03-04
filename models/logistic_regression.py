import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class LogisticRegressionModel(ModelBuildingStrategy):
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Builds logistic regression model

        X_train, y_train : The training data."""

        logging.info("Starting hyperparameter tuning for Logistic Regression.")

        param_grid = {
            "C": [10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }

        lr = LogisticRegression(random_state=42)

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression())]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logging.info(
            "Tuning complete. Best hyperparameters for Logistic regression: %s",
            grid_search.best_params_,
        )
        best_pipeline = grid_search.best_estimator_

        return best_pipeline


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """Instantiate the model strategy to be trained."""
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_train, y_train):
        return self._strategy.build_train_model(X_train, y_train)


if __name__ == "__main__":
    pass
