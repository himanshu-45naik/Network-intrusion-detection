import pandas as pd
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Builds and trains the model

        Args:
            X_train, y_train : The data on which the model trains."""

        pass


class LogisticRegressionModel(ModelBuildingStrategy):
    def build_train_model(self, X_train, y_train):
        """Builds logistic regression model

        X_train, y_train : The training data."""

        logging.info("Starting hyperparameter tuning for Logistic Regression.")

        # Define the hyperparameter grid for tuning
        param_grid = {
            "C": [10, 100],
            "penalty": ["l1","l2"],
            "solver": ["liblinear"],  # liblinear supports both l1 and l2 penalties
        }

        # Setup StratifiedKFold for balanced folds
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Get an unfitted LogisticRegression instance from model.py
        lr = LogisticRegression(random_state=42)

        # Setup GridSearchCV with the estimator, parameter grid, and CV strategy
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=skf,
            scoring="accuracy",
            n_jobs=1,
            verbose=1,
        )

        # Perform grid search
        grid_search.fit(X_train, y_train)

        logging.info(
            "Tuning complete. Best hyperparameters for Logistic regression: %s",
            grid_search.best_params_,
        )

        return grid_search.best_estimator_


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
