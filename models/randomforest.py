import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RandomForestModel(ModelBuildingStrategy):
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Builds and trains a Random Forest model."""

        logging.info("Initializing hypertuning of RF model.")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )

        param_grid = {
            "n_estimators": [97, 100],
            "max_samples": [0.9034128710297624, 1],
            "max_features": [0.1751204590963604, 0.5],
            "min_samples_leaf": [1, 2],
        }

        logging.info("Starting hyperparameter tuning using GridSearchCV.")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            verbose=3,
        )

        grid_search.fit(X_train, y_train)

        logging.info(f"Best Parameters: {grid_search.best_params_}")
        best_pipeline = grid_search.best_estimator_

        return best_pipeline


class RandomForestModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """Instantiate the model strategy to be trained."""
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_train, y_train):
        return self._strategy.build_train_model(X_train, y_train)


if __name__ == "__main__":
    pass
