import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
import lightgbm as lgb 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LightGBMModel(ModelBuildingStrategy):
    def __init__(self, binary_class: bool):
        """Initializes the strategy of training the model.

        Args:
            binary_class (bool): If the provided data is binary or multiclass.
        """
        self.binary_class = binary_class

    def model_building(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Builds and trains a LightGBM model with hyperparameter tuning."""

        logging.info("Initializing LightGBM model hyperparameter tuning.")

        param_grid = {
            "lgb__n_estimators": [100, 200],  
            "lgb__learning_rate": [0.01, 0.1],
            "lgb__max_depth": [-1, 5, 10],
            "lgb__num_leaves": [31, 50, 100],
        }

        if self.binary_class:
            objective = "binary"
            num_class = None
        else:
            objective = "multiclass"
            num_class = len(y_train.unique())

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lgb",
                    lgb.LGBMClassifier(
                        objective=objective, num_class=num_class, metric="multi_logloss"
                    ),
                ),
            ]
        )

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring="accuracy", verbose=3, n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        logging.info(f"Best Parameters: {grid_search.best_params_}")

        best_model = grid_search.best_estimator_

        return best_model


class LightGBMBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """Instantiate the model strategy to be trained."""
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_train, y_train):
        return self._strategy.model_building(X_train, y_train)  
if __name__ == "__main__":
    pass
