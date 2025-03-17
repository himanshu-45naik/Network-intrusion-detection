import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LgbModel(ModelBuildingStrategy):
    def __init__(self, binary_class: bool):
        """Initializes the strategy of training the model.

        Args:
            binary_class (bool): If the provided data is binary or multiclass
        """
        self.binary_class = binary_class

    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Builds and trains LightGBM model."""

        logging.info("Initializing LightGBM model hyperparameter tuning.")

        # Define hyperparameter grid
        param_grid = {
            "lgb__n_estimators": [100, 200],
            "lgb__learning_rate": [0.01, 0.1],
            "lgb__max_depth": [3, 5, 7],
            "lgb__subsample": [0.8, 1.0],
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
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="accuracy", verbose=3, n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        logging.info(f"Best Parameters: {grid_search.best_params_}")

        best_model = grid_search.best_estimator_

        return best_model


class LgbBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """Instantiate the model strategy to be trained."""
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_train, y_train):
        return self._strategy.build_train_model(X_train, y_train)


if __name__ == "__main__":
    pass
