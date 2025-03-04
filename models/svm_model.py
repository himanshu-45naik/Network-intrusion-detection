import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class SvmModel(ModelBuildingStrategy):
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):

        logging.info("Performing cross-validation and hyperparameter tuning for SVM.")
        param_grid = {
            "svm__C": [1, 10],
            "svm__kernel": ["linear", "rbf"],
            "svm__gamma": ["auto"],
        }

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(probability=True, random_state=42)),
            ]
        )
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        logging.info("Grid search completed.")
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")

        best_pipeline = grid_search.best_estimator_
        return best_pipeline


class SvcModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """Instantiate the model strategy to be trained."""
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_train, y_train):
        return self._strategy.build_train_model(X_train, y_train)


if __name__ == "__main__":
    pass
