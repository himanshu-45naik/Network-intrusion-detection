import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class OneClassSvmModel(ModelBuildingStrategy):
    def ocsvm_scorer(self, estimator, X, y=None):
        """Custom scorer for One-Class SVM using mean anomaly scores."""
        return estimator.named_steps["ocsvm"].score_samples(X).mean()

    def build_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series = None
    ) -> Pipeline:
        """Builds One-Class SVM model for predicting network intrusion.

        Args:
            X_train (pd.DataFrame): The training data (only BENIGN samples).

        Returns:
            Pipeline: The trained pipeline.
        """

        X_train = X_train[X_train["Attack Type"] == 0].drop(columns=["Attack Type"])

        logging.info("Performing hyperparameter tuning for OC-SVM.")

        pipeline = Pipeline([("scaler", StandardScaler()), ("ocsvm", OneClassSVM())])

        param_grid = {
            "ocsvm__kernel": ["linear", "rbf"],
            "ocsvm__nu": [0.01, 0.05, 0.1],
            "ocsvm__gamma": ["scale", "auto"],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring=make_scorer(
                self.ocsvm_scorer, greater_is_better=False, needs_proba=False
            ),
            n_jobs=1,
            verbose=3,
        )

        grid_search.fit(X_train, y_train=None)

        logging.info("Hyperparameter tuning completed.")
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")

        best_pipeline = grid_search.best_estimator_

        return best_pipeline


class OCsvmModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """Instantiate the model strategy to be trained."""
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_train, y_train=None):
        return self._strategy.build_train_model(X_train, y_train)


if __name__ == "__main__":
    pass
