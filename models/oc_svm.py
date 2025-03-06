import pandas as pd
import logging
from models.base_model import ModelBuildingStrategy
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class OneClassSvmModel(ModelBuildingStrategy):
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> Pipeline:
        """Builds One-Class SVM model for predicting network intrusion.

        Args:
            X_train (pd.DataFrame): The training data (only BENIGN samples).

        Returns:
            Pipeline: The trained pipeline.
        """

        if "Attack Type" in X_train.columns:
            X_train = X_train[X_train["Attack Type"] == 0].drop(columns=["Attack Type"])
        else:
            logging.warning("Column 'Attack Type' not found in X_train. Proceeding without filtering.")

        logging.info("Training One-Class SVM model.")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ocsvm", OneClassSVM(kernel="rbf", gamma="scale", nu=0.5, verbose=True))
            ]
        )

        pipeline.fit(X_train)
        logging.info("OC-SVM model training completed.")

        return pipeline


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
