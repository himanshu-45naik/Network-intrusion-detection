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

        logging.info("Initializing Random Forest model with predefined best parameters.")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(
                    n_estimators=97, 
                    max_samples=0.9034128710297624, 
                    max_features=0.1751204590963604, 
                    min_samples_leaf=1, 
                    random_state=42
                )),
            ]
        )

        pipeline.fit(X_train, y_train)

        logging.info("Random Forest model training completed.")

        return pipeline


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
