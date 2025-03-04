import pandas as pd
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Builds and trains the model

        Args:
            X_train, y_train : The data on which the model trains."""

        pass
