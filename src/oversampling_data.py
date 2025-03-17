import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from imblearn.combine import SMOTETomek
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class SamplingStrategy(ABC):
    """Strategy for sampling unbalanced data."""

    @abstractmethod
    def transform(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Performs sampling on data to convert it into balanced data.

        Args:
            x_train (pd.DataFrame): The unbalanced data.
            y_train (pd.DataFrame): The unbalanced label data.


        Returns:
            pd.DataFrame: The transformed balanced data.
        """
        pass


class SyntheticMinortyOverSampling(SamplingStrategy):
    def transform(self, x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Transforms the unbalanced data using SMOTE.

        Args:
            x_train (pd.DataFrame): The unbalanced data.
            y_train (pd.DataFrame): The unbalanced label data.


        Returns:
            pd.DataFrame: The transformed balanced data.
        """
        logging.info(f"Before resampling X_resampled shape: {x_train.shape}, y_resampled shape: {y_train.shape}")
        logging.info(f"Unique classes in y_train: {y_train.unique()}")
        smote = SMOTETomek(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
        x_resampled = pd.DataFrame(x_resampled)
        y_resampled = pd.Series(y_resampled)
        logging.info("Successfully performed SMOTE.")
        logging.info(f"After resamplingl X_resampled shape: {x_resampled.shape}, y_resampled shape: {y_resampled.shape}")
        logging.info(f"Unique classes in y_train: {y_train.unique()}")
        return x_resampled, y_resampled


class Sampler:
    def __init__(self, strategy: SamplingStrategy):
        """Initializes strategy with which sampling is performed."""
        self._strategy = strategy

    def set_strategy(self, strategy: SamplingStrategy):
        """Sets strategy with which sampling is performed."""
        self._strategy = strategy

    def executer_strategy(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Executes the specific strategy to perform sampling on the data."""

        return self._strategy.transform(x_train, y_train)


if __name__ == "__main__":
    pass
