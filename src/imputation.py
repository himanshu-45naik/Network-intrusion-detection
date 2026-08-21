import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ImputationStrategy(ABC):
    @abstractmethod
    def fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fits the imputer on the training split and applies it to both splits.

        Args:
            X_train (pd.DataFrame): Training features, the only data the imputer may see.
            X_test (pd.DataFrame): Test features, transformed with the training statistics.

        Returns:
            tuple: Imputed X_train and X_test.
        """
        pass


class SplitAwareImputation(ImputationStrategy):
    def __init__(self, strategy: str = "median"):
        """Initializes the imputation strategy.

        Args:
            strategy (str): Any strategy accepted by sklearn's SimpleImputer
                ("mean", "median", "most_frequent", "constant").
        """
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)

    def fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fits on X_train only, then transforms X_train and X_test.

        Fitting on the training split alone is what keeps test statistics out of the
        training data; a fill value computed over the full dataset would leak.
        """
        columns = X_train.columns

        logging.info(
            f"Fitting '{self.strategy}' imputer on {len(X_train)} training rows "
            f"({int(X_train.isnull().sum().sum())} missing values); "
            f"test split has {int(X_test.isnull().sum().sum())} missing values."
        )

        self.imputer.fit(X_train)

        X_train_imputed = pd.DataFrame(
            self.imputer.transform(X_train), columns=columns, index=X_train.index
        )
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test), columns=columns, index=X_test.index
        )

        logging.info("Imputation applied using training-split statistics only.")

        return X_train_imputed, X_test_imputed


class Imputer:
    def __init__(self, strategy: ImputationStrategy):
        """Initializes the strategy with which imputation is performed."""
        self._strategy = strategy

    def set_strategy(self, strategy: ImputationStrategy):
        """Sets the strategy with which imputation is performed."""
        self._strategy = strategy

    def execute_strategy(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Executes the strategy to impute missing values."""
        return self._strategy.fit_transform(X_train, X_test)


if __name__ == "__main__":
    pass
