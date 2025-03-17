import pandas as pd
import logging
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Abstract method for applying feature engineering strategy

        Args:
            df (pd.DataFrame): The dataframe on which feature engineering is performed

        Returns:
            pd.DataFrame: The engineered dataframe is returned
        """

        pass


class StandardScaling(FeatureEngineeringStrategy):

    def apply_transformation(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies Standard scaling transformation on given features

        Args:
            df (pd.DataFrame): The dataframe on which transformation is performed

        Returns:
            pd.DataFrame: The transformed dataframe
        """

        logging.info(f"Applying Standard scaling to features: {self.features}")
        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Standard scaling applied")

        return X_train_scaled, X_test_scaled


class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """Initializes features on which onehotencoding is applied"""

        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame):
        """Applies on hot encoding on categoriacal features

        Args:
            df (pd.DataFrame): The dataframe containing the features
        """

        logging.info(
            f"Applying one hot encoding on the categorical features:{self.features}"
        )

        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )

        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One hot encoding completed")
        return df_transformed


class LabelEncodingTarget(FeatureEngineeringStrategy):
    def __init__(self,binary: bool = False):
        """Initializes the feature for performing label encoding.

        Args:
            target (str): The target column name.
            binary (bool): Whether to perform binary encoding (BENIGN vs attack).
        """
        self.binary = binary

    def apply_transformation(self, y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Performs label encoding on the given dataframe.

        Args:
            y_train (pd.DataFrame): Training target labels.
            y_test (pd.DataFrame): Testing target labels.

        Returns:
            tuple: Transformed y_train, y_test, and the label encoder (for reference).
        """
        y_train = y_train.copy()
        y_test = y_test.copy()

        if self.binary:
            y_train_updated = y_train.apply(lambda x: 0 if x == "BENIGN" else 1)
            y_train = pd.Series(y_train_updated)
            y_test_updated = y_test.apply(lambda x: 0 if x == "BENIGN" else 1)
            y_test = pd.Series(y_test_updated)
            logging.info("Binary encoding of target feature successfully performed.")
            logging.info("Binary encoding completed: 'BENIGN' → 0, 'ATTACK' → 1")

        else:
            self.encoder = LabelEncoder()
            y_train_updated = self.encoder.fit_transform(y_train)
            y_train = pd.Series(y_train_updated)
            y_test_updated = self.encoder.transform(y_test) 
            y_test = pd.Series(y_test_updated)
            
            encoding_map = dict(zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_)))
            logging.info(f"Multiclass label encoding mapping: {encoding_map}")
            logging.info("Label encoding for multiclass classification successfully performed.")

        return y_train, y_test



class FeatureEngineer:
    def __init__(self, strategy):
        """Initializes the strategy based on which feature engineering is performed"""

        self._strategy = strategy

    def set_strategy(self, strategy):
        """Sets the strategy based on which the feature engineering is performed"""
        self._strategy = strategy

    def execute_strategy(self, df1, df2):
        """Executes the strategy to perform feature engineering on the dataframe"""
        return self._strategy.apply_transformation(df1, df2)


if __name__ == "__main__":
    pass
