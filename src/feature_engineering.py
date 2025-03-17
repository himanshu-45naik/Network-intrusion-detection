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

        logging.info(f"Applying Standard scaling to features")
        scaler = StandardScaler()

        scaler.fit(X_train)
        original_feature_names = X_train.columns.tolist()

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=original_feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=original_feature_names)

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


class LabelEncodingTarget:
    def __init__(self, target: str, binary: bool = False, rare_threshold: int = 100):
        """Initializes the feature for performing label encoding with rare class merging.

        Args:
            target (str): The target column name.
            binary (bool): Whether to perform binary encoding (BENIGN vs attack).
            rare_threshold (int): Minimum count for a class to be kept separate.
        """
        self.target = target
        self.binary = binary
        self.rare_threshold = rare_threshold
        self.encoder = LabelEncoder()

    def apply_transformation(
        self, y_train: pd.Series, y_test: pd.Series
    ) -> Tuple[pd.Series, pd.Series, dict]:
        """Performs label encoding and merges rare classes into 'Other_Attacks'.

        Args:
            y_train (pd.Series): Training target labels.
            y_test (pd.Series): Testing target labels.

        Returns:
            tuple: Encoded y_train, y_test, and the encoding map.
        """
        y_train = y_train.copy()
        y_test = y_test.copy()

        if self.binary:
            # Binary Encoding: BENIGN = 0, ATTACK = 1
            y_train_encoded = y_train.apply(lambda x: 0 if x == "BENIGN" else 1)
            y_test_encoded = y_test.apply(lambda x: 0 if x == "BENIGN" else 1)
            encoding_map = {"BENIGN": 0, "ATTACK": 1}

            logging.info("Binary encoding completed: 'BENIGN' → 0, 'ATTACK' → 1")
        else:
            # Identify rare classes
            class_counts = y_train.value_counts()
            rare_classes = class_counts[class_counts < self.rare_threshold].index.tolist()
            
            # Merge rare classes into 'Other_Attacks'
            y_train = y_train.apply(lambda x: "Other_Attacks" if x in rare_classes else x)
            y_test = y_test.apply(lambda x: "Other_Attacks" if x in rare_classes else x)
            
            # Fit Label Encoder
            y_train_encoded = self.encoder.fit_transform(y_train)
            y_test_encoded = self.encoder.transform(y_test)
            
            encoding_map = dict(zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_)))
            logging.info(f"Merged rare classes {rare_classes} into 'Other_Attacks'.")
            logging.info(f"Final label encoding mapping: {encoding_map}")

        return pd.Series(y_train_encoded, name=self.target), pd.Series(y_test_encoded, name=self.target)

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
