import logging.config
import pandas as pd
import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method for applying feature engineering strategy

        Args:
            df (pd.DataFrame): The dataframe on which feature engineering is performed

        Returns:
            pd.DataFrame: The engineered dataframe is returned
        """

        pass


class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the features on which transformtion is to be applied
        """

        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs log trasformation on the given features

        Args:
            df (pd.DataFrame): The data frame on which transformation is performed

        Returns:
            pd.DataFrame: The transformed dataframe
        """
        logging.info(f"Applying log transformation to features : {self.features}")

        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation applied")

        return df_transformed


class StandardScaling(FeatureEngineeringStrategy):

    def __init__(self, features: list):
        """Initializes standard scaling with specific features

        Args:
            features (list): The features on which the tarnsformation is performed
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Standard scaling transformation on given features

        Args:
            df (pd.DataFrame): The dataframe on which transformation is performed

        Returns:
            pd.DataFrame: The transformed dataframe
        """

        logging.info(f"Applying Standard scaling to features: {self.features}")
        df_transformed = df.copy()

        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling applied")

        return df_transformed


class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list, feature_range=(0, 1)):
        """Initializes Min-Max scaling which is performed on given features

        Args:
            features (list): The features on which scaling is perfomed
            feature_range (tuple, optional): The target range for scaling. Defaults to (0,1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies min-max scaling on given features

        Args:
            df (pd.DataFrame): The data frame on which the min-max scaling is performed.
        Returns:
            pd.DataFrame: The transformed dataframe on which min-max scaling is applied
        """
        logging.info(f"Applying min-max scaling on features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-max scaling applied")
        return df_transformed


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


class DropOneValueFeature(FeatureEngineeringStrategy):
    def __init__(self, features):
        """Initializes the features to be dropped"""
        self.features = features
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops feature which has only one unique value

        Args:
            df (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data.
        """
        num_unique = df.nunique()
        one_variable = num_unique[num_unique == 1]
        not_one_variable = num_unique[num_unique > 1].index

        dropped_cols = one_variable.index
        df = df[not_one_variable]

        logging.info(f"Sucessfully dropped columns {dropped_cols}")
        
        return df


class FeatureEngineer:
    def __init__(self, strategy):
        """Initializes the strategy based on which feature engineering is performed"""

        self._strategy = strategy

    def set_strategy(self, strategy):
        """Sets the strategy based on which the feature engineering is performed"""
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame):
        """Executes the strategy to perform feature engineering on the dataframe"""
        return self._strategy.apply_transformation(df)
        


if __name__ == "__main__":
    pass
