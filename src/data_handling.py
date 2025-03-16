import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class HandlingStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame, features: List) -> pd.DataFrame:
        """Transforms the input DataFrame and returns the transformed DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            features (List): List of features to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        pass


class ReplaceFeatureNames(HandlingStrategy):
    def transform(self, df: pd.DataFrame, features: List) -> pd.DataFrame:
        """Replaces the names by removing the unnecessary Whitesapces

        Args:
            df (pd.DataFrame) : Input Dataframe.
            features (List) : The name of features to be changed

        Returns:
            df (pd.DataFrame): The Output DataFrame"""

        feature_names = {feature: feature.strip() for feature in features}
        df.rename(columns=feature_names, inplace=True)

        # Creating a dictionary that maps each label to its attack type
        attack_map = {
            "BENIGN": "BENIGN",
            "DDoS": "DDoS",
            "DoS Hulk": "DoS",
            "DoS GoldenEye": "DoS",
            "DoS slowloris": "DoS",
            "DoS Slowhttptest": "DoS",
            "PortScan": "Port Scan",
            "FTP-Patator": "Brute Force",
            "SSH-Patator": "Brute Force",
            "Bot": "Bot",
            "Web Attack � Brute Force": "Web Attack",
            "Web Attack � XSS": "Web Attack",
            "Web Attack � Sql Injection": "Web Attack",
            "Infiltration": "Infiltration",
            "Heartbleed": "Heartbleed",
        }

        df["Attack Type"] = df["Label"].map(attack_map)
        df.drop("Label", axis=1, inplace=True)

        return df

class DropDuplicateValues(HandlingStrategy):
    def transform(self, df: pd.DataFrame, features= None) -> pd.DataFrame:
        """
        Drops the duplicate values.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        df_cleaned = df.copy()
        df_cleaned.drop_duplicates(inplace=True)
        logging.info("Successfully dropped duplicate values.")
        return df_cleaned

class ReplaceInfinteValues(HandlingStrategy):
    def transform(self, df: pd.DataFrame, features: List) -> pd.DataFrame:
        """Replaces infinte values with Nan values.

        Args:
            df (pd.DataFrame): Input Dataframe.
            features (List): List of features to be trasformed.

        Returns:
            pd.DataFrame : Transformed data.
        """
        df_cleaned = df.copy()
        df_cleaned[features] = df_cleaned[features].replace([np.inf, -np.inf], np.nan)
        logging.info(f"Infinite values replaced with Nan for features {features}.")
        logging.ingo(f"Shape of dataframe after replacing infinte values ")
        return df_cleaned


class FillingMissingValues(HandlingStrategy):

    def __init__(self, method="mean", fill_value=None):
        """Initializes the specific method with which missing values are filled

        Args:
            method : Specific method with which data is filled
            fill_value : Specific value used to fill the missing value
        """

        self.method = method
        self.fill_value = fill_value

    def transform(self, df: pd.DataFrame, features: List) -> pd.DataFrame:
        """Performs changes to the missing values

        Args:
            df (pd.DataFrame): Input DataFrame
            features (List): The columns to apply the transformation to.

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df_cleaned = df.copy()

        for feature in features:
            if self.method == "mean":
                df_cleaned[feature] = df[feature].fillna(df[feature].mean())
            elif self.method == "median":
                df_cleaned[feature] = df[feature].fillna(df[feature].median())
            elif self.method == "mode":
                df_cleaned[feature] = df[feature].fillna(
                    df[feature].mode()[0]
                )  
            elif self.method == "constant":
                if self.fill_value is None:
                    raise ValueError(
                        "fill_value must be provided for 'constant' method"
                    )
                df_cleaned[feature] = df[feature].fillna(self.fill_value)
            else:
                logging.warning(f"Unknown method '{self.method}'")

        logging.info("Missing values filled")

        return df_cleaned


class DropOneValueFeature(HandlingStrategy):
    

    def transform(self, df: pd.DataFrame,features:list) -> pd.DataFrame:
        """Drops feature which has only one unique value.
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

class DownCasting(HandlingStrategy):

    def transform(self, df: pd.DataFrame, features= None) -> pd.DataFrame:
        """Converts the int64 and float64 to int32 and float32 respectively."""

        df_tranformed = df.copy()

        for col in df.columns:
            col_type = df[col].dtype

            # Downcast float64 to float32
            if col_type == "float64":
                df_tranformed[col] = df_tranformed[col].astype(np.float32)

            # Downcast int64 to int32
            elif col_type == "int64":
                df_tranformed[col] = df_tranformed[col].astype(np.int32)
        logging.info("Sucessfully downcasted the data.")
        return df_tranformed
    
class Handler:
    def __init__(self, strategy: HandlingStrategy):
        """Initializes the strategy for handling the data."""
        self._strategy = strategy

    def set_strategy(self, strategy: HandlingStrategy):
        """Sets the strategy for handling the data."""
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, features: List) -> pd.DataFrame:
        """Executes the strategy for handling the data."""
        return self._strategy.transform(df, features)


if __name__ == "__main__":
    pass
