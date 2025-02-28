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
    def transform(self, df: pd.DataFrame, features:List) -> pd.DataFrame:
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
            'BENIGN': 'BENIGN',
            'DDoS': 'DDoS',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'PortScan': 'Port Scan',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Bot',
            'Web Attack � Brute Force': 'Web Attack',
            'Web Attack � XSS': 'Web Attack',
            'Web Attack � Sql Injection': 'Web Attack',
            'Infiltration': 'Infiltration',
            'Heartbleed': 'Heartbleed'
        }

        # Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
        df['Attack Type'] = df['Label'].map(attack_map)
        df.drop('Label', axis=1, inplace = True)
        
        return df
        
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

        return df_cleaned


class FillingMissingValues(HandlingStrategy):

    def __init__(self, method="mean", fill_value = None):
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
                df_cleaned[feature] = df[feature].fillna(df[feature].mode()[0])  # mode returns a series
            elif self.method == "constant":
                if self.fill_value is None:
                    raise ValueError("fill_value must be provided for 'constant' method")
                df_cleaned[feature] = df[feature].fillna(self.fill_value)
            else:
                logging.warning(f"Unknown method '{self.method}'")

        logging.info("Missing values filled")

        return df_cleaned


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
