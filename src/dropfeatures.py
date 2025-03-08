from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class DropFeatureStrategy(ABC):
    @abstractmethod
    def drop_features(self,df:pd.DataFrame, features:list)->pd.DataFrame:
        """Drops the features from the dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.
            features (list): The features to dropped from dataframe dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """

class DropOneValueFeature(DropFeatureStrategy):

    def drop_features(self, df: pd.DataFrame,features:list) -> pd.DataFrame:
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
