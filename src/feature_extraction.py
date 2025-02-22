import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from sklearn.decomposition import IncrementalPCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class FeatureExtractionStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs feautre extraction on the dataframe.

        Args:
            df(pd.DataFrame) : The dataframe on which feature extraction is performed.

        Returns:
            df(pd.DataFrame) : The transformed dataframe."""

        pass


class PrincipalComponentAnalysis(FeatureExtractionStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs PCA on the dataframe

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """

        size = df.columns // 2
        ipca = IncrementalPCA(n_components=size, batch_size=500)

        for batch in np.array_split(df, len(df) // 500):
            ipca.partial_fit(batch)

        logging.info(
            f"Performed PCA .Information retained: {sum(ipca.explained_variance_ratio_):.2%}"
        )
        transformed_df = ipca.transform(df)
        new_data = pd.DataFrame(
            transformed_df, columns=[f"PC{i+1}" for i in range(size)]
        )
        
        return new_data


class FeatureExtractor:
    def __init__(self, strategy: FeatureExtractionStrategy):
        """Initializes the strataegy to execute"""
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureExtractionStrategy):
        """Set strategy to the given specific strategy"""
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame):
        """Executes given strategy"""
        self._strategy.transform(df)
