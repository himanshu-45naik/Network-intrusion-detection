import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class CastingData(ABC):
    """Performs casting on the data"""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs transformation on the data.

        Args:
            df(pd.DataFrame) : The input dataframe.

        Returns:
            pd.DataFrame - The transformed data."""
        pass


class DownCasting(CastingData):

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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


class DownCaster:
    @staticmethod
    def execute(df: pd.DataFrame):
        """Executes the downcasting for the given dataframe."""
        caster = DownCasting()
        return caster.transform(df)


if __name__ == "__main__":
    pass
