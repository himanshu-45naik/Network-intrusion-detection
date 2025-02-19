import logging
import pandas as pd
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DuplicateValuesHandling(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs transformation on the dataframe.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        pass

class DropDuplicateValues(DuplicateValuesHandling):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

class Handler:
    @staticmethod
    def handle_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the transformation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        handler = DropDuplicateValues()
        return handler.transform(df)

    
if __name__ == '__main__':
    pass        
    