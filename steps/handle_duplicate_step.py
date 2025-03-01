from src.handle_duplicate_values import Handler
import pandas as pd
from zenml import step


@step
def drop_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicate values from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to perform transformation.

    Return:
        pd.DataFrame : Transformed dataframe.
    """
    handle_data = Handler()
    df_cleaned = handle_data.handle_duplicate_values(df)

    return df_cleaned
