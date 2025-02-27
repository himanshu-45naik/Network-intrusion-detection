from zenml import step
import pandas as pd
import numpy as np
from src.data_handling import Replace_infinte_values, Filling_missing_values, Handler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")

@step
def handle_missing_data(
    df: pd.DataFrame, strategy: str = "mean", fill_value = None
) -> pd.DataFrame:
    """Handles Data for missing and infinte values.

    Args:
        df (pd.DataFrame): The dataframe on which transformation is performed.
        strategy (str): The strategy to fill missing values (mean, median, mode, constant).  Defaults to "mean".
        fill_value (Optional[float]):  Value to use for "constant" strategy.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """

    numeric_df = df.select_dtypes(include=["number"])
    missing_features = [f for f in numeric_df.columns if df[f].isnull().any()]

    if not missing_features:
        logging.info("No missing values found, skipping handling.")
        return df.copy()

    if strategy in ["mean", "median", "mode", "constant"]:
        handle = Filling_missing_values(method=strategy, fill_value=fill_value) # No Handler needed, use the strategy directly
        df_cleaned = handle.transform(df, missing_features) # Call the transform method directly
        return df_cleaned
    else:
        raise ValueError(f"Unsupported missing value handling strategy {strategy}")


@step
def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles data for infinite values by replacing it with Nan.

    Args:
        df (pd.DataFrame): The dataframe to be transformed.

    Returns:
        pd.DataFrame: The transformed dataframe.
    """
    numeric_df = df.select_dtypes(include=["number"])
    inf_features = [
        col for col in numeric_df.columns if df[col].isin([np.inf, -np.inf]).any()
    ]

    if not inf_features:
        logging.info("No infinite values found, skipping handling.")
        return df.copy()


    handle = Replace_infinte_values()
    df_cleaned = handle.transform(df, inf_features)

    return df_cleaned