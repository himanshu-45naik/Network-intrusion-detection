from zenml import step
import pandas as pd
import numpy as np
from src.data_handling import (
    Handler,
    ReplaceInfinteValues,
    FillingMissingValues,
    ReplaceFeatureNames,
    DropDuplicateValues,
    DownCasting
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


@step
def handling_data(
    df: pd.DataFrame, filling_strategy: str = "mean", fill_value=None
) -> pd.DataFrame:
    """Handles Data for missing and infinite values.

    Args:
        df (pd.DataFrame): The dataframe on which transformation is performed.
        handling_strategy (str): Strategy for handling missing/infinite values.
        filling_strategy (str): Strategy to fill missing values (mean, median, mode, constant). Defaults to "mean".
        fill_value (Optional[float]): Value to use for "constant" strategy.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    
    # Replacing feature names.
    handler1 = Handler(ReplaceFeatureNames())
    feature_names = df.columns
    updated_raw_df = handler1.execute_strategy(df, feature_names)

    # Dropping Duplicate values.
    handler2 = Handler(DropDuplicateValues())  
    transformed_df = handler2.execute_strategy(updated_raw_df, features=None)

    # Replacing infinity values with NaN
    handler3 = Handler(ReplaceInfinteValues())

    numeric_df = transformed_df.select_dtypes(include=["number"])
    inf_features = [col for col in numeric_df.columns if transformed_df[col].isin([np.inf, -np.inf]).any()]

    if inf_features:
        transformed_df = handler3.execute_strategy(transformed_df, inf_features)
    else:
        logging.info("No infinite values found, skipping handling.")

    # Filling missing values
    numeric_df = transformed_df.select_dtypes(include=["number"])
    missing_features = [col for col in numeric_df.columns if transformed_df[col].isnull().any()]

    if missing_features:
        handler4 = FillingMissingValues(method=filling_strategy, fill_value=fill_value)

        if filling_strategy not in ["mean", "median", "mode", "constant"]:
            raise ValueError(f"Unsupported missing value handling strategy: {filling_strategy}")

        transformed_df = handler4.execute_strategy(transformed_df, missing_features)
    else:
        logging.info("No missing values found, skipping handling.")

    # Downcasting
    handler5 = Handler(DownCasting())
    downcasted_df = handler5.execute_strategy(transformed_df)

    return downcasted_df  
