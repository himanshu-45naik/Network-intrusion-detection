import logging

import numpy as np
import pandas as pd
from zenml import step

from src.data_handling import (
    DownCasting,
    DropDuplicateValues,
    Handler,
    ReplaceFeatureNames,
    ReplaceInfinteValues,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


@step
def handling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw dataframe using only split-independent transformations.

    Every operation here is row-wise or metadata-only, so it is safe to run before
    the train/test split. Missing-value imputation is deliberately NOT done here:
    it fits statistics on the data it sees, so running it pre-split would leak test
    statistics into training. Imputation happens after the split (see
    steps/imputation_step.py) and again inside each model pipeline for inference.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe, with infinite values represented as NaN.
    """

    # Replacing feature names.
    handler1 = Handler(ReplaceFeatureNames())
    feature_names = df.columns
    updated_raw_df = handler1.execute_strategy(df, feature_names)

    # Dropping Duplicate values (before the split, so identical rows cannot land in
    # both train and test).
    handler2 = Handler(DropDuplicateValues())
    transformed_df = handler2.execute_strategy(updated_raw_df, features=None)

    # Replacing infinity values with NaN
    handler3 = Handler(ReplaceInfinteValues())

    numeric_df = transformed_df.select_dtypes(include=["number"])
    inf_features = [
        col for col in numeric_df.columns if transformed_df[col].isin([np.inf, -np.inf]).any()
    ]

    if inf_features:
        transformed_df = handler3.execute_strategy(transformed_df, inf_features)
    else:
        logging.info("No infinite values found, skipping handling.")

    missing_count = int(transformed_df.isnull().sum().sum())
    logging.info(f"{missing_count} missing values left for post-split imputation.")

    # Downcasting
    handler5 = Handler(DownCasting())
    downcasted_df = handler5.execute_strategy(transformed_df, features=None)

    return downcasted_df
