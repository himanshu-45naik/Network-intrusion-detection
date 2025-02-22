import pandas as pd
from zenml import step
from src.feature_engineering import (
    FeatureEngineer,
    OneHotEncoding,
    MinMaxScaling,
    StandardScaling,
    LogTransformation,
    DropOneValueFeature,
)


@step
def feature_engineering(
    df: pd.DataFrame, strategy: str, features: list
) -> pd.DataFrame:
    """Performs feature engineering on the data.

    Args:
        df (pd.DataFrame): The data onw hich feature engineering is performed.

    Returns:
        pd.DataFrame: The transformed data frame.
    """
    if strategy == "log":
        feature_engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard":
        feature_engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "min-max":
        feature_engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehotencoding":
        feature_engineer = FeatureEngineer(OneHotEncoding(features))
    elif strategy == "dropfeatures":
        feature_engineer = FeatureEngineer(DropOneValueFeature)
    else:
        raise ValueError(f"Unsupported feature engineering strategy:{strategy}")

    transformed_data = feature_engineer.apply_transformation(df)
