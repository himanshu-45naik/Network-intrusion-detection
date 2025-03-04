import pandas as pd
from zenml import step
from src.feature_engineering import (
    FeatureEngineer,
    OneHotEncoding,
    MinMaxScaling,
    StandardScaling,
    LogTransformation,
    DropOneValueFeature,
    LabelEncodingTarget,
)


@step
def feature_engineering(
    df: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    """Performs feature engineering on the data.

    Args:
        df (pd.DataFrame): The data onw hich feature engineering is performed.

    Returns:
        pd.DataFrame: The transformed data frame.
    """
    if strategy == "log":
        features = []
        feature_engineer = FeatureEngineer(LogTransformation(features))

    elif strategy == "standard":
        features_df = df.drop("Attack Type", axis=1)
        features = features_df.columns
        feature_engineer = FeatureEngineer(StandardScaling(features))

    elif strategy == "min-max":
        features = []
        feature_engineer = FeatureEngineer(MinMaxScaling(features))

    elif strategy == "onehotencoding":
        features = []
        feature_engineer = FeatureEngineer(OneHotEncoding(features))

    elif strategy == "dropfeatures":
        num_unique = df.nunique()
        num_unique = num_unique[num_unique == 1].index.tolist()
        feature_engineer = FeatureEngineer(DropOneValueFeature(num_unique))
        
    elif strategy == "binaryencoding":
        feature_engineer = FeatureEngineer(
            LabelEncodingTarget("Attack Type", Binary=True)
        )
        
    elif strategy == "multiclassencoding":
        feature_engineer = FeatureEngineer(
            LabelEncodingTarget("Attack Type", Binary=False)
        )
        
    else:
        raise ValueError(f"Unsupported feature engineering strategy:{strategy}")

    transformed_data = feature_engineer.execute_strategy(df)
    return transformed_data
