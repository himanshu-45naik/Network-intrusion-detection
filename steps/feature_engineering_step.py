import pandas as pd
from zenml import step
from typing import Tuple
from src.feature_engineering import (
    FeatureEngineer,
    OneHotEncoding,
    StandardScaling,
    LabelEncodingTarget,
)


@step
def feature_engineering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    strategy: str,
) ->Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] :
    """Performs feature engineering on the data.

    Args:
        df (pd.DataFrame): The data onw hich feature engineering is performed.

    Returns:
        pd.DataFrame: The transformed data frame.
    """
    if strategy == "standard":
        feature_engineer = FeatureEngineer(StandardScaling())
        X_train, X_test = feature_engineer.execute_strategy(X_train,X_test)

    elif strategy == "onehotencoding":
        features = []
        feature_engineer = FeatureEngineer(OneHotEncoding(features))
        X_train, x_test = feature_engineer.execute_strategy(X_train,X_test)

    elif strategy == "binaryencoding":
        feature_engineer = FeatureEngineer(
            LabelEncodingTarget(target ="Attack Type", binary=True, rare_threshold=100)
        )
        print(y_train.name)
        print(y_test.name)
        y_train, y_test = feature_engineer.execute_strategy(y_train, y_test)
        
    elif strategy == "multiclassencoding":
        feature_engineer = FeatureEngineer(
            LabelEncodingTarget(target ="Attack Type", binary=False, rare_threshold=100)
        )
        y_train, y_test = feature_engineer.execute_strategy(y_train, y_test)
        
    else:
        raise ValueError(f"Unsupported feature engineering strategy:{strategy}")

    
    return X_train, X_test, y_train, y_test
