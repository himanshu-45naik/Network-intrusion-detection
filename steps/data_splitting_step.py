from typing import Tuple
import pandas as pd
from src.data_splitting import DataSplitter,DropOneValueFeature, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_updation_step(
    df: pd.DataFrame, target_column: str,
    strategy: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""

    if strategy == "split_data":
        splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
        X_train, X_test, y_train, y_test = splitter.split(df, target_column)
        return X_train, X_test, y_train, y_test

    elif strategy == "dropfeatures":
        dropper = DataSplitter(DropOneValueFeature())
        num_unique = df.nunique()
        updated_df = dropper.split(df, num_unique)
        return updated_df