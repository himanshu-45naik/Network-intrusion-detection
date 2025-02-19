from zenml import step
import pandas as pd
from src.data_handling import Replace_infinte_values, Filling_missing_values, Handler


@step
def handle_missing_data(
    df: pd.DataFrame, features: list, strategy: str
) -> pd.DataFrame:
    """Handles Data for missing and infinte values.

    Args:
        df (pd.DataFrame): The dataframe on which transformation is performed.
        Features (list): The list of features on which transformation is performed.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """

    if strategy in ["mean", "meadian", "mode", "constant"]:
        handle = Handler(Filling_missing_values(strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy {strategy}")

    df_cleaned = handle.execute_strategy(df, features)

    return df_cleaned


@step
def handle_infinite_values(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Handles data for infinite values by replacing it with Nan.

    Args:
        df (pd.DataFrame): The dataframe to be transformed.

    Returns:
        pd.DataFrame: The transformed dataframe.
    """
    handle = Handler(Replace_infinte_values)
    df_cleaned = handle.execute_strategy(df, features)

    return df_cleaned
