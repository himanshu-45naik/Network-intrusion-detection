import pandas as pd
from src.oversampling_data import Sampler, SyntheticMinortyOverSampling
from typing import Tuple
from zenml import step


@step
def sampling_data(
    x_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """Performs sampling on the unbalanced data.

    Args:
        x_train (pd.DataFrame): The unbalanced data.
        y_train (pd.DataFrame): The unbalanced label data.

    Returns:
        pd.DataFrame: The transformed balanced data.
    """
    sampling = Sampler(SyntheticMinortyOverSampling())
    x_resampled, y_resampled = sampling.executer_strategy(x_train, y_train)

    return x_resampled, y_resampled
