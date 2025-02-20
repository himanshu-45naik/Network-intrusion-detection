from zenml import step
import pandas as pd
from src.downcast_data import DownCaster


@step
def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Performs downcasting on the data

    Args:
        df(pd.DataFrame) : The input data.

    Returns:
        pd.DataFrame : The transformed data."""

    caster = DownCaster()
    return caster.execute(df)
