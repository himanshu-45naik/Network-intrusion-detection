from src.sample_data import Sampler
import pandas as pd
from zenml import step


@step
def sampling_data(
    df: pd.DataFrame, sample_size: float, random_state: int
) -> pd.DataFrame:
    """Executes sampling on the data."""

    sample_data = Sampler.execute_sampling(df, sample_size, random_state)
    return sample_data