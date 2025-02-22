import pandas as pd
from zenml import step
from src.feature_extraction import FeatureExtractor, PrincipalComponentAnalysis


@step
def feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Peforms feature extraction on the dataset."""

    extractor = FeatureExtractor(PrincipalComponentAnalysis)
    return extractor.execute_strategy(df)
