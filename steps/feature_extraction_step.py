import pandas as pd
from zenml import step
from typing import Tuple
from src.feature_extraction import PCAFeatureReduction


@step
def feature_extraction(X_train: pd.DataFrame, X_test:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Peforms feature extraction on the dataset."""

    extractor = PCAFeatureReduction()
    X_train, X_test = extractor.fit_transform(X_train, X_test)
    
    return X_train, X_test