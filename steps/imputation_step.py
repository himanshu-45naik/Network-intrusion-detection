from typing import Tuple

import pandas as pd
from zenml import step

from src.imputation import Imputer, SplitAwareImputation


@step
def impute_missing_values(
    X_train: pd.DataFrame, X_test: pd.DataFrame, strategy: str = "median"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fills missing values using statistics computed on the training split only.

    Runs after the train/test split so no test statistic reaches the training data.
    Downstream steps such as SMOTE cannot accept NaN, which is why imputation happens
    here as well as inside each model pipeline (the pipeline copy is what makes the
    saved MLflow artifact able to score raw feature rows at inference time).

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        strategy (str): SimpleImputer strategy. Defaults to "median", which is robust
            to the heavy-tailed flow features in this dataset.

    Returns:
        tuple: Imputed X_train and X_test.
    """
    imputer = Imputer(SplitAwareImputation(strategy=strategy))
    return imputer.execute_strategy(X_train, X_test)
