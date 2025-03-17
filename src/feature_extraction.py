import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.decomposition import IncrementalPCA

class PCAFeatureReduction:
    def __init__(self, n_components: float = 0.5, batch_size: int = 500):
        """Initializes the PCA transformer.

        Args:
            n_components (float or int): Number of components to retain. 
                                         If float (0-1), it represents the fraction of features.
                                         If int, it represents the exact number of components.
            batch_size (int): Batch size for incremental fitting.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = None  # Placeholder for the IncrementalPCA instance

    def fit(self, X_train: pd.DataFrame):
        """Fits PCA on the training data.

        Args:
            X_train (pd.DataFrame): The training feature set.
        """
        num_features = X_train.shape[1]
        n_components = int(self.n_components * num_features) if isinstance(self.n_components, float) else self.n_components

        self.ipca = IncrementalPCA(n_components=n_components, batch_size=self.batch_size)

        # Fit PCA in batches
        for batch in np.array_split(X_train, max(1, len(X_train) // self.batch_size)):
            self.ipca.partial_fit(batch)

        logging.info(f"PCA fitted. Information retained: {sum(self.ipca.explained_variance_ratio_):.2%}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies PCA transformation to the dataset.

        Args:
            X (pd.DataFrame): The input feature set.

        Returns:
            pd.DataFrame: Transformed dataset with principal components.
        """
        transformed_data = self.ipca.transform(X)
        return pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(self.ipca.n_components_)])

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fits PCA on X_train and transforms both X_train and X_test.

        Args:
            X_train (pd.DataFrame): Training feature set.
            X_test (pd.DataFrame): Testing feature set.
            y_train (pd.Series): Training target labels.
            y_test (pd.Series): Testing target labels.

        Returns:
            tuple: Transformed X_train, X_test with PCA components, and their corresponding labels.
        """
        self.fit(X_train)

        X_train_pca = self.transform(X_train)
        X_test_pca = self.transform(X_test)

        logging.info(f"PCA transformation applied to training and testing datasets.")

        return X_train_pca, X_test_pca
