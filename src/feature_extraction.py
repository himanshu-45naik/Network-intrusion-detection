import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.decomposition import IncrementalPCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PCAFeatureReduction:
    def __init__(self, n_components: float = 0.5, batch_size: int = 500):
        """Initializes the PCA transformer.

        Args:
            n_components (float or int or str): Number of components to retain. 
                - If float (0-1), it represents the fraction of features.
                - If int, it represents the exact number of components.
                - If "mle", it uses Minka's MLE to determine the optimal number of components.
            batch_size (int): Batch size for incremental fitting.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = None  

    def fit(self, X_train: pd.DataFrame):
        """Fits PCA on the training data.

        Args:
            X_train (pd.DataFrame): The training feature set.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame.")

        num_features = X_train.shape[1]

        if isinstance(self.n_components, float):
            if not (0 < self.n_components <= 1):
                raise ValueError("n_components as a float should be between 0 and 1.")
            n_components = int(self.n_components * num_features)
        elif isinstance(self.n_components, int):
            n_components = self.n_components
        elif self.n_components == "mle":
            n_components = "mle"
        else:
            raise ValueError("Invalid value for n_components. Use int, float, or 'mle'.")

        self.ipca = IncrementalPCA(n_components=n_components, batch_size=self.batch_size)

        # Fit PCA in batches
        batch_splits = max(1, min(len(X_train) // self.batch_size, 50))  
        for batch in np.array_split(X_train, batch_splits):
            self.ipca.partial_fit(batch)

        explained_variance = sum(self.ipca.explained_variance_ratio_)
        logging.info(f"PCA fitted. {explained_variance:.2%} variance retained.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies PCA transformation to the dataset.

        Args:
            X (pd.DataFrame): The input feature set.

        Returns:
            pd.DataFrame: Transformed dataset with principal components.
        """
        if self.ipca is None:
            raise RuntimeError("PCA model has not been fitted. Call 'fit' before 'transform'.")

        transformed_data = self.ipca.transform(X)
        return pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(self.ipca.n_components_)])

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fits PCA on X_train and transforms both X_train and X_test.

        Args:
            X_train (pd.DataFrame): Training feature set.
            X_test (pd.DataFrame): Testing feature set.

        Returns:
            tuple: Transformed X_train and X_test with PCA components.
        """
        self.fit(X_train)
        X_train_pca = self.transform(X_train)
        X_test_pca = self.transform(X_test)

        logging.info(f"PCA transformation applied. Final shape: {X_train_pca.shape}, {X_test_pca.shape}")

        return X_train_pca, X_test_pca
