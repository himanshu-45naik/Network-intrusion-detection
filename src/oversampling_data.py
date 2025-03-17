import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from imblearn.combine import SMOTE
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


class SamplingStrategy(ABC):
    """Strategy for sampling unbalanced data."""

    @abstractmethod
    def transform(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Performs sampling on data to convert it into balanced data.

        Args:
            x_train (pd.DataFrame): The unbalanced data.
            y_train (pd.DataFrame): The unbalanced label data.


        Returns:
            pd.DataFrame: The transformed balanced data.
        """
        pass


class SyntheticMinortyOverSampling(SamplingStrategy):
    def transform(self, x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Transforms the unbalanced data using SMOTE for classes with sufficient samples.
        
        Args:
            x_train (pd.DataFrame): The unbalanced data.
            y_train (pd.Series): The unbalanced label data.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: The transformed balanced data.
        """
        logging.info(f"Before resampling X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"Unique classes in y_train: {y_train.unique()}")
        
        # Count samples per class
        class_counts = y_train.value_counts()
        min_samples_needed = 6 
        
        # Identify classes with too few samples
        small_classes = class_counts[class_counts < min_samples_needed].index.tolist()
        
        if small_classes:
            logging.info(f"Classes with fewer than {min_samples_needed} samples: {small_classes}")
            
            # Create masks for samples to process with SMOTE and samples to keep as-is
            mask_for_smote = ~y_train.isin(small_classes)
            
            if mask_for_smote.sum() > 0:
                x_smote = x_train[mask_for_smote]
                y_smote = y_train[mask_for_smote]
                
                logging.info(f"Applying SMOTE to {len(y_smote.unique())} classes with sufficient samples")
                resampler = SMOTE(random_state=42)
                x_resampled_smote, y_resampled_smote = resampler.fit_resample(x_smote, y_smote)
                
                # Keep the small classes as they are
                x_small = x_train[~mask_for_smote]
                y_small = y_train[~mask_for_smote]
                
                # Combine the SMOTE-processed data with the untouched small classes
                x_resampled = pd.concat([pd.DataFrame(x_resampled_smote, columns=x_train.columns), x_small])
                y_resampled = pd.concat([pd.Series(y_resampled_smote, name=y_train.name), y_small])
            else:
                # If all classes are small, return the original data
                logging.info("All classes have insufficient samples. Returning original data.")
                x_resampled, y_resampled = x_train.copy(), y_train.copy()
        else:
            # If all classes have sufficient samples, apply SMOTE to all
            logging.info("All classes have sufficient samples. Applying SMOTE to all data.")
            resampler = SMOTE(random_state=42)
            x_resampled, y_resampled = resampler.fit_resample(x_train, y_train)
            x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        logging.info("Successfully performed resampling.")
        logging.info(f"After resampling X_resampled shape: {x_resampled.shape}, y_resampled shape: {y_resampled.shape}")
        logging.info(f"Unique classes in y_resampled: {y_resampled.unique()}")
        
        return x_resampled, y_resampled


class Sampler:
    def __init__(self, strategy: SamplingStrategy):
        """Initializes strategy with which sampling is performed."""
        self._strategy = strategy

    def set_strategy(self, strategy: SamplingStrategy):
        """Sets strategy with which sampling is performed."""
        self._strategy = strategy

    def executer_strategy(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Executes the specific strategy to perform sampling on the data."""

        return self._strategy.transform(x_train, y_train)


if __name__ == "__main__":
    pass
