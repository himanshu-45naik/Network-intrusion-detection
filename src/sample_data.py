import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SamplingDataset(ABC):
    @abstractmethod
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a subset of the CICIDS17 dataset.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The sampled dataframe.
        """
        pass


class SampleNidsData(SamplingDataset):

    def sample_data(
        self, df: pd.DataFrame, sample_size=0.05, random_state=42
    ) -> pd.DataFrame:
        """Performs stratified sampling on the dataset.

        Args:
            df (pd.DataFrame): The input dataframe.
            sample_size (float or int): Fraction or fixed number of samples.
            random_state (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: The sampled dataframe.
        """
        logging.info(f"Original size of dataset: {len(df)}")

        df = df.reset_index(drop=True)

        if isinstance(sample_size, float) and sample_size < 1.0:
            sampled_indices = self.stratified_sample_indices(
                df["Attack Type"], sample_size, random_state
            )
            sampled_df = df.iloc[sampled_indices]
        elif isinstance(sample_size, int) and sample_size < len(df):
            sampled_indices = self.stratified_sample_indices(
                df["Attack Type"], sample_size / len(df), random_state
            )
            sampled_df = df.iloc[sampled_indices]
        else:
            sampled_df = df
            logging.warning("Invalid sample_size, using the entire dataset.")

        self.validate_sample_diversity(df["Attack Type"], sampled_df["Attack Type"])

        logging.info(f"Sampled dataset size: {len(sampled_df)}")
        return sampled_df

    @staticmethod
    def stratified_sample_indices(
        labels: pd.Series, sample_size: float, random_state=42
    ):
        """
        Create indices for a stratified sample of the dataset.

        Args:
        labels : pd.Series
            The labels/classes used for stratification.
        sample_size : float
            Fraction of data to sample (0-1).
        random_state : int
            Random seed for reproducibility.

        Returns:
        indices : list
            List of sampled indices.
        """
        unique_labels = labels.unique()
        indices = []
        np.random.seed(random_state)

        for label in unique_labels:
            label_indices = labels[labels == label].index.tolist()
            n_samples = min(int(len(label_indices) * sample_size),len(label_indices))
            n_samples = max(1, n_samples)  # Ensure at least one sample per class
            sampled_indices = np.random.choice(
                label_indices, size=n_samples, replace=False
            )
            indices.extend(sampled_indices)

        return indices

    @staticmethod
    def validate_sample_diversity(original_labels, sampled_labels):
        """
        Validate that the sampled dataset represents the diversity of the original dataset
        by comparing class distributions.

        """
        logging.info("\n--- Sample Diversity Validation ---")

        # Count occurrences of each class
        original_counts = Counter(original_labels)
        sampled_counts = Counter(sampled_labels)

        # Calculate class distribution percentages
        total_original = sum(original_counts.values())
        total_sampled = sum(sampled_counts.values())

        # Prepare data for comparison
        classes = sorted(original_counts.keys())

        # Create a comparison dataframe
        comparison_data = []
        for cls in classes:
            orig_pct = (original_counts.get(cls, 0) / total_original) * 100
            samp_pct = (sampled_counts.get(cls, 0) / total_sampled) * 100

            comparison_data.append(
                {
                    "Class": cls,
                    "Original %": orig_pct,
                    "Sampled %": samp_pct,
                    "Sampling Diff": samp_pct - orig_pct,
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        logging.info(
            "Class distribution comparison:\n%s",
            comparison_df.round(2).to_string(index=False),
        )

        # Check if any class is missing from the sampled set
        missing_sampled = set(original_counts.keys()) - set(sampled_counts.keys())

        if missing_sampled:
            logging.warning(
                f"WARNING: The following classes are missing from the sampled set: {missing_sampled}"
            )

        # Make a recommendation based on the validation
        max_sampling_diff = comparison_df["Sampling Diff"].abs().max()

        logging.info("\nValidation Results:")
        if max_sampling_diff < 1.0:
            logging.info(
                "✓ SAMPLING: Excellent representation of original class distribution (max difference < 1%)"
            )
        elif max_sampling_diff < 3.0:
            logging.info(
                "✓ SAMPLING: Good representation of original class distribution (max difference < 3%)"
            )
        elif max_sampling_diff < 5.0:
            logging.warning(
                "⚠ SAMPLING: Moderate deviations in class distribution (max difference < 5%)"
            )
        else:
            logging.error(
                "✗ SAMPLING: Significant deviations in class distribution (max difference >= 5%)"
            )


class Sampler:
    @staticmethod
    def execute_sampling(df: pd.DataFrame, sample_size: float, random_state: int):
        """Executes the sampling of data,"""

        sampler = SampleNidsData()
        return sampler.sample_data(df, sample_size, random_state)
