import glob
import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Define an abstract class


class Ingest_Data(ABC):
    @abstractmethod
    def ingest_df(self, data_path: str) -> pd.DataFrame:
        """
        Abstract for ingesting data from data path
        """
        pass


class ZipDataIngester(Ingest_Data):
    def ingest_df(self, data_path: str) -> pd.DataFrame:
        """Converts .zip file and returns data in pd.DataFrame format"""

        # Ensure data is in zip format
        if not data_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the file
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find extracted CSV files at any depth (zips may nest CSVs in a folder)
        csv_files = sorted(glob.glob(os.path.join("extracted_data", "**", "*.csv"), recursive=True))

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in extracted data.")
        print(f"Number of CSV files found: {len(csv_files)}.")

        # Read the CSVs into one dataframe
        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0])
        else:
            df = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)

        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> Ingest_Data:
        """Returns proper data ingestor based on extension"""
        if file_extension == ".zip":
            return ZipDataIngester()
        else:
            raise ValueError(f"No ingestor is present for this {file_extension} file extension.")


if __name__ == "__main__":
    pass
