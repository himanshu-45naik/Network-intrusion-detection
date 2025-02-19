from src.ingest_data import DataIngestorFactory
from zenml import step
import pandas as pd


@step
def data_ingestion(file_path: str) -> pd.DataFrame:
    """Ingest data from a zip file using the appropriate Data ingester

    Args:
        file_path (str): the path of file where the zip file exists
    """
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_path)

    df = data_ingestor.ingest_df(file_path)

    return df
