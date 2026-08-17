import pandas as pd
from zenml import step

from src.ingest_data import DataIngestorFactory


@step
def data_ingestion(file_path: str) -> pd.DataFrame:
    """Ingest data from a zip file using the appropriate Data ingester

    Args:
        file_path (str): the path of file where the zip file exists
    """
    data_ingestor = DataIngestorFactory.get_data_ingestor(".zip")

    df = data_ingestor.ingest_df(file_path)

    return df
