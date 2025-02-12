import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
from zenml import step


# Define an abstract class

class Ingest_Data(ABC):
    
    @abstractmethod
    def ingest_df(self,data_path : str)-> pd.DataFrame:
        
        """
        Abstract for ingesting data from data path
        """
        pass
    

class ZipDataIngester(Ingest_Data):
    
    def ingest_df(self,data_path:str)->pd.DataFrame:
        """Converts .zip file and returns data in pd.DataFrame format"""
        
        # Ensure data is in zip format
        if not data_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")
        
        # Extract the file
        with zipfile.ZipFile(data_path,"r") as zip_ref:
            zip_ref.extractall("extracted_data")
        

        # Find extracted CSV file (assuming there is CSV file in zip file)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found.Please specify which one to use.")
    

        # Read the CSV into the dataframe
        csv_file_path = os.path.join("extracted_data",csv_files[0])
        df = pd.read_csv(csv_file_path)
        
        # Return the dataframe
        return df
    
    
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension:str)->Ingest_Data:
        """Returns proper data ingestor based on extension"""
        if file_extension == ".zip":
            return ZipDataIngester()
        else:
            raise ValueError(f"No ingestor is present for this {file_extension} file extension.")
            
        
    
if __name__ == "__main__":
    
    data_path = "/home/himanshu/Coding/archive.zip"
    file_extension = os.path.splitext(data_path)[1]
    
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension) 
    
    df = data_ingestor.ingest_df(data_path)
    
    print(df.head())