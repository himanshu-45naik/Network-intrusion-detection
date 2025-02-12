from abc import ABC,abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    
    @abstractmethod
    def inspect(self,df:pd.DataFrame):
        
        """Perform basic inspection on data
        
        parameters : df: pd.dataframe
        return: No return only prints inspection details
        """
        pass
 
class DataTypesInspectionStrategy(DataInspectionStrategy):
    
    def inspect(self,df:pd.DataFrame):
        """Inspects and prints datatype and null value counts

        Args:
            df (pd.DataFrame): Data on which inspection is carried out
        """
        
        print("\nDatatypes and nulll value counts of the df: ")
        print(df.info())
        
               
## Statistics Inspection
class StatisticsInspectionStrategy(DataInspectionStrategy):
    
    def inspect(slef,df:pd.DataFrame):
        """Prints summarry of statistics for numerical and categorical value
        
        Args: df:pd.DataFrame
        
        Returns : None
        """
        
        print("\nSummary Statistics for Numerical Features: ")
        print(df.describe())
        print("\nSummary of categorical values: ")
        print(df.describe(include=["O"]))
        

        
        
## This class allows you to switch between different data inspection strategy
class DataInspector:
        
    def __init__(self,strategy: DataInspectionStrategy):    
        """
        Initializes Data Inspector with specific inspection methods
        
        Parameters: 
        strategy (DataInspectionStrategy): The strategy to be used for data inspection
        
        Returns : No returns
        """
        self._strategy = strategy
        
    def set_strategy(self,strategy:DataInspectionStrategy):
        """
        Sets a new strategy for Data inspector.
        
        Parameters:
        Strategy (DataInspectionStrategy): The new strategy to be implemented
        
        Returns : None
        """
        self._strategy = strategy

    def execute_inspection(self,df:pd.DataFrame):
        """Executes the inspection on the dataframe

        Args:
            df (pd.DataFrame): Dataset on which inspection is carried out
        """
        self._strategy.inspect(df)
        
        
if __name__ == "__main__":
    
    # Load the data
    df = pd.read_csv("/home/himanshu/Coding/House_Price/steps/extracted_data/AmesHousing.csv") 
    
    # Initialize the Data Inspector with specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)
    
    inspector.set_strategy(StatisticsInspectionStrategy())
    inspector.execute_inspection(df)
