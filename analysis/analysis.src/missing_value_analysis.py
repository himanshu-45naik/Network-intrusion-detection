import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# Class defining interface
class MissingValueTemplate(ABC):
    def analyze(self,df:pd.DataFrame):
        """Analyzing the missing values and visualizing it

        Args:
            df (pd.DataFrame): df for which missing value is to be analyzed
        
        Return: None
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
        
        @abstractmethod
        def identify_missing_values(self,df:pd.DataFrame):
            """
            Identifies missing value in the dataframe
            
            Parameters:
            df:pd.DataFrame : Identifies missing values for this dataset
            
            Returns : None
            """
            pass
        
        @abstractmethod
        def visualize_missing_values(self,df:pd.DataFrame):
            """Visualizes the missing values of the dataframe

            Args:
                df (pd.DataFrame): Visulaizes the missing values for this dataframe
                
            Returns: None
            """
            pass
        
class ImplementMissingValueAnalysis(MissingValueTemplate):
    
        
    def identify_missing_values(self, df:pd.DataFrame):
        """
        Prints amount of missing values in each columns

        Args:
            df (pd.DataFrame):df for which missing value is identified
        
        Returns:
        None
        """
        
        print("\nMissing values count by columns: ")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])
        
        
    def visualize_missing_values(self,df:pd.DataFrame):
        
        """
        Create heatmap to visualize missing values of each columns
        
        Parameters:
        df:pd.DataFrame : Dataframe on which visualization is done for missing values
        """
        print("\nVisualizing Missing values....")
        plt.figure(figsize=(10,12))
        sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
        
        
if __name__ == "__main__":
    
    df = pd.read_csv("/home/himanshu/Coding/House_Price/steps/extracted_data/AmesHousing.csv") 
    missing_value_analyzer = ImplementMissingValueAnalysis()
    
    missing_value_analyzer.analyze(df)
    
