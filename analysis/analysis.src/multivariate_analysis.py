import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MultivariateAnalysisTemplate(ABC):
    def analyze(self,df:pd.DataFrame):
        """Performs Multivariate Analysis
        
        Args:
        df :pd.DataFrame - The data frame on which analysis is performed
        """
        
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        
    @abstractmethod
    def generate_correlation_heatmap(self,df:pd.DataFrame):
        """Generate heatmap of correlation between features

        Args:
            df (pd.DataFrame): DataFrame for which correlation analysis is performed
        """
        pass
        
    def generate_pairplot(self,df:pd.DataFrame):
        """Generate pairplot of given features

        Args:
            df (pd.DataFrame): Data on which anlysis is performed
        """
        pass
    
    
    
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df:pd.DataFrame):
        """Performs heatmap analysis on the data

        Args:
            df (pd.DataFrame): the dataframe on which analysis is performed
        """
        plt.figure(figsize=(8,7))
        sns.heatmap(df.corr(),annot=True,fmt=".2f",cmap="cool",cbar=True)
        plt.title("Coorelation Heatmap")
        plt.show()
        
    def generate_pairplot(self, df:pd.DataFrame):
        """Generate pairplot analysis of given features

        Args:
            df (pd.DataFrame): The dataframe containing the data
            
        Returns: 
        None
        """
        sns.pairplot(df)
        plt.suptitle("Pairplot of selected Features",y=1.02)
        plt.show()
        
if __name__ == "__main__":
    pass