import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC,abstractmethod

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df:pd.DataFrame,feature:str):
        """
        Performs univariate analysis on each feature of dataframe

        Args:
            df (pd.DataFrame): The dataframe containing the data
            feature (str) : The name of the feature to be analzed
            
        Returns:
        None
        """
        
        pass
    
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        """Performs univariate analysis on the numerical feature
        
        Args:
        df (pd.DataFrame) - The dataframe on which analysis is performed
        feature (str) - Numerical feature on which analysis is performed
        
        Returns:
        None
        """
        
        plt.figure(figsize=(8,4))
        sns.histplot(df[feature],kde=True,bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        
        plt.show()
        
class CategoricalUnivariateaAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        """Analyzes the categorical feature

        Args:
            df (pd.DataFrame): The dataframe on which the analysis is done
            feature (str): Categorical feature on which analysis is done
        """
        
        plt.figure(figsize=(8,4))
        sns.countplot(x=feature,data=df,palette="muted",hue=feature)
        plt.title((f"Distribtion of Categorical feature: {feature}"))
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
        
        
class UnivariateAnalyzer:
    """Implements a particular strategy"""
    
    def __init__(self,strategy:UnivariateAnalysisStrategy):
        self._strategy = strategy
        
    def set_strategy(self,strategy:UnivariateAnalysisStrategy):
        """Sets a new strategy"""
        self._strategy = strategy
    
    def execute_strategy(self,df:pd.DataFrame,feature:str):
        """Executes a particular analaysis on the feature
        
        Args:
        df (pd.DataFrame) - Dataframe on which anlaysis is carried out
        feature (str) - Feature of the df  
        """
        self._strategy.analyze(df,feature)
        
if __name__ == "__main__":
    pass