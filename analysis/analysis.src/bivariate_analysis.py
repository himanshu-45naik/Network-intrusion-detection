from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df:pd.DataFrame,feature1:str,feature2:str):
        """Perform Bivariate analysis on the dataframe
        
        Args:
        df(pd.dataframe) - The dataframe on which the analysis is performed
        feature1 (str) - Feature1 of the df
        feature2 (str) - Feature2 of the df
        """
        
        pass
    
class NumericalvsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature1:str, feature2:str):
        """Performs bivariate analysis on the the numerical features 

        Args:
            df : The dataframe on which analysis is performed
            feature1 : Numerical Feature of the df 
            feature2 : Numerical Feature of the df
        """
        
        plt.figure(figsize=(7,6))
        sns.scatterplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        
class CategoricalvsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature1:str, feature2:str):
        """Performs data analysis on categorical and numerical features

        Args:
            df (pd.DataFrame): df on which analysis is performed
            feature1 (str): Categorical feature
            feature2 (str): Numerical feature
        """
        plt.figure(figsize=(7,6))
        sns.boxplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()
        
        
class BivariateAnalyzer:
    """Performs the bivariate Analysis for the given strategy""" 
    
    def __init__(self,strategy:BivariateAnalysisStrategy):
        """Initializes the Bivariate Analysis Strategy

        Args:
            strategy (BivariateAnalysisStrategy): the strategy for which the analysis is to be carried out
        """

        self._strategy = strategy
        
    def set_strategy(self,strategy:BivariateAnalysisStrategy):
        """Sets the strategy for which the analysis is to be carried out

        Args:
            strategy (BivariateAnalysisStrategy): The analsis is carried based on given strategy
        """
    
        self.set_strategy = strategy
        
    def execute_strategy(self,df,feature1,feature2):
        """It implements the analysis based on the strategy

        Args:
           df : dataframe on which analysis is performed
           feature1 
           feature2
        """
        
        self._strategy.analyze(df,feature1,feature2)
        
if __name__=="__main__":
    pass
        