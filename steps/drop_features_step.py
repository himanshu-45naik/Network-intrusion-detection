from zenml import step
import pandas as pd
from src.dropfeatures import DropOneValueFeature


@step
def drop_feature_step(df:pd.DataFrame,features:list)->pd.DataFrame:
    
    dropper = (DropOneValueFeature())
    num_unique = df.nunique()
    updated_df = dropper.drop_features(df, num_unique)
    return updated_df