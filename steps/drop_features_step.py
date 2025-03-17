from zenml import step
import pandas as pd
from src.dropfeatures import DropOneValueFeature


@step
def drop_feature(df:pd.DataFrame)->pd.DataFrame:
    
    dropper = (DropOneValueFeature())
    num_unique = df.nunique()
    updated_df = dropper.drop_features(df, num_unique)
    print(updated_df.columns)
    return updated_df