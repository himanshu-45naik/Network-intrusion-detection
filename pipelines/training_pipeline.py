from zenml import Model, pipeline, step
import pandas as pd
import numpy as np
from steps.data_ingestion_step import data_ingestion
from steps.handle_duplicate_step import drop_duplicate_values
from steps.data_handling_step import handle_missing_values, handle_infinite_values


@pipeline(model=Model(name="intrusion_predictor"))
def ml_pipeline():
    """Defines end-to-end machine learning pipeline."""

    ## Data Ingestion step.
    raw_df = data_ingestion("/home/himanshu/Coding/Network Intrusion/archive.zip")

    ## Drop duplicate values step.
    transformed_df = drop_duplicate_values(raw_df)

    ## Replacing infinity values with Nan step.
    numeric_df = transformed_df.select_dtypes(["number"])
    inf_features = [
        col for col in numeric_df.columns if np.isinf(transformed_df[col]).any()
    ]
    inf_replaced_data = handle_infinite_values(transformed_df, inf_features)

    ## Filling missing values step.
    missing_features = [f for f in numeric_df.columns if np.isnan(numeric_df[f]).any()]
    filled_df = handle_missing_values(inf_replaced_data, missing_features, "median")
    
    ## Downcast
    
    ## Scaling
    
    ## PCA
    
    ## SMOTE
