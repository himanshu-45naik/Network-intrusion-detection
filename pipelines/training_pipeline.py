from zenml import Model, pipeline, step
import pandas as pd
import numpy as np
from steps.data_ingestion_step import data_ingestion
from steps.handle_duplicate_step import drop_duplicate_values
from steps.data_handling_step import handle_missing_values, handle_infinite_values
from steps.downcast_step import downcast
from steps.feature_engineering_step import feature_engineering
from steps.feature_extraction_step import feature_extraction


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
    
    ## Downcasting step.
    downcasted_df = downcast(filled_df)
    
    ## Dropping feature step.
    modified_df = feature_engineering(downcasted_df,"dropfeatures",features=None)
    
    ## Scaling step.
    features_df = modified_df.drop('Attack type',axis = 1)
    attacks_df = modified_df['Attack type']
    features = modified_df.columns
    scaled_df = feature_engineering(features_df,"standard",features)
    
    ## PCA step.
    extracted_df = feature_extraction(scaled_df)
    extracted_df["Attack Type"] = attacks_df.values
    
    ## SMOTE.
