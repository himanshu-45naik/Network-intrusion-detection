from zenml import Model, pipeline, step
import pandas as pd
import numpy as np
from steps.data_ingestion_step import data_ingestion
from steps.handle_duplicate_step import drop_duplicate_values
from steps.data_handling_step import (
    handle_missing_data,
    handle_infinite_values,
    feature_name_handling,
)
from steps.downcast_step import downcast
from steps.feature_engineering_step import feature_engineering
from steps.feature_extraction_step import feature_extraction
from steps.data_splitting_step import data_splitter_step
from steps.oversampling_data_step import sampling_data
from steps.model_building_step import model_building
from steps.model_evaluation_step import model_evaluation 


@pipeline(model=Model(name="intrusion_predictor"))
def ml_pipeline():
    """Defines end-to-end machine learning pipeline."""

    ## Data Ingestion step.
    raw_df = data_ingestion("/home/himanshu/Coding/Network Intrusion/cicids17.zip")

    ## Updating Feature name step
    updated_raw_df = feature_name_handling(raw_df)
    ## Drop duplicate values step.
    transformed_df = drop_duplicate_values(updated_raw_df)

    ## Replacing infinity values with Nan step.
    inf_replaced_data = handle_infinite_values(transformed_df)

    ## Filling missing values step.

    filled_df = handle_missing_data(inf_replaced_data, "median")

    ## Downcasting step.
    downcasted_df = downcast(filled_df)

    ## Dropping feature step.
    modified_df = feature_engineering(downcasted_df, "dropfeatures")

    ## Scaling step.
    scaled_df = feature_engineering(
        modified_df, "standard"
    )  ## This do not contain the attack type feature

    ## PCA step.
    extracted_df = feature_extraction(scaled_df)
    
    ## Label encoding for binary classification step
    encoded_df = feature_engineering(extracted_df, "binaryencoding")
    
    ## Data splitting step.
    X_train, X_test, y_train, y_test = data_splitter_step(encoded_df, "Attack Type")
    
    ## SMOTE step.
    X_train_resampled, y_train_resampled = sampling_data(X_train, y_train)

    ## Model building step
    lr_model = model_building(X_train_resampled, y_train_resampled, "logisticregression")
    
    ## Model evaluation step
    lr_evaluation = model_evaluation(X_test, y_test, lr_model, "logisticregression")
    
if __name__ == "__main__":
    pass
