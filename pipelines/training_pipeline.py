from zenml import Model, pipeline
from dotenv import load_dotenv
import yaml
import os
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
from steps.mlflow_tracking_step import mlflow_tracker

load_dotenv(dotenv_path="./config/.env")

DATA_PATH = os.getenv("DATA_PATH")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")


@pipeline(model=Model(name="intrusion_detection"))
def ml_pipeline():
    """Defines end-to-end machine learning pipeline with MLflow tracking."""
    # Data Ingestion step
    raw_df = data_ingestion(DATA_PATH)

    # Updating Feature name step
    updated_raw_df = feature_name_handling(raw_df)

    # Drop duplicate values step
    transformed_df = drop_duplicate_values(updated_raw_df)

    # Replacing infinity values with Nan step
    inf_replaced_data = handle_infinite_values(transformed_df)

    # Filling missing values step
    filled_df = handle_missing_data(inf_replaced_data, "median")

    # Downcasting step
    downcasted_df = downcast(filled_df)

    # Dropping feature step
    modified_df = feature_engineering(downcasted_df, "dropfeatures")

    # Scaling step
    scaled_df = feature_engineering(modified_df, "standard")

    # PCA step
    extracted_df = feature_extraction(scaled_df)

    # Label encoding for binary classification step
    binary_df = feature_engineering(extracted_df, "binaryencoding")
    multiclass_df = feature_engineering(extracted_df, "multiclassencoding")

    # Data splitting step
    X_train_binary, X_test_binary, y_train_bianry, y_test_binary = data_splitter_step(binary_df, "Attack Type")
    X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = data_splitter_step(multiclass_df, "Attack Type")
    
    # SMOTE step
    X_train_binary, y_train_binary = sampling_data(X_train_binary, y_train_bianry)
    X_train_multiclass, y_train_multiclass = sampling_data(X_train_multiclass, y_train_multiclass)
    
    # Model building step (logistic regression)
    lr_binary_model = model_building(
        X_train_binary, y_train_binary, "logisticregression"
    )

    # Log logistic regression model to MLflow
    lr_bi_run_id = mlflow_tracker(
        model=lr_binary_model,
        model_name="logisticregression",
        X_test=X_test_binary,
        y_test=y_test_binary,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    # Model building step (rf)
    rf_binary_model = model_building(X_train_binary, y_train_binary, "rf_binary")
    
    # Log rf model to MLflow
    rf_bi_run_id = mlflow_tracker(
        model=rf_binary_model,
        model_name="rf_binary",
        X_test=X_test_binary,
        y_test=y_test_binary,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME
    )
    
    # Model building step (rf multiclass classification)
    rf_multiclass_model = model_building(X_train_multiclass,y_train_multiclass,"rf_multiclass")
    
    # Log  rf model to mlflow
    rf_bi_run_id = mlflow_tracker(
        model=rf_multiclass_model,
        model_name="rf_multiclas",
        X_test=X_test_multiclass,
        y_test=y_test_multiclass,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME
    )
    # # Model building step (xgboost binary classfication)
    # xgb_binary_model = model_building(X_train_binary, y_train_binary, "xgb_binary")
    
    # # Log xgb binary model to MLflow
    # xgb_bi_run_id = mlflow_tracker(
    #     model=xgb_binary_model,
    #     model_name="xgb_binary",
    #     X_test=X_test_binary,
    #     y_test=y_test_binary,
    #     tracking_uri=MLFLOW_TRACKING_URI,
    #     experiment_name=MLFLOW_EXPERIMENT_NAME
    # )
    
    # # Model building step (xgboost multiclass classification)
    # xgb_multiclass_model = model_building(X_train_multiclass, y_train_multiclass, "xgb_multiclass")
    
    # # Log xgb multiclass model to mlflow
    # xgb_multi_run_id = mlflow_tracker(model = xgb_multiclass_model,
    #     model_name = "xgb_multiclass",
    #     X_test=X_test_multiclass,
    #     y_test=y_test_multiclass,
    #     tracking_uri=MLFLOW_TRACKING_URI,
    #     experiment_name=MLFLOW_EXPERIMENT_NAME
    #   ) 
    
    # Model building step (lighgbm bianry classification)
    lgb_binary_model = model_building(X_train_binary,y_train_bianry,"lgbm_binary")
    
    # Log lightgbm model to mlflow
    lgbm_bi_run_id = mlflow_tracker(
        model=lgb_binary_model,
        model_name="lgbm_binary",
        X_test=X_test_binary,
        y_test=y_test_binary,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME
    )
    
    # Model building step (xgboost multiclass classification)
    lgbm_multiclass_model = model_building(X_train_multiclass, y_train_multiclass, "lgbm_multiclass")
    
    # Log lgbm multiclass model to mlflow
    lgbm_multi_run_id = mlflow_tracker(model = lgbm_multiclass_model,
        model_name = "lgbm_multiclass",
        X_test=X_test_multiclass,
        y_test=y_test_multiclass,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME
        ) 
    
    # Model Building step (ocSVM)
    # ocsvm_model = model_building(binary_df, y_train_bianry, "ocsvm")

    # # Log oc_SVM model to MLflow
    # oc_svm_run_id = mlflow_tracker(
    #     model=ocsvm_model,
    #     model_name="oc-svm",
    #     X_test=X_test_binary,
    #     y_test=y_test_binary,
    #     tracking_uri=MLFLOW_TRACKING_URI,
    #     experiment_name=MLFLOW_EXPERIMENT_NAME,
    # )



if __name__ == "__main__":
    pass
