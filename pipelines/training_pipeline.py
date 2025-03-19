from zenml import Model, pipeline
from dotenv import load_dotenv
import yaml
import os
from steps.data_ingestion_step import data_ingestion
from steps.data_handling_step import handling_data
from steps.feature_engineering_step import feature_engineering
from steps.feature_extraction_step import feature_extraction
from steps.data_splitting_step import data_splitting
from steps.drop_features_step import drop_feature
from steps.oversampling_data_step import over_sampling_data
from steps.model_building_step import model_building
from steps.mlflow_tracking_step import mlflow_tracker
from steps.model_downloader import download_model_from_mlflow
from steps.sample_data_step import sampling_data

load_dotenv(dotenv_path="config/.env")

DATA_PATH = os.getenv("DATA_PATH")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URL")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")


@pipeline(model=Model(name="intrusion_detection"))
def ml_pipeline():
    """Defines end-to-end machine learning pipeline with MLflow tracking."""


    # Data Ingestion step
    raw_df = data_ingestion(DATA_PATH)
    
    # Data Handling step.
    processed_df = handling_data(raw_df,"mean",fill_value=None)
    # Sample Dataset
    sampled_df = sampling_data(processed_df, sample_size=0.05, random_state=42)

    # Dropping feature step
    modified_df = drop_feature(sampled_df)

    # Data splitting step
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = data_splitting(
        modified_df, "Attack Type"
    )
    X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = (
        data_splitting(modified_df, "Attack Type")
    )

    # Label encoding for binary classification step
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = feature_engineering(
        X_train_binary, X_test_binary, y_train_binary, y_test_binary, "binaryencoding"
    )
    X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = feature_engineering(
        X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass, "multiclassencoding"
    )

    # SMOTE step
    X_train_binary, y_train_binary = over_sampling_data(X_train_binary, y_train_binary)
    X_train_multiclass, y_train_multiclass = over_sampling_data(
        X_train_multiclass, y_train_multiclass
    )

    # Scaling step
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = feature_engineering(X_train_binary, X_test_binary, y_train_binary, y_test_binary, "standard")
    X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = feature_engineering(
        X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass, "standard"
    )

    # PCA step
    X_train_binary , X_test_binary = feature_extraction(X_train_binary, X_test_binary)
    X_train_multiclass, X_test_multiclass = feature_extraction(X_train_multiclass, X_test_multiclass)

    
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
    lr_model_path = download_model_from_mlflow(lr_bi_run_id, "logisticregression")
    print(f"Model saved to: {lr_model_path}")

    # Model building step (rf)
    rf_binary_model = model_building(X_train_binary, y_train_binary, "rf_binary")

    # Log rf model to MLflow
    rf_bi_run_id = mlflow_tracker(
        model=rf_binary_model,
        model_name="rf_binary",
        X_test=X_test_binary,
        y_test=y_test_binary,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    rf_bi_model_path = download_model_from_mlflow(rf_bi_run_id, "rf_binary")
    print(f"Model saved to: {rf_bi_model_path}")

    # Model building step (rf multiclass classification)
    rf_multiclass_model = model_building(
        X_train_multiclass, y_train_multiclass, "rf_multiclass"
    )

    # Log  rf model to mlflow
    rf_multi_run_id = mlflow_tracker(
        model=rf_multiclass_model,
        model_name="rf_multiclass",
        X_test=X_test_multiclass,
        y_test=y_test_multiclass,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    rf_multi_model_path = download_model_from_mlflow(rf_multi_run_id, "rf_multiclass")
    print(f"Model saved to: {rf_bi_model_path}")

    # Model building step (xgboost binary classfication)
    xgb_binary_model = model_building(X_train_binary, y_train_binary, "xgb_binary")

    # Log xgb binary model to MLflow
    xgb_bi_run_id = mlflow_tracker(
        model=xgb_binary_model,
        model_name="xgb_binary",
        X_test=X_test_binary,
        y_test=y_test_binary,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    xgb_bi_model_path = download_model_from_mlflow(xgb_bi_run_id, "xgb_binary")
    print(f"Model saved to: {xgb_bi_model_path}")

    # Model building step (xgboost multiclass classification)
    xgb_multiclass_model = model_building(
        X_train_multiclass, y_train_multiclass, "xgb_multiclass"
    )

    # Log xgb multiclass model to mlflow
    xgb_multi_run_id = mlflow_tracker(
        model=xgb_multiclass_model,
        model_name="xgb_multiclass",
        X_test=X_test_multiclass,
        y_test=y_test_multiclass,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    xgb_multi_model_path = download_model_from_mlflow(
        xgb_multi_run_id, "xgb_multiclass"
    )
    print(f"Model saved to: {xgb_multi_model_path}")

    # Model building step (lighgbm bianry classification)
    lgbm_binary_model = model_building(X_train_binary, y_train_binary, "lgbm_binary")

    # Log lightgbm model to mlflow
    lgbm_bi_run_id = mlflow_tracker(
        model=lgbm_binary_model,
        model_name="lgbm_binary",
        X_test=X_test_binary,
        y_test=y_test_binary,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    lgbm_bi_model_path = download_model_from_mlflow(lgbm_bi_run_id, "lgbm_binary")
    print(f"Model saved to: {lgbm_bi_model_path}")
    
    # Model building step (xgboost multiclass classification)
    lgbm_multiclass_model = model_building(
        X_train_multiclass, y_train_multiclass, "lgbm_multiclass"
    )
    # Log lgbm multiclass model to mlflow
    lgbm_multi_run_id = mlflow_tracker(
        model=lgbm_multiclass_model,
        model_name="lgbm_multiclass",
        X_test=X_test_multiclass,
        y_test=y_test_multiclass,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    lgbm_multi_model_path = download_model_from_mlflow(
        lgbm_multi_run_id, "lgbm_multiclass"
    )
    print(f"Model saved to: {lgbm_multi_model_path}")
    
    # oc_svm model building step
    oc_svm = model_building(X_train_multiclass, y_train_multiclass, "oc_svm")
    
    # Log oc_svm model to mlflow
    oc_svm_run_id = mlflow_tracker(
        model=oc_svm,
        model_name="oc_svm",
        X_test=X_test_multiclass,
        y_test=y_test_multiclass,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    oc_svm_model_path = download_model_from_mlflow(
        oc_svm_run_id, "oc_svm"
    )
    print(f"Model saved to: {oc_svm_model_path}")
    
if __name__ == "__main__":
    pass