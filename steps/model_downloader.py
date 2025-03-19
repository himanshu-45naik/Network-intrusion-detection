import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from zenml import step

load_dotenv(dotenv_path="config/.env")
MODEL_DOWNLOAD_PATH = os.getenv("MODEL_DOWNLOAD_PATH")

@step
def download_model_from_mlflow(run_id, model_name, output_dir="/home/himanshu/Network-intrusion-detection/saved_models"):
    """
    Download a trained model from MLflow to the saved_model directory.
    
    Args:
        run_id (str): The MLflow run ID from which to download the model
        model_name (str): The name of the model as registered in MLflow
        output_dir (str): Local directory where the model will be saved
        
    Returns:
        str: Path to the downloaded model
    """
    # Set up the saved_model directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_dir = os.path.join(output_dir, f"{model_name}_{run_id}")
    
    client = MlflowClient()
    
    print(f"Downloading model '{model_name}' from run ID: {run_id}")
    client.download_artifacts(run_id, model_name, model_dir)
    
    model_path = os.path.join(model_dir, model_name)
    print(f"Model downloaded successfully to: {model_path}")
    
    return model_path


def load_downloaded_model(model_path):
    """
    Load the downloaded model for inference.
    
    Args:
        model_path (str): Path to the downloaded model
        
    Returns:
        object: The loaded model
    """
    # Load the model
    model = mlflow.sklearn.load_model(model_path)
    print(f"Model loaded successfully from: {model_path}")
    
    return model


if __name__ == "__main__":
    pass
    