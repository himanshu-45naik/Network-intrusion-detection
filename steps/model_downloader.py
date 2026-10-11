import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from zenml import step

load_dotenv(dotenv_path="config/.env")
MODEL_DOWNLOAD_PATH = os.getenv("MODEL_DOWNLOAD_PATH")

# Anchored to the repo root (this file lives in <root>/steps/) rather than to an
# absolute path or the caller's working directory, so downloads land in the same
# place on any machine and from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@step
def download_model_from_mlflow(run_id, model_name, output_dir="saved_models"):
    """
    Download a trained model from MLflow to the saved_model directory.

    Args:
        run_id (str): The MLflow run ID from which to download the model
        model_name (str): The name of the model as registered in MLflow
        output_dir (str): Directory where the model will be saved. A relative
            path is resolved against the repository root; an absolute path is
            used as given.

    Returns:
        str: Path to the downloaded model
    """
    # Set up the saved_model directory
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    model_dir = output_path / f"{model_name}_{run_id}"

    client = MlflowClient()

    print(f"Downloading model '{model_name}' from run ID: {run_id}")
    client.download_artifacts(run_id, model_name, str(model_dir))

    model_path = model_dir / model_name
    print(f"Model downloaded successfully to: {model_path}")

    return str(model_path)


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
