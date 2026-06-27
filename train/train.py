import os
from pathlib import Path

from loan_approval_predictor.train import train_model

import mlflow

from dotenv import load_dotenv
load_dotenv()  # Load environment variables

if __name__ == "__main__":
    tracking_uri = os.environ.get("ML_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        
    with mlflow.start_run(run_name="loan-v1"):
        mlflow.set_tag("model_version", "v1")
        data_path = Path(__file__).parent.parent / "datasets" / "loan_approval_dataset.csv"
        train_model(data_path)
        
    