import os
from pathlib import Path

from loan_approval_predictor.train import train_model

import mlflow

from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    
    with mlflow.start_run(run_name="loan-v1"):
        mlflow.set_tag("model_version", "v1")
        data_path = Path(__file__).parent.parent / "datasets" / "loan_approval_dataset.csv"
        train_model(data_path)
        
    