from pathlib import Path

from loan_approval_predictor.train import train_model

import mlflow


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    exp_id = mlflow.create_experiment("loan_approval_predictor")
    mlflow.set_experiment("loan_approval_predictor")

    with mlflow.start_run(run_name="loan-v1"):
        mlflow.set_tag("model_version")
        data_path = Path(__file__).parent.parent / "datasets" / "loan_approval_dataset.csv"
        train_model(data_path)
        
    