from pathlib import Path

from loan_approval_predictor.train import train_model

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "datasets" / "loan_approval_dataset.csv"
    train_model(data_path)
    