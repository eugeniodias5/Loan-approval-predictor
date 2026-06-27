from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

RANDOM_SEED = 42

# ------ PATHS --------
LOCAL_PATHS = False

# The following paths are used for local development. Otherwise, the paths are set as environment variables in the deployment environment.
DS_PATH = PROJECT_ROOT / "datasets" / "loan_approval_dataset.csv"
SAVE_PATH = PACKAGE_ROOT / "models" / "pipeline.pkl"
LOAD_PATH = PACKAGE_ROOT / "models" / "pipeline.pkl"

# ------ DATA COLUMNS --------
TARGET = "loan_status"
FEATURES = [
    "education",
    "self_employed",
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "luxury_assets_value",
    "residential_assets_value",
    "commercial_assets_value",
    "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
NUMERICAL_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "luxury_assets_value",
]
LOG_FEATURES = [
    "residential_assets_value",
    "commercial_assets_value",
    "bank_asset_value",
]


# ------ TRAINING CONFIG --------
TRAIN_RATIO = 0.7
MODELS = {
    "logistic": {"classifier__C": [0.01, 0.1, 1, 10]},
    "random_forest": {
        "classifier__n_estimators": [20, 50, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [20, 50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7, 10],
    },
}


if __name__ == "__main__":
    print(f"DS_PATH: {DS_PATH}")
    print(f"SAVE_PATH: {SAVE_PATH}")
    print(f"LOAD_PATH: {LOAD_PATH}")
