RANDOM_SEED = 42

# ------ PATHS --------
DS_PATH = "datasets/loan_approval_dataset.csv"
SAVE_PATH = "models/pipeline.pkl"
LOAD_PATH = "models/pipeline.pkl"


# ------ DATA COLUMNS --------
TARGET = "loan_status"
FEATURES = [
    "education",
    "self_employed",
    'no_of_dependents', 
    'income_annum', 
    'loan_amount', 
    'loan_term',
    'cibil_score', 
    'luxury_assets_value',
    'residential_assets_value', 
    'commercial_assets_value', 
    'bank_asset_value'
]
CATEGORICAL_FEATURES = [
    "education",
    "self_employed"
]
NUMERICAL_FEATURES = [
    'no_of_dependents', 
    'income_annum', 
    'loan_amount', 
    'loan_term',
    'cibil_score', 
    'luxury_assets_value', 
]
LOG_FEATURES = [
    'residential_assets_value', 
    'commercial_assets_value', 
    'bank_asset_value'
]


# ------ TRAINING CONFIG --------
TRAIN_RATIO = 0.7
MODELS = {
    "logistic": {
        "classifier__C": [0.01, 0.1, 1, 10]
    },
    "random_forest": {
        "classifier__n_estimators": [20, 50, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10]
    },
    "gradient_boosting": {
        "classifier__n_estimators": [20, 50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7, 10]
    }
}
