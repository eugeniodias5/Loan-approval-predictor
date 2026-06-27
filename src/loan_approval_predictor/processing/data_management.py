import os

import joblib
import pandas as pd

import mlflow

from loan_approval_predictor.config.config import (
    FEATURES,
    DS_PATH,
    SAVE_PATH,
    LOAD_PATH,
    TARGET,
)


def load_data(path=None):
    local_paths = os.environ.get("LOCAL_PATHS", "true").lower() == "true"
    if path is None:
        if local_paths:
            path = DS_PATH
    else:
        path = os.environ.get("DS_PATH", None)
    
    # Check if the path is valid
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found. Please check the path: {path}. "
            "You can set the path using the DS_PATH environment variable, "
            "or set LOCAL_PATHS to True."
        )
    
    df = pd.read_csv(path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    has_target = TARGET in df.columns
    X = df.drop(columns=[TARGET]) if has_target else df.copy()
    y = df[TARGET] if has_target else None

    missing = FEATURES.copy()
    additional = []
    # Check if data is in the correct format
    for column in X.columns:
        if column not in FEATURES:
            additional.append(column)
        else:
            missing.remove(column)

    if len(additional) > 0:
        print(
            f"The following data column was passed and it is not expected: {additional}. It will be ignored."
        )
    if len(missing) > 0:
        print(
            f"The following data column is missing: {missing}, it might affect model performance."
        )

    # Remove additional columns
    X = X.drop(columns=additional, inplace=False)

    if has_target:
        return X, y
    return X


def save_pipeline(model):
    LOCAL_PATHS = os.environ.get("LOCAL_PATHS", "true").lower() == "true"
    path = SAVE_PATH
    if not LOCAL_PATHS:
        path = os.environ.get("SAVE_PATH", None)
    
    if path is None or not os.path.exists(os.path.dirname(path)):
        raise ValueError(
            "SAVE_PATH environment variable is not set or does not exist. " \
            "Please set it to the desired path for saving the model "
            "or set LOCAL_PATHS to True."
        )
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    joblib.dump(model, path)
    
    mlflow.sklearn.log_model(
        model, 
        "loan_approval_model",
        code_paths=["src/loan_approval_predictor"])


def load_pipeline():
    path = LOAD_PATH
    LOCAL_PATHS = os.environ.get("LOCAL_PATHS", "true").lower() == "true"
    if not LOCAL_PATHS:
        path = os.environ.get("LOAD_PATH", None)
    
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline file not found. Please check the path: {path}. "
            "You can set the path using the LOAD_PATH environment variable, "
            "or set LOCAL_PATHS to True."
        )
    
    return joblib.load(path)
