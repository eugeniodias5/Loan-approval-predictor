import os

import joblib
import pandas as pd

from config.config import FEATURES, DS_PATH, SAVE_PATH, LOAD_PATH, TARGET


def load_data(path = DS_PATH):
    df = pd.read_csv(path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    missing = FEATURES.copy()
    additional = []
    # Check if data is in the correct format
    for column in X.columns:
        if column not in FEATURES:
            additional.append(column)
        else:
            missing.remove(column)

    if len(additional) > 0:
        print(f"The following data column was passed and it is not expected: {additional}. It will be ignored.")
    if len(missing) > 0 :
        print(f"The following data column is missing: {missing}, it might affect model performance.")

    # Remove additional columns
    X = X.drop(columns=additional, inplace=False)

    return X, y


def save_pipeline(model, path=SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_pipeline(path=LOAD_PATH):
    return joblib.load(path)
