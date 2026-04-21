import joblib
import pandas as pd

from config import DATA_COLUMNS


def load_data(data: dict):
    missing = DATA_COLUMNS.copy()
    additional = []
    # Check if data is in the correct format
    for column in data.keys():
        if column not in DATA_COLUMNS:
            additional.append(column)
        else:
            missing.remove(column)
    
    if len(additional) > 0:
        print(f"The following data column was passed and it is not expected: {additional}. It will be ignored.")
    if len(missing) > 0 :
        print(f"The following data column is missing: {missing}, it might affect model performance.")

    df = pd.DataFrame(data)

    return df


def save_pipeline(model, path):
    joblib.dump(model, path)


def load_pipeline(path):
    return joblib.load(path)
