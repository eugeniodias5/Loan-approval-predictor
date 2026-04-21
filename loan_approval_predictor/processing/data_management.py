import pandas as pd


DATA_COLUMNS = [
               "no_of_dependents",
               "education",
                "self_employed",
                "income_annum",
                "loan_amount",
                "loan_term",
                "cibil_score",
                "residential_assets_value",
                "commercial_assets_value",
                "luxury_assets_value",
                "bank_asset_value",
                "loan_status"
]



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

    return data
