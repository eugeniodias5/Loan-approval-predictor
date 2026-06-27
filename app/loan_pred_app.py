from typing import List, Literal

from fastapi import FastAPI
import uvicorn
import pandas as pd
from pydantic import BaseModel

from loan_approval_predictor.processing.data_management import load_pipeline

from dotenv import load_dotenv
load_dotenv()  # Load environment variables

app = FastAPI()

# Load the pipeline once at startup and reuse it for every request,
# instead of reading and deserializing the .pkl on each prediction.
pipeline = load_pipeline()


class LoanData(BaseModel):
    no_of_dependents: int
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.get("/")
def index():
    return {"message": "Welcome to the Loan Approval Predictor API!. To get a prediction, send a POST request to /predict with the required data."}


@app.post("/predict")
def get_prediction(data: LoanData):
    loan_input = data.dict()

    predictions = pipeline.predict(pd.DataFrame([loan_input]))

    return {"loan_status": predictions[0]}


@app.post("/predict-batch")
def get_batch_prediction(data: List[LoanData]):
    loan_inputs = [item.dict() for item in data]

    predictions = pipeline.predict(pd.DataFrame(loan_inputs))

    return {"loan_status": list(predictions)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)