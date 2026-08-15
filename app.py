from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Credit Risk Prediction API")

model_pipeline = joblib.load("models/model_pipeline.pkl")
OPTIMAL_THRESHOLD = 0.62

class ApplicantData(BaseModel):
    age: int
    income: float
    credit_score: int
    employment_type: str

@app.post("/predict")
def predict_risk(data: ApplicantData):
    df = pd.DataFrame([data.model_dump()])
    
    probability = model_pipeline.predict_proba(df)[0][1]
    
    is_risky = bool(probability >= OPTIMAL_THRESHOLD)
    
    return {
        "probability_of_default": round(probability, 4),
        "reject_loan": is_risky,
        "message": "Loan Rejected - High Risk!" if is_risky else "Loan Approved - Low Risk!"
    }
