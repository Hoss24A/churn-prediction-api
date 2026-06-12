from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_churn

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability with SHAP explanations",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    top_factors: list

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost Churn Classifier v1.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        result = predict_churn(customer.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))