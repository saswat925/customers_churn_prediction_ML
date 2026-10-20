
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# Instantiate API application
app = FastAPI(
    title="Telecom Customer Churn Prediction Service",
    description="Production-ready endpoint to compute active user churn risk parameters.",
    version="1.0.0"
)

# Load pre-trained model pipeline
try:
    model = joblib.load('models/telecom_churn_model.joblib')
except Exception as e:
    raise RuntimeException(f"Unable to load prediction model from disk: {e}")

# Define structured client payload contract
class CustomerPayload(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer's age")
    num_dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    estimated_salary: float = Field(..., ge=0, description="Annual estimated income")
    calls_made: int = Field(..., ge=0, description="Total voice calls initiated")
    sms_sent: int = Field(..., ge=0, description="Total SMS text packets routed")
    data_used: float = Field(..., ge=0.0, description="Internet payload consumption in MBs")
    tenure_days: int = Field(..., ge=0, description="Registration longevity tenure in days")
    telecom_partner_BSNL: bool = False
    telecom_partner_Reliance_Jio: bool = Field(False, alias="telecom_partner_Reliance Jio")
    telecom_partner_Vodafone: bool = False
    gender_M: bool = Field(False, description="True if Male, False if Female")

    class Config:
        populate_by_name = True

@app.get("/health", tags=["Utility"])
def health_check():
    # Checks service running capability status.
    return {"status": "Operational", "model_loaded": model is not None}

@app.post("/predict", tags=["ML Engine"])
def predict_churn(customer: CustomerPayload):
    # Analyzes single customer payload to determine probability risk category.
    try:
        # Convert schema model back to Pandas Dataframe matches X_train signature
        input_dict = {
            'age': customer.age,
            'num_dependents': customer.num_dependents,
            'estimated_salary': customer.estimated_salary,
            'calls_made': customer.calls_made,
            'sms_sent': customer.sms_sent,
            'data_used': customer.data_used,
            'tenure_days': customer.tenure_days,
            'telecom_partner_BSNL': customer.telecom_partner_BSNL,
            'telecom_partner_Reliance Jio': customer.telecom_partner_Reliance_Jio,
            'telecom_partner_Vodafone': customer.telecom_partner_Vodafone,
            'gender_M': customer.gender_M
        }

        df_input = pd.DataFrame([input_dict])

        # Execute predictions pipeline
        prob = float(model.predict_proba(df_input)[0, 1])
        label = int(prob >= 0.5)

        # Categorize risk levels
        if prob >= 0.7:
            risk = "High"
        elif prob >= 0.3:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_probability": round(prob, 2),
            "predicted_churn": label,
            "risk_category": risk
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal inference exception: {err}")
