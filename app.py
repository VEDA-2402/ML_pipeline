from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Sales Prediction API")


class PredictRequest(BaseModel):
    Store: int = Field(..., ge=2, le=1115)
    DayOfWeek: int = Field(..., ge=1, le=7)
    Customers: int = Field(..., ge=0, le=4582)

    Promo: int = Field(..., ge=0, le=1)
    SchoolHoliday: int = Field(..., ge=0, le=1)

    CompetitionDistance: float = Field(..., ge=20, le=27190)
    CompetitionOpenSinceMonth: int = Field(..., ge=0, le=12)
    CompetitionOpenSinceYear: int = Field(..., ge=0, le=2025)  # adjust if needed

    Promo2SinceWeek: int = Field(..., ge=0, le=52)
    Promo2SinceYear: int = Field(..., ge=0, le=2015)

    Month: int = Field(..., ge=1, le=12)
    Day: int = Field(..., ge=1, le=31)
    WeekOfYear: int = Field(..., ge=1, le=52)
    IsWeekend: int = Field(..., ge=0, le=1)

    Sales_lag1: float = Field(..., ge=0, le=41551)
    Sales_roll7: float = Field(..., ge=1276, le=29967)
    Sales_roll30: float = Field(..., ge=1417, le=22213)


@app.post("/predict")
def predict(payload: PredictRequest):
    # Hidden features auto-filled
    hidden_defaults = {
        "Open": 1,
        "Promo2": 0,
        "Sales_lag7": 0,
        "Sales_lag14": 0,
        "Year": 2015,
        "StateHoliday": 0
    }

    # Build the full feature vector in the exact training order
    features = [
        payload.Sales_roll30,
        payload.Sales_roll7,
        0,                  # Sales_lag14 (removed)
        0,                  # Sales_lag7 (removed)
        payload.Sales_lag1,
        payload.IsWeekend,
        payload.WeekOfYear,
        payload.Day,
        payload.Month,
        2015,               # Year default
        payload.Promo2SinceYear,
        payload.Promo2SinceWeek,
        0,                  # Promo2 (removed)
        payload.CompetitionOpenSinceYear,
        payload.CompetitionOpenSinceMonth,
        payload.CompetitionDistance,
        payload.SchoolHoliday,
        0,                  # StateHoliday default
        payload.Promo,
        1,                  # Open default
        payload.Customers,
        payload.DayOfWeek,
        payload.Store
    ]

    X = np.array([features])
    prediction = float(model.predict(X)[0])
    return {"prediction": prediction}

