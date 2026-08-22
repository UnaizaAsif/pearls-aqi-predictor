# FastAPI backend for Pearls AQI Predictor
# Run with: uvicorn app.api:app --reload --port 8000

import os
import json
import tempfile

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

os.makedirs("C:\\tmp", exist_ok=True)
tempfile.tempdir = "C:\\tmp"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="REST API for AQI forecasting across Pakistani cities",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITIES = ["karachi", "lahore", "islamabad", "peshawar", "hyderabad"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "training_pipeline", "model")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache", "latest.json")

# ---------------------------------------------------------------------------
# Model loading (lazy, cached in module scope)
# ---------------------------------------------------------------------------

_model = None
_encoder = None
_feature_cols = None


def get_model():
    global _model, _encoder, _feature_cols
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "aqi_model.pkl")
        encoder_path = os.path.join(MODEL_DIR, "city_encoder.pkl")
        feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Run the training pipeline first."
            )
        _model = joblib.load(model_path)
        _encoder = joblib.load(encoder_path)
        _feature_cols = joblib.load(feature_path)
    return _model, _encoder, _feature_cols


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    city: str
    hour: int
    day_of_week: int
    month: int
    pm25: float
    temperature: float
    humidity: float
    wind: float
    pressure: float
    aqi_lag1: float = 0.0
    aqi_change_rate: float = 0.0


class PredictResponse(BaseModel):
    city: str
    predicted_aqi: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/cities")
def cities():
    """Return the list of supported cities."""
    return {"cities": CITIES}


@app.get("/aqi/{city}")
def get_aqi(city: str):
    """Return the latest AQI data for a city from the local JSON cache."""
    city = city.lower()
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found. Valid cities: {CITIES}")

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if city in cache:
                return cache[city]
        except (json.JSONDecodeError, KeyError):
            pass

    # Placeholder if cache is not available yet
    return {
        "city": city,
        "aqi": None,
        "pm25": None,
        "temperature": None,
        "humidity": None,
        "timestamp": None,
        "note": "Cache not yet populated. Run the feature pipeline to generate data.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict AQI for a city given feature inputs."""
    city = req.city.lower()
    if city not in CITIES:
        raise HTTPException(status_code=422, detail=f"City '{city}' not supported. Valid: {CITIES}")

    try:
        model, encoder, feature_cols = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        city_encoded = int(encoder.transform([city])[0])
    except Exception:
        raise HTTPException(status_code=422, detail=f"City '{city}' not recognized by encoder.")

    row = {
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "month": req.month,
        "pm25": req.pm25,
        "temperature": req.temperature,
        "humidity": req.humidity,
        "wind": req.wind,
        "pressure": req.pressure,
        "city_encoded": city_encoded,
        "aqi_lag1": req.aqi_lag1,
        "aqi_change_rate": req.aqi_change_rate,
    }

    X = pd.DataFrame([row])
    # Align columns to what the model expects
    available = [c for c in feature_cols if c in X.columns]
    X = X[available]

    predicted_aqi = int(round(float(model.predict(X)[0])))
    return PredictResponse(city=city, predicted_aqi=predicted_aqi)
