# Pearls AQI Predictor

> Real-time Air Quality Index forecasting for 5 major Pakistani cities — powered by an end-to-end ML pipeline with automated data ingestion, model training, and a live public dashboard.

**Live Dashboard:** [pearls-aqi-predictor-5sgmy7ysssfsnquczwejs9.streamlit.app](https://pearls-aqi-predictor-5sgmy7ysssfsnquczwejs9.streamlit.app)

---

## Overview

Air pollution is a critical public health issue in Pakistan. Pearls AQI Predictor monitors and forecasts the Air Quality Index (AQI) for **Karachi, Lahore, Islamabad, Peshawar, and Hyderabad** using real-time data from the AQICN API. An automated ML pipeline trains models daily and serves predictions through an interactive dashboard.

---

## Architecture

```
AQICN API (live AQI data)
        |
        v
Feature Pipeline (fetch_features.py)          <-- runs every hour via GitHub Actions
        |
        v
Hopsworks Feature Store (aqi_features v1)     <-- centralized feature storage (HUDI format)
        |
        v
Training Pipeline (train.py)                  <-- runs daily at 02:00 UTC via GitHub Actions
        |
        v
Hopsworks Model Registry (aqi_forecaster)     <-- versioned model storage
        |
        v
Streamlit Dashboard (dashboard.py)            <-- reads live data + model from Hopsworks
FastAPI Backend    (api.py)                   <-- REST API for predictions
```

---

## Features

| Feature | Details |
|---|---|
| **Live AQI data** | Fetched hourly from AQICN API for 5 Pakistani cities |
| **Feature Store** | Hopsworks `aqi_features` feature group (HUDI format, versioned) |
| **Derived features** | `aqi_lag1` (previous AQI), `aqi_change_rate` (rate of change) |
| **3 ML models** | Ridge Regression, Random Forest, Keras Neural Network (TensorFlow) |
| **Auto model selection** | Best model by RMSE saved to Hopsworks Model Registry |
| **Daily retraining** | GitHub Actions cron runs training pipeline every day at 02:00 UTC |
| **SHAP explainability** | Feature importance chart displayed in dashboard |
| **EDA script** | Automated analysis with 3 saved plots |
| **FastAPI backend** | REST endpoints for health check, city list, and AQI prediction |
| **Public dashboard** | Deployed on Streamlit Cloud with warm earth theme |

---

## Project Structure

```
Pearls AQI Predictor/
├── app/
│   ├── dashboard.py          # Streamlit dashboard (main UI)
│   └── api.py                # FastAPI backend
├── feature_pipeline/
│   ├── fetch_features.py     # Hourly AQI ingestion → Hopsworks
│   └── backfill.py           # One-time historical data backfill
├── training_pipeline/
│   ├── train.py              # Model training + Hopsworks model registry
│   └── model/                # Locally saved model artifacts
├── eda/
│   ├── eda.py                # Exploratory data analysis script
│   └── plots/                # Saved EDA charts (PNG)
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml   # Hourly GitHub Actions workflow
│       └── training_pipeline.yml  # Daily GitHub Actions workflow
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── requirements.txt          # Python dependencies (Streamlit Cloud)
├── runtime.txt               # Python version pin (3.11)
└── .env                      # Local secrets (NOT committed — gitignored)
```

---

## Cities Monitored

| City | Coordinates |
|---|---|
| Karachi | 24.8607° N, 67.0011° E |
| Lahore | 31.5497° N, 74.3436° E |
| Islamabad | 33.6844° N, 73.0479° E |
| Peshawar | 34.0151° N, 71.5249° E |
| Hyderabad | 25.3960° N, 68.3578° E |

---

## ML Models

Three models are trained and evaluated on each daily run. The best-performing model by RMSE is saved to the Hopsworks Model Registry.

### 1. Ridge Regression
- Linear model with L2 regularization (`alpha=1.0`)
- Fast baseline, works well with limited data

### 2. Random Forest Regressor
- Ensemble of 100 decision trees (`n_estimators=100`)
- Handles non-linear feature interactions
- Used for SHAP feature importance (TreeExplainer)

### 3. Keras Neural Network (TensorFlow)
- Architecture: `Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)`
- Optimizer: Adam | Loss: MSE | Epochs: 50

### Feature Columns
```
hour, day_of_week, month, pm25, temperature, humidity,
wind, pressure, city_encoded, aqi_lag1, aqi_change_rate
```

`aqi_lag1` and `aqi_change_rate` are derived in the training pipeline from sorted feature store data (not stored in the feature group directly).

---

## Automated Pipelines (GitHub Actions)

### Feature Pipeline — every hour
```yaml
cron: '0 * * * *'
script: python feature_pipeline/fetch_features.py
```
Fetches current AQI from AQICN for all 5 cities and inserts a new row into the Hopsworks Feature Store.

### Training Pipeline — daily at 02:00 UTC
```yaml
cron: '0 2 * * *'
script: python training_pipeline/train.py
```
Reads all feature data from Hopsworks, trains all 3 models, evaluates by RMSE, and saves the best model to the Hopsworks Model Registry.

---

## Dashboard

The Streamlit dashboard connects directly to Hopsworks to read live feature data and the latest trained model.

**Sections:**
- **KPI Cards** — Current AQI, PM2.5, Temperature, Humidity
- **AQI Gauge** — Visual indicator with Good / Moderate / Unhealthy bands
- **Health Alert** — Contextual advice based on AQI level
- **AQI Trend** — Historical AQI line chart
- **AQI by Hour of Day** — Average AQI per hour (bar chart)
- **PM2.5 vs AQI Scatter** — Correlation plot
- **SHAP Feature Importance** — Which features drive the model's predictions

**Theme:** White + warm earth tones (`#f5f2ec` background, `#c0392b` accent)

---

## FastAPI Backend

Run locally:
```bash
uvicorn app.api:app --reload
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/cities` | GET | List of monitored cities |
| `/aqi/{city}` | GET | Latest AQI data for a city |
| `/predict` | POST | Predict AQI given feature values |

---

## EDA Script

Connects to Hopsworks, loads all feature data, and outputs:
- Distribution statistics per city
- AQI by city (box plot)
- AQI by hour of day (line chart)
- Feature correlation heatmap

Run:
```bash
python eda/eda.py
```

Plots are saved to `eda/plots/`.

---

## Local Setup

### Prerequisites
- Python 3.11
- Hopsworks account (EU-West cluster)
- AQICN API token (free at [aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/))

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install tensorflow fastapi uvicorn shap
```

### 4. Create `.env` file
```
AQICN_TOKEN=your_aqicn_token_here
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=your_project_name_here
```

> Never commit `.env` — it is listed in `.gitignore`.

### 5. Run the feature pipeline (initial data load)
```bash
python feature_pipeline/backfill.py   # backfill 4 days of historical data
python feature_pipeline/fetch_features.py  # fetch current AQI
```

### 6. Train the model
```bash
python training_pipeline/train.py
```

### 7. Launch the dashboard
```bash
streamlit run app/dashboard.py
```

---

## GitHub Actions Secrets Required

Go to **Settings → Secrets and variables → Actions** in your repository and add:

| Secret | Description |
|---|---|
| `HOPSWORKS_API_KEY` | Your Hopsworks API key |
| `HOPSWORKS_PROJECT` | Your Hopsworks project name |
| `AQICN_TOKEN` | Your AQICN API token |

---

## Streamlit Cloud Deployment

1. Connect your GitHub repository on [share.streamlit.io](https://share.streamlit.io)
2. Set **Main file path** to `app/dashboard.py`
3. Add secrets under **Advanced settings → Secrets** in TOML format:
   ```toml
   HOPSWORKS_API_KEY = "your_key_here"
   HOPSWORKS_PROJECT = "your_project_here"
   AQICN_TOKEN = "your_token_here"
   ```
4. Python version is pinned to 3.11 via `runtime.txt`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data source | AQICN API |
| Feature store | Hopsworks (EU-West, HUDI format) |
| Model registry | Hopsworks Model Registry |
| ML models | scikit-learn, TensorFlow/Keras |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| Backend API | FastAPI |
| Automation | GitHub Actions |
| Deployment | Streamlit Cloud |
| Language | Python 3.11 |

---

## AQI Scale Reference

| AQI Range | Category | Health Implication |
|---|---|---|
| 0 – 50 | Good | Air quality is satisfactory |
| 51 – 100 | Moderate | Acceptable; some pollutants may affect sensitive groups |
| 101 – 150 | Unhealthy for Sensitive Groups | General public is not likely to be affected |
| 151 – 200 | Unhealthy | Everyone may begin to experience health effects |
| 201 – 300 | Very Unhealthy | Health alert; everyone may experience more serious effects |
| 300+ | Hazardous | Emergency conditions; entire population affected |

---

## Author

**Unaiza Asif**
Gatronova Group / Novatex Limited
[amaltaf3kp@gatronova.com](mailto:amaltaf3kp@gatronova.com)

---

## License

This project is for academic and research purposes.
