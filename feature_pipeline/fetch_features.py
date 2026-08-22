import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import hopsworks

load_dotenv()

import tempfile
os.environ["HOPSWORKS_ALLOW_WRITEACCESS"] = "true"
os.makedirs("C:\\tmp", exist_ok=True)
tempfile.tempdir = "C:\\tmp"

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

CITIES = ["karachi", "lahore", "islamabad", "peshawar", "hyderabad"]


def fetch_aqi(city: str) -> dict:
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data["status"] != "ok":
        raise ValueError(f"API error for {city}: {data}")

    d = data["data"]
    iaqi = d.get("iaqi", {})

    now = datetime.now(timezone.utc)

    return {
        "city": city,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "month": now.month,
        "aqi": d.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "temperature": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
        "wind": iaqi.get("w", {}).get("v"),
        "pressure": iaqi.get("p", {}).get("v"),
    }


def run():
    records = []
    for city in CITIES:
        try:
            record = fetch_aqi(city)
            records.append(record)
            print(f"Fetched: {city} | AQI: {record['aqi']}")
        except Exception as e:
            print(f"Failed for {city}: {e}")

    if not records:
        print("No data fetched. Exiting.")
        return

    df = pd.DataFrame(records)
    float_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co", "temperature", "humidity", "wind", "pressure"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    # Derived features: aqi_lag1 and aqi_change_rate
    # On a real-time single-run, we cannot access previous stored values easily,
    # so we default both to 0.0. These will be populated with real values during
    # backfill and will become meaningful as the feature store accumulates data.
    df["aqi_lag1"] = 0.0
    df["aqi_change_rate"] = 0.0

    print(df)

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly AQI and weather features per city",
        time_travel_format="HUDI",
    )

    fg.insert(df)
    print("Data inserted into Hopsworks Feature Store.")


if __name__ == "__main__":
    run()
