import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import hopsworks
import tempfile

load_dotenv()

os.makedirs("C:\\tmp", exist_ok=True)
tempfile.tempdir = "C:\\tmp"

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

CITIES = {
    "karachi":   (24.8607, 67.0011),
    "lahore":    (31.5497, 74.3436),
    "islamabad": (33.6844, 73.0479),
    "peshawar":  (34.0151, 71.5249),
    "hyderabad": (25.3960, 68.3578),
}

DAYS_BACK = 4


def fetch_openweather_history(city: str, lat: float, lon: float, dt: datetime) -> dict:
    unix_ts = int(dt.timestamp())
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={unix_ts}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    hour_data = data.get("data", [{}])[0]

    return {
        "temperature": hour_data.get("temp"),
        "humidity": hour_data.get("humidity"),
        "wind": hour_data.get("wind_speed"),
        "pressure": hour_data.get("pressure"),
    }


def fetch_current_aqi(city: str) -> dict:
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "ok":
        raise ValueError(f"API error for {city}")
    d = data["data"]
    iaqi = d.get("iaqi", {})
    return {
        "aqi": d.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
    }


def run():
    records = []
    now = datetime.now(timezone.utc)

    for city, (lat, lon) in CITIES.items():
        print(f"\nBackfilling: {city}")
        try:
            aqi_data = fetch_current_aqi(city)
        except Exception as e:
            print(f"  AQI fetch failed for {city}: {e}")
            continue

        for days_ago in range(DAYS_BACK, 0, -1):
            for hour in [0, 6, 12, 18]:
                dt = (now - timedelta(days=days_ago)).replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )
                try:
                    weather = fetch_openweather_history(city, lat, lon, dt)
                except Exception as e:
                    print(f"  Weather fetch failed for {city} at {dt}: {e}")
                    weather = {"temperature": None, "humidity": None, "wind": None, "pressure": None}

                record = {
                    "city": city,
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                    "month": dt.month,
                    **aqi_data,
                    **weather,
                }
                records.append(record)
                print(f"  {dt.strftime('%Y-%m-%d %H:00')} | AQI: {aqi_data['aqi']}")

    if not records:
        print("No records generated.")
        return

    df = pd.DataFrame(records)
    float_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co", "temperature", "humidity", "wind", "pressure"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    print(f"\nTotal records: {len(df)}")
    print(df[["city", "timestamp", "aqi"]].to_string())

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
    print("\nBackfill data inserted into Hopsworks Feature Store.")


if __name__ == "__main__":
    run()
