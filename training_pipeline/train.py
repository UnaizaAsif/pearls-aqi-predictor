import os
import tempfile
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import hopsworks

# Optional TensorFlow/Keras — skipped gracefully if not installed
try:
    import tensorflow as tf
    from tensorflow import keras

    class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
        """Sklearn-compatible wrapper for a Keras Sequential regressor."""

        def __init__(self, input_dim, epochs=50, batch_size=16):
            self.input_dim = input_dim
            self.epochs = epochs
            self.batch_size = batch_size
            self.model_ = None

        def _build_model(self):
            model = keras.Sequential([
                keras.layers.Input(shape=(self.input_dim,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation="linear"),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

        def fit(self, X, y):
            self.model_ = self._build_model()
            self.model_.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0,
            )
            return self

        def predict(self, X):
            return self.model_.predict(X, verbose=0).flatten()

    TF_AVAILABLE = True
    print("TensorFlow available — Keras model will be included.")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed — Keras model will be skipped.")

load_dotenv()

os.makedirs("C:\\tmp", exist_ok=True)
tempfile.tempdir = "C:\\tmp"

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FEATURE_COLS = ["hour", "day_of_week", "month", "pm25",
                "temperature", "humidity", "wind", "pressure", "city_encoded",
                "aqi_lag1", "aqi_change_rate"]
TARGET_COL = "aqi"


def load_data():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    try:
        df = fg.read()
    except Exception:
        df = fg.read(read_options={"use_hive": True})
    print(f"Loaded {len(df)} rows from Feature Store")
    return df, project


def prepare_features(df):
    le = LabelEncoder()
    df["city_encoded"] = le.fit_transform(df["city"])

    df = df.dropna(subset=[TARGET_COL])

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    core_cols = ["hour", "day_of_week", "month", "city_encoded"]
    df = df.dropna(subset=[c for c in core_cols if c in df.columns])
    df[available_cols] = df[available_cols].fillna(0.0)

    return df, le


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train():
    df, project = load_data()
    df, le = prepare_features(df)

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df[TARGET_COL]

    if len(X) < 10:
        print("Not enough data to train yet. Run the feature pipeline more times first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    if TF_AVAILABLE:
        models["Keras Neural Network"] = KerasRegressorWrapper(
            input_dim=len(X_train.columns),
            epochs=50,
            batch_size=16,
        )

    best_model = None
    best_rmse = float("inf")
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(name, y_test, y_pred)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (RMSE: {best_rmse:.2f})")

    model_dir = "training_pipeline/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/aqi_model.pkl")
    joblib.dump(le, f"{model_dir}/city_encoder.pkl")
    joblib.dump(available_cols, f"{model_dir}/feature_cols.pkl")
    print("Model saved locally.")

    mr = project.get_model_registry()
    model_obj = mr.python.create_model(
        name="aqi_forecaster",
        metrics={"rmse": best_rmse},
        description=f"Best model: {best_name}",
    )
    model_obj.save(model_dir)
    print("Model saved to Hopsworks Model Registry.")


if __name__ == "__main__":
    train()
