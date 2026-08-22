"""
Exploratory Data Analysis for Pearls AQI Predictor
----------------------------------------------------
Loads data from the Hopsworks Feature Store and prints statistical summaries.
Saves three matplotlib plots to eda/plots/.
"""

import os
import tempfile

os.makedirs("C:\\tmp", exist_ok=True)
tempfile.tempdir = "C:\\tmp"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

NUMERIC_COLS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind", "pressure",
    "aqi_lag1", "aqi_change_rate",
]


def load_data() -> pd.DataFrame:
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    print(f"Loaded {len(df)} rows from Hopsworks Feature Store.\n")
    return df


def run_eda(df: pd.DataFrame):
    # ------------------------------------------------------------------
    # 1. Basic stats
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. BASIC STATS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Data types:")
    print(df.dtypes.to_string())
    print("\nDescriptive statistics:")
    print(df.describe(include="all").to_string())

    # ------------------------------------------------------------------
    # 2. Missing values
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. MISSING VALUE COUNTS PER COLUMN")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values found.")

    # ------------------------------------------------------------------
    # 3. AQI distribution per city
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. AQI DISTRIBUTION PER CITY")
    print("=" * 60)
    city_aqi = df.groupby("city")["aqi"].agg(
        min="min", max="max", mean="mean", median="median", std="std"
    ).round(2)
    print(city_aqi.to_string())

    # ------------------------------------------------------------------
    # 4. Correlation matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. CORRELATION MATRIX (numeric columns)")
    print("=" * 60)
    available_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[available_numeric].corr().round(3)
    print(corr.to_string())

    # ------------------------------------------------------------------
    # 5. Hourly AQI pattern
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. AVERAGE AQI BY HOUR OF DAY (all cities)")
    print("=" * 60)
    hourly = df.groupby("hour")["aqi"].mean().round(2)
    for hour, val in hourly.items():
        bar = "#" * int(val / 10)
        print(f"  Hour {hour:02d}: {val:6.2f}  {bar}")

    # ------------------------------------------------------------------
    # 6. City ranking by average AQI
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. CITY RANKING BY AVERAGE AQI (descending)")
    print("=" * 60)
    city_ranking = df.groupby("city")["aqi"].mean().sort_values(ascending=False).round(2)
    for rank, (city, avg) in enumerate(city_ranking.items(), start=1):
        print(f"  {rank}. {city.title():12s}  avg AQI: {avg}")

    # ------------------------------------------------------------------
    # 7. Save plots
    # ------------------------------------------------------------------
    _plot_aqi_by_city(df)
    _plot_aqi_by_hour(df, hourly)
    _plot_correlation_heatmap(df, available_numeric)

    print("\nPlots saved to:", PLOTS_DIR)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_aqi_by_city(df: pd.DataFrame):
    city_avg = df.groupby("city")["aqi"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [c.title() for c in city_avg.index],
        city_avg.values,
        color=["#c0392b", "#e67e22", "#f39c12", "#27ae60", "#2980b9"],
        edgecolor="white",
        linewidth=1.2,
    )
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
    ax.set_title("Average AQI per City", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("City", fontsize=11)
    ax.set_ylabel("Average AQI", fontsize=11)
    ax.set_ylim(0, city_avg.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#faf8f4")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "aqi_by_city.png"), dpi=150)
    plt.close(fig)
    print("  Saved aqi_by_city.png")


def _plot_aqi_by_hour(df: pd.DataFrame, hourly: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly.index, hourly.values, color="#7b8cde", linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#7b8cde")
    ax.set_title("Average AQI by Hour of Day", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax.set_ylabel("Average AQI", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#faf8f4")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "aqi_by_hour.png"), dpi=150)
    plt.close(fig)
    print("  Saved aqi_by_hour.png")


def _plot_correlation_heatmap(df: pd.DataFrame, cols: list):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="#e8e4da",
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved correlation_heatmap.png")


if __name__ == "__main__":
    data = load_data()
    run_eda(data)
