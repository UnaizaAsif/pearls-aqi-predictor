import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import joblib
import hopsworks

load_dotenv()

os.makedirs("C:\\tmp", exist_ok=True)
tempfile.tempdir = "C:\\tmp"

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

CITIES = ["karachi", "lahore", "islamabad", "peshawar", "hyderabad"]

AQI_LEVELS = [
    (0,   50,  "Good",                          "#00c853", "#e8f5e9"),
    (51,  100, "Moderate",                      "#ffd600", "#fffde7"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff6d00", "#fff3e0"),
    (151, 200, "Unhealthy",                     "#d50000", "#ffebee"),
    (201, 300, "Very Unhealthy",                "#6a1b9a", "#f3e5f5"),
    (301, 500, "Hazardous",                     "#37474f", "#eceff1"),
]

CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#faf8f4",
    font=dict(color="#2c2416", family="sans-serif"),
    margin=dict(l=40, r=20, t=20, b=60),
    xaxis=dict(
        showgrid=False,
        linecolor="#e8e4da",
        tickcolor="#e8e4da",
        tickfont=dict(size=10, color="#a09080"),
    ),
    yaxis=dict(
        gridcolor="#ede9e0",
        linecolor="#e8e4da",
        tickfont=dict(size=10, color="#a09080"),
        zeroline=False,
    ),
    hovermode="x unified",
)


def get_aqi_info(aqi):
    for low, high, label, color, bg in AQI_LEVELS:
        if low <= aqi <= high:
            return label, color, bg
    return "Hazardous", "#37474f", "#eceff1"


@st.cache_resource(ttl=3600)
def load_model_and_data():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    mr = project.get_model_registry()
    model_obj = mr.get_best_model("aqi_forecaster", metric="rmse", direction="min")
    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
    encoder = joblib.load(os.path.join(model_dir, "city_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    return model, encoder, feature_cols, df


def predict_next_3_days(city, model, encoder, feature_cols, df):
    city_df = df[df["city"] == city].copy()
    if len(city_df) == 0:
        return None
    latest = city_df.sort_values("timestamp").iloc[-1]
    predictions = []
    now = datetime.now(timezone.utc)
    for day in range(1, 4):
        for hour in [6, 12, 18]:
            future_dt = now + timedelta(days=day)
            future_dt = future_dt.replace(hour=hour, minute=0, second=0)
            row = {
                "hour": hour,
                "day_of_week": future_dt.weekday(),
                "month": future_dt.month,
                "pm25": latest.get("pm25", 0) or 0,
                "temperature": latest.get("temperature", 25) or 25,
                "humidity": latest.get("humidity", 50) or 50,
                "wind": latest.get("wind", 2) or 2,
                "pressure": latest.get("pressure", 1013) or 1013,
                "city_encoded": encoder.transform([city])[0],
            }
            X = pd.DataFrame([row])
            X = X[[c for c in feature_cols if c in X.columns]]
            pred = model.predict(X)[0]
            predictions.append({
                "datetime": future_dt.strftime("%Y-%m-%d %H:00"),
                "day": future_dt.strftime("%A"),
                "date": future_dt.strftime("%b %d"),
                "hour": hour,
                "predicted_aqi": round(pred),
            })
    return pd.DataFrame(predictions)


def aqi_gauge_html(aqi, label, color):
    pct = min(aqi / 300 * 100, 100)
    return f"""
    <div style="background: white; border-radius: 20px; padding: 18px 30px; text-align: center; border: 1px solid #e8e4da; box-shadow: 0 2px 12px rgba(0,0,0,0.06); height: 190px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 12px; color: #b8a898; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;">Current AQI</div>
        <div style="font-size: 56px; font-weight: 900; color: {color}; line-height: 1; margin-bottom: 6px;">{aqi}</div>
        <div style="display: inline-block; background: {color}18; border: 1px solid {color}44; color: {color}; padding: 4px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; margin-bottom: 20px;">{label}</div>
        <div style="background: #f0ece4; border-radius: 10px; height: 8px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #27ae60, #f39c12, #c0392b); width: {pct}%; height: 100%; border-radius: 10px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 10px; color: #c0b8a8;">
            <span>0</span><span>Good</span><span>Moderate</span><span>Unhealthy</span><span>300</span>
        </div>
    </div>
    """


def forecast_card_html(day, date, aqi, label, color, bg):
    return f"""
    <div style="background: white; border-radius: 16px; padding: 24px 16px; text-align: center; border-top: 4px solid {color}; border-left: 1px solid #e8e4da; border-right: 1px solid #e8e4da; border-bottom: 1px solid #e8e4da; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin: 0 4px;">
        <div style="font-size: 12px; color: #c0b0a0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{day}</div>
        <div style="font-size: 12px; color: #d0c8bc; margin-bottom: 16px;">{date}</div>
        <div style="font-size: 48px; font-weight: 900; color: {color}; line-height: 1; margin-bottom: 8px;">{aqi}</div>
        <div style="font-size: 11px; color: {color}; background: {color}18; padding: 3px 10px; border-radius: 12px; display: inline-block; border: 1px solid {color}33;">{label}</div>
    </div>
    """


def make_line_chart(x, y, line_color, fill_color, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=line_color, width=2.5),
        fill="tozeroy",
        fillcolor=fill_color,
        name=y_label,
        hovertemplate=f"%{{x}}<br>{y_label}: %{{y}}<extra></extra>",
    ))
    layout = dict(CHART_LAYOUT)
    fig.update_layout(**layout, height=260, showlegend=False)
    return fig


def main():
    st.set_page_config(
        page_title="Pearls AQI Predictor",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
        .stApp { background: #f5f2ec; }
        .block-container { padding-top: 2rem; max-width: 1100px; }
        h1, h2, h3 { color: #2c2416 !important; }

        /* Hide black header bar */
        header[data-testid="stHeader"] { background: #f5f2ec !important; box-shadow: none !important; }
        header[data-testid="stHeader"] * { color: #2c2416 !important; }
        [data-testid="stToolbar"] { display: none !important; }

        /* White dropdown */
        .stSelectbox label { color: #8a7a6a !important; font-size: 13px !important; }
        div[data-baseweb="select"] { background: white !important; }
        div[data-baseweb="select"] > div {
            background: white !important;
            border: 1px solid #e0dbd0 !important;
            border-radius: 12px !important;
            color: #2c2416 !important;
        }
        div[data-baseweb="select"] span { color: #2c2416 !important; }
        div[data-baseweb="select"] svg { fill: #a09080 !important; }
        [data-baseweb="popover"] { background: white !important; }
        [role="option"] { background: white !important; color: #2c2416 !important; }
        [role="option"]:hover { background: #f5f2ec !important; }

        .stPlotlyChart { border-radius: 16px; overflow: hidden; border: 1px solid #e8e4da; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 36px; font-weight: 900; color: #2c2416;">🌬️ Pearls AQI Predictor</span>
    </div>
    <div style="font-size: 14px; color: #a09080; margin-bottom: 32px; letter-spacing: 1px;">
        REAL-TIME AIR QUALITY FORECAST FOR PAKISTANI CITIES
    </div>
    """, unsafe_allow_html=True)

    col_city, _ = st.columns([1, 2])
    with col_city:
        city = st.selectbox("Select City", CITIES, format_func=lambda x: x.title())

    with st.spinner(""):
        try:
            model, encoder, feature_cols, df = load_model_and_data()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    city_df = df[df["city"] == city].sort_values("timestamp")

    if len(city_df) == 0:
        st.warning("No data available for this city yet.")
        return

    latest = city_df.iloc[-1]
    current_aqi = int(latest.get("aqi", 0) or 0)
    label, color, bg = get_aqi_info(current_aqi)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(aqi_gauge_html(current_aqi, label, color), unsafe_allow_html=True)

    CARD_STYLE = "background: white; border-radius: 16px; padding: 0 24px; text-align: center; border: 1px solid #e8e4da; box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 190px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;"

    with col2:
        pm25 = latest.get("pm25", "N/A")
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="font-size: 12px; color: #b8a898; letter-spacing: 2px; text-transform: uppercase;">PM2.5</div>
            <div style="font-size: 44px; font-weight: 800; color: #2980b9; line-height: 1;">{pm25}</div>
            <div style="font-size: 11px; color: #c0b8a8;">μg/m³</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        temp = latest.get("temperature", "N/A")
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="font-size: 12px; color: #b8a898; letter-spacing: 2px; text-transform: uppercase;">Temperature</div>
            <div style="font-size: 44px; font-weight: 800; color: #e67e22; line-height: 1;">{temp}</div>
            <div style="font-size: 11px; color: #c0b8a8;">°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        humidity = latest.get("humidity", "N/A")
        st.markdown(f"""
        <div style="{CARD_STYLE}">
            <div style="font-size: 12px; color: #b8a898; letter-spacing: 2px; text-transform: uppercase;">Humidity</div>
            <div style="font-size: 44px; font-weight: 800; color: #16a085; line-height: 1;">{humidity}</div>
            <div style="font-size: 11px; color: #c0b8a8;">%</div>
        </div>
        """, unsafe_allow_html=True)

    if current_aqi > 150:
        st.markdown(f"""
        <div style="margin-top: 20px; background: #fdecea; border: 1px solid #f5c6c2; border-radius: 12px; padding: 16px 20px; color: #c0392b;">
            🚨 <strong>Health Alert:</strong> Air quality in {city.title()} is <strong>{label}</strong>. Avoid all outdoor activities. Keep windows closed.
        </div>
        """, unsafe_allow_html=True)
    elif current_aqi > 100:
        st.markdown(f"""
        <div style="margin-top: 20px; background: #fef3e2; border: 1px solid #f5d9a8; border-radius: 12px; padding: 16px 20px; color: #d35400;">
            ⚠️ <strong>Caution:</strong> Air quality in {city.title()} is <strong>{label}</strong>. Sensitive groups should limit outdoor exposure.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 20px; font-weight: 700; color: #2c2416; margin-bottom: 16px;">📅 3-Day Forecast — {city.title()}</div>', unsafe_allow_html=True)

    predictions = predict_next_3_days(city, model, encoder, feature_cols, df)

    if predictions is not None:
        daily = predictions.groupby(["day", "date"])["predicted_aqi"].mean().round().astype(int).reset_index()

        cols = st.columns(3)
        for i, (_, row) in enumerate(daily.iterrows()):
            lbl, clr, bgg = get_aqi_info(row["predicted_aqi"])
            with cols[i]:
                st.markdown(forecast_card_html(row["day"], row["date"], row["predicted_aqi"], lbl, clr, bgg), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_forecast = make_line_chart(
            x=predictions["datetime"],
            y=predictions["predicted_aqi"],
            line_color="#c0392b",
            fill_color="rgba(192,57,43,0.08)",
            y_label="Predicted AQI",
        )
        st.plotly_chart(fig_forecast, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 20px; font-weight: 700; color: #2c2416; margin-bottom: 16px;">📈 Historical AQI</div>', unsafe_allow_html=True)
    recent = city_df.tail(24).copy()
    fig_history = make_line_chart(
        x=recent["timestamp"],
        y=recent["aqi"],
        line_color="#7b8cde",
        fill_color="rgba(123,140,222,0.10)",
        y_label="AQI",
    )
    st.plotly_chart(fig_history, width="stretch")

    # ------------------------------------------------------------------
    # SHAP Feature Importance
    # ------------------------------------------------------------------
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge

        LOCAL_MODEL_PATH = os.path.join(
            os.path.dirname(__file__), "..", "training_pipeline", "model", "aqi_model.pkl"
        )
        LOCAL_MODEL_PATH = os.path.normpath(LOCAL_MODEL_PATH)

        if os.path.exists(LOCAL_MODEL_PATH):
            local_model = joblib.load(LOCAL_MODEL_PATH)

            SHAP_FEATURE_NAMES = [
                "Hour", "Day of Week", "Month", "PM2.5",
                "Temperature", "Humidity", "Wind", "Pressure", "City",
                "AQI Lag 1", "AQI Change Rate",
            ]

            # Prepare sample data for SHAP (up to 50 rows for the selected city)
            shap_df = city_df.copy()
            shap_df = shap_df.dropna(subset=["aqi"])
            # Align to feature_cols from the loaded Hopsworks model
            shap_df = shap_df[[c for c in feature_cols if c in shap_df.columns]].fillna(0.0)
            shap_sample = shap_df.head(50)

            if len(shap_sample) > 0:
                if isinstance(local_model, RandomForestRegressor):
                    explainer = shap.TreeExplainer(local_model)
                else:
                    explainer = shap.LinearExplainer(local_model, shap_sample)

                shap_values = explainer.shap_values(shap_sample)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

                # Map feature_cols names to display names
                col_display_map = {
                    "hour": "Hour",
                    "day_of_week": "Day of Week",
                    "month": "Month",
                    "pm25": "PM2.5",
                    "temperature": "Temperature",
                    "humidity": "Humidity",
                    "wind": "Wind",
                    "pressure": "Pressure",
                    "city_encoded": "City",
                    "aqi_lag1": "AQI Lag 1",
                    "aqi_change_rate": "AQI Change Rate",
                }
                used_cols = [c for c in feature_cols if c in shap_sample.columns]
                display_names = [col_display_map.get(c, c) for c in used_cols]

                # Sort by importance descending
                sorted_idx = np.argsort(mean_abs_shap)
                sorted_names = [display_names[i] for i in sorted_idx]
                sorted_vals = mean_abs_shap[sorted_idx]

                fig_shap = go.Figure(go.Bar(
                    x=sorted_vals,
                    y=sorted_names,
                    orientation="h",
                    marker=dict(
                        color=sorted_vals,
                        colorscale=[[0, "#e8c9a0"], [1, "#c0392b"]],
                        showscale=False,
                    ),
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                ))

                shap_layout = dict(CHART_LAYOUT)
                fig_shap.update_layout(
                    **shap_layout,
                    height=320,
                    showlegend=False,
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="",
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size: 20px; font-weight: 700; color: #2c2416; margin-bottom: 16px;">'
                    "Feature Importance (SHAP)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig_shap, width="stretch")

    except Exception:
        # If shap is not installed or model not found locally, skip silently
        pass

    st.markdown("""
    <div style="margin-top: 40px; text-align: center; color: #b8a898; font-size: 12px; letter-spacing: 1px;">
        PEARLS AQI PREDICTOR • DATA FROM AQICN • UPDATED HOURLY
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
