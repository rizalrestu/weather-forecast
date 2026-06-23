"""Streamlit dashboard — AQI & PM2.5 forecast monitor."""
import os
import psycopg2
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_URL = os.getenv("AQI_DB_URL", "postgresql://airflow:airflow@localhost/aqi_db")

st.set_page_config(
    page_title="AQI Monitor — Indonesia",
    page_icon="🌫️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_readings(city: str) -> pd.DataFrame:
    sql = """
        SELECT timestamp, pm2_5_ugm3, pm10_ugm3, temperature_2m_c,
               relative_humidity_pct, wind_speed_10m_kmh
        FROM aqi_readings
        WHERE city = %s
        ORDER BY timestamp DESC
        LIMIT 48
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql_query(sql, conn, params=(city,))
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_predictions(city: str) -> pd.DataFrame:
    sql = """
        SELECT timestamp, actual_pm25, predicted_pm25
        FROM aqi_predictions
        WHERE city = %s
        ORDER BY timestamp DESC
        LIMIT 48
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql_query(sql, conn, params=(city,))
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_forecasts(city: str) -> pd.DataFrame:
    sql = """
        SELECT timestamp, predicted_pm25, forecast_made_at
        FROM aqi_forecasts
        WHERE city = %s
        ORDER BY timestamp ASC
        LIMIT 24
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql_query(sql, conn, params=(city,))
    return df


@st.cache_data(ttl=300)
def load_anomalies(city: str) -> pd.DataFrame:
    sql = """
        SELECT timestamp, pm2_5_ugm3, anomaly_score
        FROM aqi_anomalies
        WHERE city = %s AND is_anomaly = TRUE
        ORDER BY timestamp DESC
        LIMIT 48
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql_query(sql, conn, params=(city,))
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pm25_status(value: float) -> tuple[str, str]:
    """Return (label, color) based on PM2.5 µg/m³."""
    if value <= 15:
        return "Baik", "#2ecc71"
    elif value <= 65:
        return "Sedang", "#f1c40f"
    elif value <= 150:
        return "Tidak Sehat", "#e67e22"
    elif value <= 250:
        return "Sangat Tidak Sehat", "#e74c3c"
    else:
        return "Berbahaya", "#8e44ad"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🌫️ AQI & PM2.5 Forecast Monitor")
st.caption("Data dari open-meteo · diperbarui tiap jam via Airflow")

from src.ingestion.extract_api import CITIES as CITY_CONFIGS  # noqa: E402
city = st.sidebar.selectbox("Kota", list(CITY_CONFIGS.keys()), format_func=str.capitalize)
st.sidebar.markdown("---")
st.sidebar.info("Dashboard auto-refresh setiap 5 menit.")

try:
    readings    = load_readings(city)
    predictions = load_predictions(city)
    anomalies   = load_anomalies(city)
    forecasts   = load_forecasts(city)
except Exception as e:
    st.error(f"Tidak bisa konek ke database: {e}")
    st.stop()

if readings.empty:
    st.warning("Belum ada data. Trigger DAG `aqi_pipeline` di Airflow dulu.")
    st.stop()

latest = readings.iloc[-1]
label, color = pm25_status(latest["pm2_5_ugm3"])

# Metric cards
st.subheader(f"Kondisi Terkini — {city.capitalize()}")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("PM2.5", f"{latest['pm2_5_ugm3']:.1f} µg/m³")
col2.metric("PM10",  f"{latest['pm10_ugm3']:.1f} µg/m³")
col3.metric("Suhu",  f"{latest['temperature_2m_c']:.1f} °C")
col4.metric("Kelembaban", f"{latest['relative_humidity_pct']:.0f}%")
col5.metric("Angin", f"{latest['wind_speed_10m_kmh']:.1f} km/h")

st.markdown(
    f"**Status PM2.5:** <span style='color:{color}; font-weight:bold'>{label}</span>",
    unsafe_allow_html=True,
)

st.divider()

# PM2.5 time series chart
st.subheader("PM2.5 — Aktual vs Prediksi (48 jam terakhir)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=readings["timestamp"], y=readings["pm2_5_ugm3"],
    name="Aktual", line=dict(color="#3498db", width=2),
))

if not predictions.empty:
    fig.add_trace(go.Scatter(
        x=predictions["timestamp"], y=predictions["predicted_pm25"],
        name="Prediksi (XGBoost)",
        line=dict(color="#e67e22", width=2, dash="dash"),
    ))

if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"], y=anomalies["pm2_5_ugm3"],
        name="Anomali", mode="markers",
        marker=dict(color="red", size=10, symbol="x"),
    ))

fig.update_layout(
    xaxis_title="Waktu",
    yaxis_title="PM2.5 (µg/m³)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=400,
    margin=dict(t=40),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# 24-hour forecast chart
st.subheader("Forecast PM2.5 — 24 Jam ke Depan (XGBoost Recursive)")

if forecasts.empty:
    st.info("Belum ada data forecast. Trigger DAG sekali lagi setelah update terbaru.")
else:
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=forecasts["timestamp"], y=forecasts["predicted_pm25"],
        name="Forecast PM2.5",
        line=dict(color="#e67e22", width=2.5),
        mode="lines+markers",
        marker=dict(size=6),
    ))

    fig2.update_layout(
        xaxis_title="Waktu (Jam ke Depan)",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified",
        height=350,
        margin=dict(t=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

    made_at = forecasts["forecast_made_at"].iloc[0] if "forecast_made_at" in forecasts.columns else "—"
    st.caption(f"Forecast dibuat pada: {made_at}")

st.divider()

# Recent readings table
st.subheader("Data Terbaru")
display_df = readings[["timestamp", "pm2_5_ugm3", "pm10_ugm3",
                        "temperature_2m_c", "relative_humidity_pct",
                        "wind_speed_10m_kmh"]].tail(24).sort_values(
    "timestamp", ascending=False
).rename(columns={
    "timestamp":             "Waktu",
    "pm2_5_ugm3":            "PM2.5 (µg/m³)",
    "pm10_ugm3":             "PM10 (µg/m³)",
    "temperature_2m_c":      "Suhu (°C)",
    "relative_humidity_pct": "Kelembaban (%)",
    "wind_speed_10m_kmh":    "Angin (km/h)",
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

if not anomalies.empty:
    st.divider()
    extreme = anomalies[anomalies["anomaly_score"] > 0.5]
    if not extreme.empty:
        st.error(
            f"🚨 {len(extreme)} anomali **ekstrem** terdeteksi di {city.capitalize()} — "
            f"PM2.5 tertinggi: {extreme['pm2_5_ugm3'].max():.1f} µg/m³"
        )
    elif len(anomalies) > 0:
        st.warning(f"⚠️ {len(anomalies)} anomali terdeteksi di {city.capitalize()}.")

    def _severity(score: float) -> str:
        if score > 0.5:   return "🔴 Ekstrem"
        elif score > 0.3: return "🟠 Tinggi"
        else:             return "🟡 Sedang"

    display_anom = (
        anomalies
        .sort_values("timestamp", ascending=False)
        .assign(severity=lambda df: df["anomaly_score"].map(_severity))
        .rename(columns={
            "timestamp":     "Waktu",
            "pm2_5_ugm3":   "PM2.5 (µg/m³)",
            "anomaly_score": "Anomaly Score",
            "severity":      "Tingkat",
        })
    )
    st.subheader(f"Anomali Terdeteksi ({len(anomalies)} titik)")
    st.dataframe(display_anom, use_container_width=True, hide_index=True)

# Auto-refresh every 5 minutes via browser meta-refresh (non-blocking)
components.html('<meta http-equiv="refresh" content="300">', height=0)
