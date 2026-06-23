"""Validate incoming data and write to PostgreSQL."""
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

AQI_DB_URL = os.getenv("AQI_DB_URL", "postgresql://airflow:airflow@localhost/aqi_db")

_READINGS_COLS    = [
    "timestamp", "city", "latitude", "longitude",
    "temperature_2m_c", "relative_humidity_pct", "precipitation_mm", "wind_speed_10m_kmh",
    "pm2_5_ugm3", "pm10_ugm3", "carbon_monoxide_ugm3", "nitrogen_dioxide_ugm3",
]
_PREDICTIONS_COLS = ["timestamp", "city", "actual_pm25", "predicted_pm25"]
_ANOMALIES_COLS   = ["timestamp", "city", "pm2_5_ugm3", "anomaly_score", "is_anomaly"]
_FORECASTS_COLS   = ["timestamp", "city", "predicted_pm25"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in critical columns and obvious out-of-range values."""
    df = df.dropna(subset=["timestamp", "city", "pm2_5_ugm3"])
    df = df[df["pm2_5_ugm3"] >= 0]
    df = df[df["temperature_2m_c"].between(-10, 55)]
    return df.reset_index(drop=True)


def _upsert(sql: str, rows: list) -> None:
    with psycopg2.connect(AQI_DB_URL) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)


def write_readings(df: pd.DataFrame) -> int:
    """Upsert clean hourly readings into aqi_readings. Returns row count."""
    df   = validate(df)
    rows = df[_READINGS_COLS].values.tolist()
    _upsert(
        f"INSERT INTO aqi_readings ({', '.join(_READINGS_COLS)}) VALUES %s "
        "ON CONFLICT (timestamp, city) DO NOTHING",
        rows,
    )
    return len(rows)


def write_predictions(df: pd.DataFrame) -> int:
    """Upsert prediction results into aqi_predictions. Returns row count."""
    rows = df.rename(columns={"pm2_5_ugm3": "actual_pm25"})[_PREDICTIONS_COLS].values.tolist()
    _upsert(
        f"INSERT INTO aqi_predictions ({', '.join(_PREDICTIONS_COLS)}) VALUES %s "
        "ON CONFLICT (timestamp, city) DO UPDATE "
        "SET predicted_pm25 = EXCLUDED.predicted_pm25, predicted_at = NOW()",
        rows,
    )
    return len(rows)


def write_forecasts(df: pd.DataFrame) -> int:
    """Upsert 24h recursive forecast results into aqi_forecasts. Returns row count."""
    rows = df[_FORECASTS_COLS].values.tolist()
    _upsert(
        f"INSERT INTO aqi_forecasts ({', '.join(_FORECASTS_COLS)}) VALUES %s "
        "ON CONFLICT (timestamp, city) DO UPDATE "
        "SET predicted_pm25 = EXCLUDED.predicted_pm25, forecast_made_at = NOW()",
        rows,
    )
    return len(rows)


def write_anomalies(df: pd.DataFrame) -> int:
    """Upsert anomaly detection results into aqi_anomalies. Returns row count."""
    rows = df[_ANOMALIES_COLS].values.tolist()
    _upsert(
        f"INSERT INTO aqi_anomalies ({', '.join(_ANOMALIES_COLS)}) VALUES %s "
        "ON CONFLICT (timestamp, city) DO UPDATE "
        "SET anomaly_score = EXCLUDED.anomaly_score, "
        "is_anomaly = EXCLUDED.is_anomaly, detected_at = NOW()",
        rows,
    )
    return len(rows)
