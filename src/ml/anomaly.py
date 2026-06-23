"""
Isolation Forest anomaly detection for AQI readings.
Train offline: python -m src.ml.anomaly
"""
import os   
import pandas as pd
import joblib
import mlflow
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Includes actual PM2.5 so the model detects spikes relative to recent history + weather context
ANOMALY_FEATURE_COLS = [
    "hour", "month", "city_encoded",
    "temperature_2m_c", "relative_humidity_pct", "wind_speed_10m_kmh",
    "pm2_5_lag_1h", "pm2_5_lag_24h", "pm2_5_lag_168h",
    "pm2_5_roll_mean_24h", "pm2_5_roll_std_24h",
    "pm2_5_ugm3",
]

DEFAULT_IF_PATH = Path("models/isolation_forest.pkl")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def train(df: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    """Train Isolation Forest on historical feature data."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(df[ANOMALY_FEATURE_COLS])
    return model


def save_model(model: IsolationForest, path: Path = DEFAULT_IF_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Anomaly model saved -> {path}")


def load_anomaly_model(path: Path = DEFAULT_IF_PATH) -> IsolationForest:
    if not path.exists():
        raise FileNotFoundError(
            f"Anomaly model not found at '{path}'. "
            "Run `python -m src.ml.anomaly` to train it first."
        )
    return joblib.load(path)


def detect(model: IsolationForest, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run anomaly detection on feature-engineered dataframe.
    Returns df with anomaly_score (higher = more anomalous) and is_anomaly columns.
    """
    scores = model.decision_function(df[ANOMALY_FEATURE_COLS])
    preds  = model.predict(df[ANOMALY_FEATURE_COLS])

    out = df[["timestamp", "city", "pm2_5_ugm3"]].copy()
    out["anomaly_score"] = (-scores).round(4)
    out["is_anomaly"]    = preds == -1
    return out


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("pm25_isolation_forest")

    with mlflow.start_run():
        from src.transform.features import build_features

        DATA_PATH = Path("data/processed/historical_data.csv")
        print(f"Loading {DATA_PATH} ...")
        df_raw  = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        df_feat = build_features(df_raw)

        print(f"Training Isolation Forest on {len(df_feat)} rows, {len(ANOMALY_FEATURE_COLS)} features ...")
        model = train(df_feat)
        result      = detect(model, df_feat)
        n_anomalies = result["is_anomaly"].sum()
        print(f"\nAnomalies in training data: {n_anomalies} / {len(result)} ({n_anomalies/len(result)*100:.1f}%)")
        print("\nTop 5 most anomalous readings:")
        top = result.nlargest(5, "anomaly_score")[["timestamp", "city", "pm2_5_ugm3", "anomaly_score"]]
        print(top.to_string(index=False))

        mlflow.log_params({
            "n_estimators": 200,
            "contamination": 0.05,
            "max_samples": "auto",
            "random_state": 42,
            "n_jobs": -1,
            "n_features": len(ANOMALY_FEATURE_COLS),
        })

        mlflow.log_metric("anomaly_rate", float(n_anomalies / len(result)))
        save_model(model)
        mlflow.log_artifact(str(DEFAULT_IF_PATH), artifact_path="model")
