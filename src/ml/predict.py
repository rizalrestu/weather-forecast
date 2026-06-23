"""Load trained model and run inference on new raw data."""
import os
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from src.transform.features import build_features, FEATURE_COLS

# Supports MODEL_PATH env var so Docker and local both resolve correctly
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/xgb_pm25.pkl"))


def load_model(path: Path = DEFAULT_MODEL_PATH) -> XGBRegressor:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run `python -m src.ml.train` first."
        )
    return joblib.load(path)


def predict(model: XGBRegressor, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering + predict on raw dataframe. For one-shot / notebook use."""
    df_feat = build_features(df_raw)
    return predict_from_features(model, df_feat)


def predict_from_features(model: XGBRegressor, df_feat: pd.DataFrame) -> pd.DataFrame:
    """Predict on already-feature-engineered dataframe. Used by Airflow DAG tasks."""
    preds = model.predict(df_feat[FEATURE_COLS])
    out   = df_feat[["timestamp", "city", "pm2_5_ugm3"]].copy()
    out["predicted_pm25"] = preds.round(2)
    return out
