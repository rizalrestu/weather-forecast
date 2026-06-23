"""
Train XGBoost PM2.5 forecasting model on historical data.

Run from project root:
    python -m src.ml.train
"""
import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.transform.features import build_features, FEATURE_COLS, TARGET_COL

DATA_PATH  = Path("data/processed/historical_data.csv")
MODEL_PATH = Path("models/xgb_pm25.pkl")
SPLIT_DATE = "2026-06-01"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def split_train_test(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = df[df["timestamp"] < split_date]
    test  = df[df["timestamp"] >= split_date]
    return (
        train[FEATURE_COLS], train[TARGET_COL],
        test[FEATURE_COLS],  test[TARGET_COL],
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    print(f"  {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_model(model: XGBRegressor, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved -> {path}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("pm25_xgboost")
    
    with mlflow.start_run():
        print("Loading data ...")
        df_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

        print("Building features ...")
        df_feat = build_features(df_raw)
        print(f"  {len(df_feat)} rows, {len(FEATURE_COLS)} features")

        print(f"Splitting at {SPLIT_DATE} ...")
        X_train, y_train, X_test, y_test = split_train_test(df_feat, SPLIT_DATE)
        print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        print("Training XGBoost ...")
        model = train_model(X_train, y_train)

        print("\nEvaluation:")
        naive_pred = X_test["pm2_5_lag_24h"]
        evaluate("Naive baseline", y_test, naive_pred)

        mlflow.log_params({
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "split_date": SPLIT_DATE,
        })

        metrics = evaluate("XGBoost", y_test, model.predict(X_test))
        mlflow.log_metrics({"mape": metrics["mape"], "mae": metrics["mae"]})

        save_model(model)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")