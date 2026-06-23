"""
Retrain DAG — weekly retraining of XGBoost + Isolation Forest.

Data source: aqi_readings table (Postgres), not API or CSV.
Runs in parallel: retrain_xgb and retrain_isolation_forest after fetch_data.

    fetch_data → retrain_xgb
               ↘ retrain_isolation_forest
"""
import pendulum
from airflow.decorators import dag, task


@dag(
    dag_id="retrain_pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 6, 22, tz="Asia/Jakarta"),
    catchup=False,
    tags=["retrain", "mlflow"],
)
def retrain_pipeline():

    @task()
    def fetch_data() -> list:
        """
        Pull training data for retraining.
        Primary source: historical CSV (full history).
        Supplement: recent rows from aqi_readings not yet in CSV.
        """
        import os
        import psycopg2
        import pandas as pd
        from pathlib import Path

        CSV_PATH = Path("/opt/airflow/data/processed/historical_data.csv")
        db_url   = os.getenv("AQI_DB_URL", "postgresql://airflow:airflow@postgres/aqi_db")

        # Load historical CSV
        df_hist = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]) if CSV_PATH.exists() else pd.DataFrame()
        print(f"CSV rows: {len(df_hist)}")

        # Load recent DB rows not covered by CSV
        sql = "SELECT * FROM aqi_readings ORDER BY city, timestamp"
        with psycopg2.connect(db_url) as conn:
            df_db = pd.read_sql_query(sql, conn)
        df_db["timestamp"] = pd.to_datetime(df_db["timestamp"], utc=True).dt.tz_convert(None)
        print(f"DB rows: {len(df_db)}")

        # Merge and deduplicate
        keep_cols = ["timestamp", "city", "latitude", "longitude",
                     "temperature_2m_c", "relative_humidity_pct",
                     "precipitation_mm", "wind_speed_10m_kmh",
                     "pm2_5_ugm3", "pm10_ugm3",
                     "carbon_monoxide_ugm3", "nitrogen_dioxide_ugm3"]
        df_db = df_db[[c for c in keep_cols if c in df_db.columns]]

        df = pd.concat([df_hist, df_db], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp", "city"]).sort_values(["city", "timestamp"])
        print(f"Total rows after merge: {len(df)} ({df['city'].nunique()} cities)")

        df["timestamp"] = df["timestamp"].astype(str)
        return df.to_dict(orient="records")

    @task()
    def retrain_xgb(records: list) -> None:
        """Retrain XGBoost PM2.5 model and log run to MLflow."""
        import os
        import numpy as np
        import pandas as pd
        import mlflow
        from pathlib import Path
        from src.transform.features import build_features, FEATURE_COLS, TARGET_COL
        from src.ml.train import train_model, evaluate, save_model

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_feat = build_features(df)

        # Use last 30 days as test set
        split_date = df_feat["timestamp"].max() - pd.Timedelta(days=30)
        train = df_feat[df_feat["timestamp"] < split_date]
        test  = df_feat[df_feat["timestamp"] >= split_date]

        X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
        X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | split: {split_date.date()}")

        model_path = Path("/opt/airflow/models/xgb_pm25.pkl")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("pm25_xgboost")

        with mlflow.start_run(run_name="weekly_retrain"):
            model = train_model(X_train, y_train)

            naive_metrics = evaluate("Naive baseline", y_test, X_test["pm2_5_lag_24h"])
            xgb_metrics   = evaluate("XGBoost",        y_test, model.predict(X_test))

            mlflow.log_params({
                "n_estimators":   500,
                "learning_rate":  0.05,
                "max_depth":      6,
                "subsample":      0.8,
                "colsample_bytree": 0.8,
                "split_date":     str(split_date.date()),
                "train_rows":     len(X_train),
                "test_rows":      len(X_test),
            })
            mlflow.log_metrics({
                "mape":       xgb_metrics["mape"],
                "mae":        xgb_metrics["mae"],
                "rmse":       xgb_metrics["rmse"],
                "naive_mape": naive_metrics["mape"],
            })
            save_model(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        print(f"XGBoost retrain done. MAPE={xgb_metrics['mape']:.2f}%")

    @task()
    def retrain_isolation_forest(records: list) -> None:
        """Retrain Isolation Forest anomaly model and log run to MLflow."""
        import os
        import pandas as pd
        import mlflow
        from pathlib import Path
        from src.transform.features import build_features
        from src.ml.anomaly import train, detect, save_model, ANOMALY_FEATURE_COLS

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_feat = build_features(df)

        model_path = Path("/opt/airflow/models/isolation_forest.pkl")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("pm25_isolation_forest")

        with mlflow.start_run(run_name="weekly_retrain"):
            model = train(df_feat)

            result       = detect(model, df_feat)
            n_anomalies  = int(result["is_anomaly"].sum())
            anomaly_rate = n_anomalies / len(result)

            mlflow.log_params({
                "n_estimators": 200,
                "contamination": 0.05,
                "n_features":   len(ANOMALY_FEATURE_COLS),
                "train_rows":   len(df_feat),
            })
            mlflow.log_metric("anomaly_rate", float(anomaly_rate))
            save_model(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        print(f"Isolation Forest retrain done. Anomaly rate={anomaly_rate*100:.1f}%")

    records = fetch_data()
    retrain_xgb(records)
    retrain_isolation_forest(records)


retrain_pipeline()
