"""
AQI Pipeline DAG — hourly extract → transform → load → predict.
Uses TaskFlow API (@task decorator style).

Airflow UI: http://localhost:8080  (admin / admin)
"""
import pendulum
from airflow.decorators import dag, task


@dag(
    dag_id="aqi_pipeline",
    schedule="@hourly",
    start_date=pendulum.datetime(2026, 6, 22, tz="Asia/Jakarta"),
    catchup=False,
    tags=["aqi", "weather", "forecast"],
    doc_md="""
    ## AQI Pipeline
    Pulls weather + air quality data from open-meteo for all 5 cities
    (Jakarta, Surabaya, Bandung, Medan, Makassar), builds lag/rolling features,
    stores clean data to Postgres, then runs XGBoost PM2.5 forecast and stores predictions.
    """,
)
def aqi_pipeline():

    @task()
    def extract() -> list:
        """Fetch last 7 days + current from open-meteo for all cities."""
        from pathlib import Path
        from src.ingestion.extract_api import fetch_all_cities
        raw_dir = Path("/opt/airflow/data/raw")
        records = fetch_all_cities(past_days=7, raw_dir=raw_dir)
        print(f"Extracted {len(records)} records across all cities.")
        return records

    @task()
    def transform(raw_records: list) -> list:
        """Build lag + rolling features. Returns feature records for recent 24h."""
        import pandas as pd
        from src.transform.features import build_features, FEATURE_COLS

        df = pd.DataFrame(raw_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        feat_df = build_features(df)

        # Keep only most recent 24h for prediction output
        cutoff = feat_df["timestamp"].max() - pd.Timedelta(hours=23)
        recent = feat_df[feat_df["timestamp"] >= cutoff].copy()

        keep_cols = ["timestamp", "city", "latitude", "longitude",
                     "pm2_5_ugm3", "pm10_ugm3", "carbon_monoxide_ugm3",
                     "nitrogen_dioxide_ugm3", "temperature_2m_c",
                     "relative_humidity_pct", "precipitation_mm",
                     "wind_speed_10m_kmh"] + FEATURE_COLS
        # deduplicate keep_cols while preserving order
        seen = set()
        keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

        print(f"Transform: {len(recent)} rows ready.")
        recent["timestamp"] = recent["timestamp"].astype(str)
        return recent[keep_cols].to_dict(orient="records")

    @task()
    def load(feat_records: list) -> list:
        """Validate and write clean readings to aqi_readings table."""
        import pandas as pd
        from src.transform.clean import write_readings

        df = pd.DataFrame(feat_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        n = write_readings(df)
        print(f"Loaded {n} rows into aqi_readings.")
        return feat_records  # pass through for predict task

    @task()
    def predict(feat_records: list) -> None:
        """Run XGBoost predictions and store results in aqi_predictions."""
        import pandas as pd
        from pathlib import Path
        from src.ml.predict import load_model, predict_from_features
        from src.transform.clean import write_predictions

        model_path = Path("/opt/airflow/models/xgb_pm25.pkl")
        model      = load_model(model_path)

        df = pd.DataFrame(feat_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        results = predict_from_features(model, df)

        n = write_predictions(results)
        print(f"Stored {n} predictions in aqi_predictions.")

        print("\n--- PM2.5 Predictions ---")
        for _, row in results.iterrows():
            diff = row["predicted_pm25"] - row["pm2_5_ugm3"]
            sign = "+" if diff >= 0 else ""
            print(
                f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                f"{row['city']:10} | "
                f"actual={row['pm2_5_ugm3']:6.1f} | "
                f"pred={row['predicted_pm25']:6.1f} | "
                f"diff={sign}{diff:.1f}"
            )

    @task()
    def detect_anomaly(feat_records: list) -> None:
        """Detect anomalies using Isolation Forest and store results in aqi_anomalies."""
        import pandas as pd
        from pathlib import Path
        from src.ml.anomaly import load_anomaly_model, detect
        from src.transform.clean import write_anomalies

        model_path = Path("/opt/airflow/models/isolation_forest.pkl")
        model      = load_anomaly_model(model_path)

        df = pd.DataFrame(feat_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        results = detect(model, df)

        n = write_anomalies(results)
        print(f"Stored {n} anomaly records in aqi_anomalies.")

        n_flagged = results["is_anomaly"].sum()
        print(f"\n--- Anomaly Summary: {n_flagged}/{len(results)} flagged ---")
        if n_flagged > 0:
            flagged = results[results["is_anomaly"]].sort_values("anomaly_score", ascending=False)
            for _, row in flagged.iterrows():
                print(
                    f"[ANOMALY] {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                    f"{row['city']:10} | "
                    f"PM2.5={row['pm2_5_ugm3']:6.1f} µg/m³ | "
                    f"score={row['anomaly_score']:.4f}"
                )

    @task()
    def run_forecast(raw_records: list, _loaded: list) -> None:
        """Recursive 24h PM2.5 forecast using XGBoost + open-meteo weather forecast."""
        import pandas as pd
        from pathlib import Path
        from src.ml.predict import load_model
        from src.ml.forecast import forecast_24h
        from src.transform.clean import write_forecasts

        model = load_model(Path("/opt/airflow/models/xgb_pm25.pkl"))
        df    = pd.DataFrame(raw_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_results = []
        for city in df["city"].unique():
            city_df = df[df["city"] == city].copy()
            result  = forecast_24h(model, city_df)
            all_results.append(result)
            print(f"{city}: forecast t+1={result['predicted_pm25'].iloc[0]:.1f} "
                  f"t+24={result['predicted_pm25'].iloc[-1]:.1f} µg/m³")

        df_all = pd.concat(all_results, ignore_index=True)
        n = write_forecasts(df_all)
        print(f"Stored {n} forecast rows in aqi_forecasts.")

    raw_data   = extract()
    feat_data  = transform(raw_data)
    clean_data = load(feat_data)
    predict(clean_data)
    detect_anomaly(clean_data)
    run_forecast(raw_data, clean_data)


aqi_pipeline()
