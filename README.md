# AQI & PM2.5 Forecasting Pipeline

An end-to-end machine learning pipeline that ingests hourly air quality and weather data for five major Indonesian cities, generates 24-hour PM2.5 forecasts using XGBoost, detects anomalous readings with Isolation Forest, and presents results on a live Streamlit dashboard — all orchestrated by Apache Airflow and running locally via Docker Compose.

---

## Features

- **Automated ETL** — Airflow DAG fetches weather + AQI data from [open-meteo](https://open-meteo.com/) every hour for Jakarta, Surabaya, Bandung, Medan, and Makassar
- **16-feature engineering** — lag features (1h / 24h / 168h), rolling windows (3h / 24h / 7d), calendar features, and meteorological variables
- **XGBoost forecasting** — recursive 24-hour PM2.5 prediction (MAPE ≈ 7.87%)
- **Isolation Forest anomaly detection** — unsupervised, multi-feature anomaly scoring with severity labels
- **MLflow experiment tracking** — every retrain logs metrics, parameters, and model artifacts
- **Streamlit dashboard** — real-time charts for actual vs. predicted PM2.5, 24h forecast, and anomaly highlights
- **Fully containerized** — one `docker compose up` starts everything

## Architecture

```text
open-meteo API
      │  hourly HTTP
      ▼
┌─────────────────────────────────────────────────┐
│ Airflow  aqi_pipeline DAG                       │
│ extract → transform → load → predict            │
│                         └──→ detect_anomaly     │
│                         └──→ run_forecast       │
└──────────────────┬──────────────────────────────┘
                   │  upsert
                   ▼
             PostgreSQL
   aqi_readings / aqi_predictions
   aqi_anomalies / aqi_forecasts
                   │  SELECT
          ┌────────┴─────────┐
          ▼                  ▼
    Streamlit             MLflow
    Dashboard          Experiment
   (port 8501)         Tracking
                       (port 5000)
```

A second DAG (`retrain_pipeline`) runs weekly, retraining both models from the accumulated historical data.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Apache Airflow 2.9 |
| Database | PostgreSQL 15 |
| Forecasting | XGBoost |
| Anomaly Detection | Isolation Forest (scikit-learn) |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker Compose |
| Data Source | open-meteo (free, no API key) |

## Quick Start

**Prerequisites:** Docker Desktop (8 GB RAM, 15 GB disk), Python 3.11, `uv`

```bash
# 1. Create required directories
mkdir -p data/raw data/processed models

# 2. Backfill historical data
python notebooks/historical_backfill.py
python scripts/backfill_new_cities.py

# 3. Start all services
docker compose up -d --build

# 4. Train models (requires stack to be running for MLflow)
python -m src.ml.train
python -m src.ml.anomaly

# 5. Trigger the first pipeline run in Airflow UI
open http://localhost:8080   # admin / admin
```

| Service | URL |
|---------|-----|
| Airflow UI | <http://localhost:8080> |
| Dashboard | <http://localhost:8501> |
| MLflow UI | <http://localhost:5000> |

For the full setup guide, maintenance instructions, and design decisions, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Environment Variables

Copy `.env.example` to `.env` before running scripts locally:

```bash
cp .env.example .env
```

No API key is required — open-meteo is fully free. Database credentials default to `airflow/airflow` (matching the Docker Compose config).

## Project Structure

```text
├── dags/                   # Airflow DAGs (pipeline + retrain)
├── src/
│   ├── ingestion/          # API fetch + record builder
│   ├── transform/          # Feature engineering + DB writes
│   ├── ml/                 # Train, predict, anomaly, forecast
│   └── serve/              # Streamlit dashboard
├── notebooks/              # EDA + historical backfill script
├── scripts/                # One-time data utilities
├── init-db/                # PostgreSQL init SQL
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
