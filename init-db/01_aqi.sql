-- Create AQI database (separate from Airflow metadata DB)
CREATE DATABASE aqi_db;

-- Create MLflow database (separate from Airflow + AQI to avoid Alembic conflicts)
CREATE DATABASE mlflow;

\connect aqi_db;

CREATE TABLE IF NOT EXISTS aqi_readings (
    id                    SERIAL PRIMARY KEY,
    timestamp             TIMESTAMPTZ NOT NULL,
    city                  VARCHAR(50)  NOT NULL,
    latitude              FLOAT,
    longitude             FLOAT,
    temperature_2m_c      FLOAT,
    relative_humidity_pct FLOAT,
    precipitation_mm      FLOAT,
    wind_speed_10m_kmh    FLOAT,
    pm2_5_ugm3            FLOAT,
    pm10_ugm3             FLOAT,
    carbon_monoxide_ugm3  FLOAT,
    nitrogen_dioxide_ugm3 FLOAT,
    ingested_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (timestamp, city)
);

CREATE TABLE IF NOT EXISTS aqi_predictions (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    city            VARCHAR(50)  NOT NULL,
    actual_pm25     FLOAT,
    predicted_pm25  FLOAT,
    predicted_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (timestamp, city)
);

CREATE TABLE IF NOT EXISTS aqi_anomalies (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    city            VARCHAR(50)  NOT NULL,
    pm2_5_ugm3      FLOAT,
    anomaly_score   FLOAT,
    is_anomaly      BOOLEAN,
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (timestamp, city)
);

CREATE TABLE IF NOT EXISTS aqi_forecasts (
    id               SERIAL PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL,
    city             VARCHAR(50) NOT NULL,
    predicted_pm25   FLOAT,
    forecast_made_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (timestamp, city)
);
