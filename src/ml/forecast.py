"""
Recursive 24-hour PM2.5 forecast using XGBoost.

Strategy:
  - open-meteo returns past 7 days + next 24h weather in one response.
  - We use the 7-day history as the lag/rolling context buffer.
  - For each future hour (t+1 to t+24), we pull weather features from the
    API forecast and compute PM2.5 lags from previously predicted values.
  - Error accumulates over 24 steps — acceptable for a 24h horizon at hourly res.
"""
import pandas as pd
from xgboost import XGBRegressor

from src.transform.features import FEATURE_COLS, CITY_ENCODING


def forecast_24h(model: XGBRegressor, df_city: pd.DataFrame) -> pd.DataFrame:
    """
    Recursive 24h PM2.5 forecast for a single city.

    df_city: raw hourly records for ONE city, as returned by fetch_all_cities().
             Must contain both historical rows and the 24 future weather rows
             from open-meteo (forecast_days=1).
    Returns: DataFrame with columns [timestamp, city, predicted_pm25] — 24 rows.
    """
    df   = df_city.sort_values("timestamp").reset_index(drop=True)
    city         = df["city"].iloc[0]
    city_encoded = CITY_ENCODING.get(city, -1)

    all_ts = df["timestamp"]
    # Last 24 rows = open-meteo weather forecast; row -25 = last confirmed actual hour
    now            = all_ts.iloc[-25]
    history        = df[df["timestamp"] <= now].copy()
    future_weather = df[df["timestamp"] > now].copy()

    pm25_buf = history["pm2_5_ugm3"].tolist()
    no2_buf  = history["nitrogen_dioxide_ugm3"].tolist()

    def _get(buf: list, n: int) -> float:
        return buf[-n] if len(buf) >= n else buf[-1]

    def _mean(buf: list, n: int) -> float:
        window = buf[-n:]
        return sum(window) / len(window)

    def _std(buf: list, n: int) -> float:
        window = buf[-n:]
        if len(window) < 2:
            return 0.0
        m = sum(window) / len(window)
        # sample std (ddof=1) — matches pandas .std() used during training
        return (sum((x - m) ** 2 for x in window) / (len(window) - 1)) ** 0.5

    results = []
    for h in range(1, 25):
        ts  = now + pd.Timedelta(hours=h)
        row = future_weather[future_weather["timestamp"] == ts]

        if row.empty:
            # Fallback: carry last known weather forward
            temp  = history["temperature_2m_c"].iloc[-1]
            humid = history["relative_humidity_pct"].iloc[-1]
            wind  = history["wind_speed_10m_kmh"].iloc[-1]
        else:
            temp  = float(row["temperature_2m_c"].iloc[0])
            humid = float(row["relative_humidity_pct"].iloc[0])
            wind  = float(row["wind_speed_10m_kmh"].iloc[0])

        feat = pd.DataFrame([{
            "hour":                  ts.hour,
            "day_of_week":           ts.dayofweek,
            "month":                 ts.month,
            "is_weekend":            int(ts.dayofweek >= 5),
            "temperature_2m_c":      temp,
            "relative_humidity_pct": humid,
            "wind_speed_10m_kmh":    wind,
            "pm2_5_lag_1h":          _get(pm25_buf, 1),
            "pm2_5_lag_24h":         _get(pm25_buf, 24),
            "pm2_5_lag_168h":        _get(pm25_buf, 168),
            "no2_lag_1h":            _get(no2_buf, 1),
            "pm2_5_roll_mean_3h":    _mean(pm25_buf, 3),
            "pm2_5_roll_mean_24h":   _mean(pm25_buf, 24),
            "pm2_5_roll_mean_7d":    _mean(pm25_buf, 168),
            "pm2_5_roll_std_24h":    _std(pm25_buf, 24),
            "city_encoded":          city_encoded,
        }])

        pred = max(0.0, round(float(model.predict(feat[FEATURE_COLS])[0]), 2))
        results.append({"timestamp": ts, "city": city, "predicted_pm25": pred})

        pm25_buf.append(pred)
        no2_buf.append(no2_buf[-1])  # carry last NO2 forward

    return pd.DataFrame(results)
