"""Feature engineering pipeline — used by both training and inference."""
import pandas as pd

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_2m_c", "relative_humidity_pct", "wind_speed_10m_kmh",
    "pm2_5_lag_1h", "pm2_5_lag_24h", "pm2_5_lag_168h", "no2_lag_1h",
    "pm2_5_roll_mean_3h", "pm2_5_roll_mean_24h", "pm2_5_roll_mean_7d", "pm2_5_roll_std_24h",
    "city_encoded",
]
TARGET_COL = "pm2_5_ugm3"

# Single source of truth for city encoding — import this in forecast.py and anywhere else needed
CITY_ENCODING = {"jakarta": 0, "surabaya": 1, "bandung": 2, "medan": 3, "makassar": 4}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering to raw hourly dataframe.

    Input : raw df with columns — timestamp, city, weather cols, pollutant cols.
    Output: df with feature columns added, NaN rows dropped (from lag 168h).

    Groupby-city for lags/rolling prevents cross-city leakage.
    """
    df = df.copy()
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Lag features — groupby city prevents cross-city leakage
    grp = df.groupby("city")
    df["pm2_5_lag_1h"]   = grp["pm2_5_ugm3"].shift(1)
    df["pm2_5_lag_24h"]  = grp["pm2_5_ugm3"].shift(24)
    df["pm2_5_lag_168h"] = grp["pm2_5_ugm3"].shift(168)
    df["no2_lag_1h"]     = grp["nitrogen_dioxide_ugm3"].shift(1)

    def _roll(col: str, window: int, func: str = "mean", min_periods: int = 1):
        return df.groupby("city")[col].transform(
            lambda x: getattr(x.rolling(window, min_periods=min_periods), func)()
        )

    df["pm2_5_roll_mean_3h"]  = _roll("pm2_5_ugm3", 3)
    df["pm2_5_roll_mean_24h"] = _roll("pm2_5_ugm3", 24)
    df["pm2_5_roll_mean_7d"]  = _roll("pm2_5_ugm3", 168)
    df["pm2_5_roll_std_24h"]  = _roll("pm2_5_ugm3", 24, func="std", min_periods=2)

    df["city_encoded"] = df["city"].map(CITY_ENCODING).fillna(-1).astype(int)

    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
