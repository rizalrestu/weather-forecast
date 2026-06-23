"""Fetch current weather + air quality from open-meteo for all target cities."""
import json
import requests
from datetime import datetime
from pathlib import Path

CITIES = {
    "jakarta":  {"latitude": -6.2088,  "longitude": 106.8456},
    "surabaya": {"latitude": -7.2575,  "longitude": 112.7521},
    "bandung":  {"latitude": -6.9175,  "longitude": 107.6191},
    "medan":    {"latitude":  3.5952,  "longitude":  98.6722},
    "makassar": {"latitude": -5.1477,  "longitude": 119.4327},
}

WEATHER_URL     = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_weather(lat: float, lon: float, past_days: int = 7) -> dict:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone":   "Asia/Jakarta",
        "past_days":  past_days,
        "forecast_days": 1,
    }
    resp = requests.get(WEATHER_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_air_quality(lat: float, lon: float, past_days: int = 7) -> dict:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
        "timezone":   "Asia/Jakarta",
        "past_days":  past_days,
        "forecast_days": 1,
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_city_records(city: str, coords: dict, weather: dict, aq: dict) -> list[dict]:
    """Parse weather + AQ API responses for one city into hourly record dicts."""
    w, a    = weather["hourly"], aq["hourly"]
    min_len = min(len(w["time"]), len(a["time"]))
    return [
        {
            "timestamp":             w["time"][i],
            "city":                  city,
            "latitude":              coords["latitude"],
            "longitude":             coords["longitude"],
            "temperature_2m_c":      w["temperature_2m"][i],
            "relative_humidity_pct": w["relative_humidity_2m"][i],
            "precipitation_mm":      w["precipitation"][i],
            "wind_speed_10m_kmh":    w["wind_speed_10m"][i],
            "pm2_5_ugm3":            a["pm2_5"][i],
            "pm10_ugm3":             a["pm10"][i],
            "carbon_monoxide_ugm3":  a["carbon_monoxide"][i],
            "nitrogen_dioxide_ugm3": a["nitrogen_dioxide"][i],
        }
        for i in range(min_len)
    ]


def fetch_all_cities(past_days: int = 7, raw_dir: Path = None) -> list[dict]:
    """
    Fetch recent weather + AQ for all cities.
    Returns list of dicts (XCom-serializable for Airflow).
    past_days=7 gives enough history to compute the 168h lag features.
    """
    if raw_dir:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for city, coords in CITIES.items():
        lat, lon = coords["latitude"], coords["longitude"]
        weather  = fetch_weather(lat, lon, past_days=past_days)
        aq       = fetch_air_quality(lat, lon, past_days=past_days)

        if raw_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            (raw_dir / f"{city}_weather_{ts}.json").write_text(
                json.dumps(weather, indent=2), encoding="utf-8"
            )
            (raw_dir / f"{city}_airquality_{ts}.json").write_text(
                json.dumps(aq, indent=2), encoding="utf-8"
            )

        records.extend(build_city_records(city, coords, weather, aq))

    return records
