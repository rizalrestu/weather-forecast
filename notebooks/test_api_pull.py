"""
Step 1: manual test pull from open-meteo APIs.
Run once to verify both endpoints respond and data looks sensible.

Usage: python notebooks/test_api_pull.py
"""
import json
import requests
from datetime import datetime
from pathlib import Path

CITIES = {
    "jakarta":   {"latitude": -6.2088, "longitude": 106.8456},
    "surabaya":  {"latitude": -7.2575, "longitude": 112.7521},
}

WEATHER_URL     = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch 1-day hourly weather forecast from open-meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "Asia/Jakarta",
        "forecast_days": 1,
    }
    resp = requests.get(WEATHER_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_air_quality(lat: float, lon: float) -> dict:
    """Fetch 1-day hourly air quality from open-meteo Air Quality API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
        "timezone": "Asia/Jakarta",
        "forecast_days": 1,
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def save_raw(city: str, data_type: str, payload: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{city}_{data_type}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_summary(city: str, weather: dict, aq: dict) -> None:
    temps  = [t for t in weather.get("hourly", {}).get("temperature_2m", []) if t is not None]
    pm25   = [v for v in aq.get("hourly", {}).get("pm2_5", []) if v is not None]
    pm10   = [v for v in aq.get("hourly", {}).get("pm10", []) if v is not None]

    print(f"\n--- {city.upper()} ---")
    if temps:
        print(f"  Temp range   : {min(temps):.1f}°C – {max(temps):.1f}°C")
    if pm25:
        print(f"  PM2.5 range  : {min(pm25):.1f} – {max(pm25):.1f} µg/m³")
    if pm10:
        print(f"  PM10 range   : {min(pm10):.1f} – {max(pm10):.1f} µg/m³")
    print(f"  Data points  : {len(temps)} hourly weather, {len(pm25)} hourly AQ")


if __name__ == "__main__":
    print(f"open-meteo test pull — {datetime.now().isoformat()}\n")

    for city, coords in CITIES.items():
        lat, lon = coords["latitude"], coords["longitude"]
        print(f"Fetching {city} (lat={lat}, lon={lon}) ...")

        weather = fetch_weather(lat, lon)
        aq      = fetch_air_quality(lat, lon)

        w_path  = save_raw(city, "weather", weather)
        aq_path = save_raw(city, "airquality", aq)

        print(f"  Saved: {w_path.name}")
        print(f"  Saved: {aq_path.name}")
        print_summary(city, weather, aq)

    print("\nDone. Check data/raw/ for raw JSON files.")
