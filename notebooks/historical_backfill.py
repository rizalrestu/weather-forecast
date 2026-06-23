"""
Step 2: historical backfill from open-meteo (2026-01-01 to today).
Pulls hourly weather + air quality for Jakarta & Surabaya.
Outputs: raw JSON in data/raw/, merged CSV in data/processed/, and a Google Sheet.

Setup before running:
  1. Create a blank Google Spreadsheet.
  2. Share it with your service account email (Editor access).
  3. Copy .env.example -> .env and fill in the two variables.

Usage: python notebooks/historical_backfill.py
"""
import json
import os
import sys
import requests
import pandas as pd
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.ingestion.extract_api import CITIES  # noqa: E402

START_DATE      = "2026-01-01"
END_DATE        = date.today().isoformat()
ARCHIVE_URL     = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
CSV_DIR  = Path(__file__).parent.parent / "data" / "processed"

GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_TAB     = "historical"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_historical_weather(lat: float, lon: float) -> dict:
    """Fetch hourly weather from open-meteo archive API for the full date range."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "hourly":     "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone":   "Asia/Jakarta",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_historical_air_quality(lat: float, lon: float) -> dict:
    """Fetch hourly air quality from open-meteo for the full date range."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "hourly":     "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
        "timezone":   "Asia/Jakarta",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def build_dataframe(city: str, coords: dict, weather: dict, aq: dict) -> pd.DataFrame:
    """Merge weather and air quality hourly series into one tidy DataFrame."""
    w  = weather["hourly"]
    a  = aq["hourly"]

    # Align on timestamp — both APIs should return same length, but guard mismatches
    min_len = min(len(w["time"]), len(a["time"]))

    df = pd.DataFrame({
        "timestamp":             w["time"][:min_len],
        "city":                  city,
        "latitude":              coords["latitude"],
        "longitude":             coords["longitude"],
        "temperature_2m_c":      w["temperature_2m"][:min_len],
        "relative_humidity_pct": w["relative_humidity_2m"][:min_len],
        "precipitation_mm":      w["precipitation"][:min_len],
        "wind_speed_10m_kmh":    w["wind_speed_10m"][:min_len],
        "pm2_5_ugm3":            a["pm2_5"][:min_len],
        "pm10_ugm3":             a["pm10"][:min_len],
        "carbon_monoxide_ugm3":  a["carbon_monoxide"][:min_len],
        "nitrogen_dioxide_ugm3": a["nitrogen_dioxide"][:min_len],
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_raw(city: str, data_type: str, payload: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{city}_{data_type}_historical.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"    raw JSON  -> {path.name}")


def save_csv(df: pd.DataFrame) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_DIR / "historical_data.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"    CSV       -> {path}")
    return path


# ---------------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------------

def _get_sheets_service():
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
    if not creds_path:
        raise EnvironmentError("GOOGLE_CREDENTIALS_PATH not set in .env")
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=GSHEET_SCOPES
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _ensure_sheet_tab(service, spreadsheet_id: str, tab_name: str) -> None:
    """Create the sheet tab if missing; otherwise clear its contents."""
    meta     = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    existing = {s["properties"]["title"] for s in meta["sheets"]}

    if tab_name not in existing:
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
        ).execute()
        print(f"    Created sheet tab '{tab_name}'")
    else:
        service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=f"{tab_name}!A1:Z",
        ).execute()
        print(f"    Cleared existing '{tab_name}' tab")


def push_to_sheets(df: pd.DataFrame) -> None:
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
    if not spreadsheet_id:
        raise EnvironmentError("GOOGLE_SPREADSHEET_ID not set in .env")

    service = _get_sheets_service()
    _ensure_sheet_tab(service, spreadsheet_id, SHEET_TAB)

    # Convert timestamps to string + NaN -> None (Sheets API can't handle NaN)
    df_export = df.copy()
    df_export["timestamp"] = df_export["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")
    df_export = df_export.astype(object).where(pd.notnull(df_export), None)

    values = [df_export.columns.tolist()] + df_export.values.tolist()

    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{SHEET_TAB}!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()

    print(f"    Google Sheets -> {len(df)} rows in '{SHEET_TAB}' tab")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Historical backfill: {START_DATE} -> {END_DATE}\n")

    city_dfs = []

    for city, coords in CITIES.items():
        lat, lon = coords["latitude"], coords["longitude"]
        print(f"[{city}] fetching ...")

        weather = fetch_historical_weather(lat, lon)
        aq      = fetch_historical_air_quality(lat, lon)

        save_raw(city, "weather",    weather)
        save_raw(city, "airquality", aq)

        df = build_dataframe(city, coords, weather, aq)
        city_dfs.append(df)
        print(f"    {len(df)} hourly rows  "
              f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

    combined = pd.concat(city_dfs, ignore_index=True)

    print("\n[output]")
    save_csv(combined)

    print("\n[google sheets]")
    try:
        push_to_sheets(combined)
    except EnvironmentError as e:
        print(f"    Skipped: {e}")

    print(f"\nDone. {len(combined)} total rows across {len(CITIES)} cities.")
