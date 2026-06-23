"""
Fetch historical data for new cities (Bandung, Medan, Makassar) and:
  1. Append to data/processed/historical_data.csv
  2. Append to Google Sheets 'historical' tab (after existing Jakarta+Surabaya data)

Usage:
    python scripts/backfill_new_cities.py
"""
import os
import sys
import requests
import pandas as pd
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Allow `from src.*` imports when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from src.ingestion.extract_api import CITIES, build_city_records  # noqa: E402

# Backfill only cities not already in historical_data.csv (Jakarta + Surabaya done)
_EXISTING = {"jakarta", "surabaya"}
NEW_CITIES = {k: v for k, v in CITIES.items() if k not in _EXISTING}

START_DATE      = "2026-01-01"
END_DATE        = date.today().isoformat()
ARCHIVE_URL     = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CSV_PATH    = Path("data/processed/historical_data.csv")
SHEET_TAB   = "historical"
GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "Asia/Jakarta",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_air_quality(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
        "timezone": "Asia/Jakarta",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()




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


def append_to_sheets(df: pd.DataFrame) -> None:
    """Append rows to existing sheet tab without clearing existing data."""
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
    if not spreadsheet_id:
        raise EnvironmentError("GOOGLE_SPREADSHEET_ID not set in .env")

    service = _get_sheets_service()

    df_export = df.copy()
    df_export["timestamp"] = df_export["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")
    df_export = df_export.astype(object).where(pd.notnull(df_export), None)

    # No header row — append after existing data
    rows = df_export.values.tolist()

    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{SHEET_TAB}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()

    print(f"  Google Sheets <- appended {len(df)} rows ke tab '{SHEET_TAB}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Backfill kota baru: {START_DATE} -> {END_DATE}\n")

    city_dfs = []

    for city, coords in NEW_CITIES.items():
        lat, lon = coords["latitude"], coords["longitude"]
        print(f"[{city}] fetching ...")
        weather  = fetch_weather(lat, lon)
        aq       = fetch_air_quality(lat, lon)
        records  = build_city_records(city, coords, weather, aq)
        df       = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        city_dfs.append(df)
        print(f"  {len(df)} rows  ({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

    df_new = pd.concat(city_dfs, ignore_index=True)

    # 1. Update CSV
    print("\n[csv]")
    df_old = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df_combined = (
        pd.concat([df_old, df_new], ignore_index=True)
        .drop_duplicates(subset=["timestamp", "city"])
        .sort_values(["city", "timestamp"])
    )
    df_combined.to_csv(CSV_PATH, index=False)
    print(f"  CSV updated: {len(df_combined)} total rows")
    print(df_combined.groupby("city").size().to_string())

    # 2. Append to Google Sheets
    print("\n[google sheets]")
    try:
        append_to_sheets(df_new)
    except EnvironmentError as e:
        print(f"  Skipped: {e}")

    print(f"\nDone. {len(df_new)} rows baru untuk {list(NEW_CITIES.keys())}.")
