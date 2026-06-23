"""One-time script: append new cities' recent data to historical CSV."""
import pandas as pd
from pathlib import Path
from src.ingestion.extract_api import fetch_all_cities

CSV_PATH = Path("data/processed/historical_data.csv")

print("Fetching 7 days dari open-meteo (semua 5 kota)...")
records = fetch_all_cities(past_days=7)
df_new = pd.DataFrame(records)
df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

df_old = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
print(f"CSV lama: {len(df_old)} rows, kota: {df_old['city'].unique().tolist()}")

df = pd.concat([df_old, df_new], ignore_index=True)
df = df.drop_duplicates(subset=["timestamp", "city"]).sort_values(["city", "timestamp"])
df.to_csv(CSV_PATH, index=False)

print(f"\nCSV updated: {len(df)} total rows")
print(df.groupby("city").size().to_string())
