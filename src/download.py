"""
download.py: downloads the two public datasets.
1. BIXI 2024 trip history (one row per trip, millions of trips)
2. Hourly Montreal weather for the 2024 season, from Open-Meteo

Run:  python -m src.download
"""
import zipfile

import pandas as pd
import requests

from src.config import (
    BIXI_URL, MONTREAL_LAT, MONTREAL_LON, RAW_TRIPS_DIR, RAW_WEATHER,
    SEASON_END, SEASON_START, WEATHER_URL,
)


def download_trips():
    """Download the BIXI zip (a few hundred MB) and unzip the CSVs."""
    RAW_TRIPS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_TRIPS_DIR.parent / "bixi_2024.zip"
    if not zip_path.exists():
        print("Downloading BIXI trips (this can take a few minutes)...")
        with requests.get(BIXI_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB at a time
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        for name in csvs:
            z.extract(name, RAW_TRIPS_DIR)
    print(f"Unzipped {len(csvs)} CSV file(s) into {RAW_TRIPS_DIR}")


def download_weather():
    """Hourly weather in UTC. We convert to Montreal time later, the same way as the trips."""
    params = {
        "latitude": MONTREAL_LAT,
        "longitude": MONTREAL_LON,
        "start_date": SEASON_START,
        "end_date": SEASON_END,
        "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
        "timezone": "GMT",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame({
        "time_utc": hourly["time"],
        "temperature": hourly["temperature_2m"],          # °C
        "precipitation": hourly["precipitation"],         # mm in that hour
        "humidity": hourly["relative_humidity_2m"],       # %
        "wind_speed": hourly["wind_speed_10m"],           # km/h
    })
    RAW_WEATHER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_WEATHER, index=False)
    print(f"Saved {len(df)} hours of weather to {RAW_WEATHER}")


if __name__ == "__main__":
    download_weather()
    download_trips()