"""
config.py: every setting in one place. Every other file imports from here.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
RAW_TRIPS_DIR = ROOT / "data" / "raw" / "bixi"          # BIXI trip CSVs go here
RAW_WEATHER = ROOT / "data" / "raw" / "weather.csv"
STATIONS_FILE = ROOT / "data" / "processed" / "stations.csv"
DATASET = ROOT / "data" / "processed" / "station_hourly.csv"
TEST_PREDICTIONS = ROOT / "data" / "processed" / "test_predictions.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
METRICS_FILE = MODELS_DIR / "metrics.json"

# ---------- Data sources (public) ----------
BIXI_URL = "https://cdn.bixi.com/wp-content/uploads/2025/01/DonneesOuvertes2024_010203040506070809101112.zip"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
MONTREAL_LAT, MONTREAL_LON = 45.5017, -73.5673

# ---------- Time ----------
TIMEZONE = "America/Montreal"
SEASON_START = "2024-04-15"   # BIXI runs roughly mid April to mid November
SEASON_END = "2024-11-15"
VALID_START = "2024-09-10"    # last 3 weeks of training data, used only for tuning
TEST_START = "2024-10-01"     # everything from here on is the locked test set

# Quebec statutory holidays inside the 2024 season
HOLIDAYS = ["2024-05-20", "2024-06-24", "2024-07-01", "2024-09-02", "2024-10-14"]

# ---------- Modelling ----------
N_STATIONS = 100
TARGETS = ["departures", "arrivals"]
WEATHER_COLS = ["temperature", "precipitation", "humidity", "wind_speed"]
LAG_COLS = ["departures_lag_24", "departures_lag_168", "arrivals_lag_24", "arrivals_lag_168"]
FEATURES = (
    ["station_id", "lat", "lon", "hour", "dow", "month", "is_weekend", "is_holiday"]
    + WEATHER_COLS
    + LAG_COLS
)
RANDOM_STATE = 42