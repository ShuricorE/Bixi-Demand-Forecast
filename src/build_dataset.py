"""
build_dataset.py: turns millions of individual trips into a modelling table
with one row per station per hour.

1. Count departures and arrivals per station per hour (read in chunks to save memory)
2. Keep the 100 busiest stations
3. Fill in every station x hour combination, with 0 where nobody rode (important!)
4. Add calendar, weather, and lag features

Run:  python -m src.build_dataset
"""
import numpy as np
import pandas as pd

from src.config import (
    DATASET, HOLIDAYS, N_STATIONS, RAW_TRIPS_DIR, RAW_WEATHER, SEASON_END,
    SEASON_START, STATIONS_FILE, TIMEZONE,
)

TRIP_COLS = [
    "STARTSTATIONNAME", "STARTSTATIONARRONDISSEMENT",
    "STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE",
    "ENDSTATIONNAME", "STARTTIMEMS", "ENDTIMEMS",
]


def to_local_hour(timestamps):
    """Unix timestamps (UTC) -> Montreal local time, rounded down to the hour.
    Above 1e11 the numbers are milliseconds, otherwise seconds."""
    unit = "ms" if timestamps.max() > 1e11 else "s"
    t = pd.to_datetime(timestamps, unit=unit, utc=True)
    return t.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.floor("h")


def count_trips():
    files = sorted(RAW_TRIPS_DIR.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {RAW_TRIPS_DIR}. Run python -m src.download first.")
    dep_parts, arr_parts, station_parts = [], [], []
    for f in files:
        print(f"Reading {f.name} ...")
        for i, chunk in enumerate(pd.read_csv(f, usecols=TRIP_COLS, chunksize=1_000_000)):
            print(f"  chunk {i + 1}")
            chunk = chunk.dropna(subset=["STARTSTATIONNAME", "ENDSTATIONNAME", "STARTTIMEMS", "ENDTIMEMS"])
            dep = pd.DataFrame({"station": chunk["STARTSTATIONNAME"],
                                "hour": to_local_hour(chunk["STARTTIMEMS"])})
            arr = pd.DataFrame({"station": chunk["ENDSTATIONNAME"],
                                "hour": to_local_hour(chunk["ENDTIMEMS"])})
            dep_parts.append(dep.value_counts().rename("departures"))
            arr_parts.append(arr.value_counts().rename("arrivals"))
            station_parts.append(
                chunk[["STARTSTATIONNAME", "STARTSTATIONARRONDISSEMENT",
                       "STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE"]]
                .drop_duplicates("STARTSTATIONNAME"))
    # Each chunk gave partial counts; add them up.
    departures = pd.concat(dep_parts).groupby(level=[0, 1]).sum()
    arrivals = pd.concat(arr_parts).groupby(level=[0, 1]).sum()
    stations = (pd.concat(station_parts).drop_duplicates("STARTSTATIONNAME")
                .rename(columns={"STARTSTATIONNAME": "station",
                                 "STARTSTATIONARRONDISSEMENT": "borough",
                                 "STARTSTATIONLATITUDE": "lat",
                                 "STARTSTATIONLONGITUDE": "lon"}))
    return departures, arrivals, stations


def load_weather():
    w = pd.read_csv(RAW_WEATHER)
    w["hour"] = (pd.to_datetime(w.pop("time_utc"), utc=True)
                 .dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.floor("h"))
    # When clocks fall back in November, two UTC hours land on the same local hour: average them.
    return w.groupby("hour", as_index=False).mean()


def build():
    departures, arrivals, stations = count_trips()

    # 1) Keep the busiest stations within the season.
    start = pd.Timestamp(SEASON_START)
    end = pd.Timestamp(SEASON_END) + pd.Timedelta(hours=23)
    h = departures.index.get_level_values("hour")
    dep_season = departures[(h >= start) & (h <= end)]
    top = dep_season.groupby(level="station").sum().nlargest(N_STATIONS).index

    stations = stations[stations["station"].isin(top)].sort_values("station").reset_index(drop=True)
    stations["station_id"] = np.arange(len(stations))   # 0..99, used by the model
    STATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(STATIONS_FILE, index=False)

    # 2) Full grid: every station x every hour. Hours with no trips become 0.
    hours = pd.date_range(start, end, freq="h")
    grid = pd.MultiIndex.from_product([stations["station"], hours], names=["station", "hour"])
    df = pd.DataFrame(index=grid)
    df["departures"] = departures.reindex(grid, fill_value=0)
    df["arrivals"] = arrivals.reindex(grid, fill_value=0)
    df = df.reset_index().merge(stations[["station", "station_id", "lat", "lon"]], on="station")

    # 3) Calendar features
    df["hour_of_day"] = df["hour"].dt.hour
    df["dow"] = df["hour"].dt.dayofweek          # 0 = Monday
    df["month"] = df["hour"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["hour"].dt.normalize().isin(pd.to_datetime(HOLIDAYS)).astype(int)

    # 4) Weather. Missing hours are filled from their neighbours so no gaps reach the model.
    weather = (load_weather().set_index("hour").reindex(hours)
               .interpolate().ffill().bfill().rename_axis("hour").reset_index())
    df = df.merge(weather, on="hour", how="left")

    # 5) Lag features: this station 1 day and 1 week earlier, same hour. Past only.
    df = df.sort_values(["station_id", "hour"]).reset_index(drop=True)
    for target in ["departures", "arrivals"]:
        g = df.groupby("station_id")[target]
        df[f"{target}_lag_24"] = g.shift(24)
        df[f"{target}_lag_168"] = g.shift(168)
    df = df.dropna(subset=["departures_lag_168"])  # the first week has no lag yet

    # "hour" is the timestamp; the model uses "hour_of_day". Rename for clarity.
    df = df.rename(columns={"hour": "timestamp", "hour_of_day": "hour"})
    df.to_csv(DATASET, index=False)
    print(f"Saved {len(df):,} rows ({df['station_id'].nunique()} stations) to {DATASET}")
    return df


if __name__ == "__main__":
    build()
    