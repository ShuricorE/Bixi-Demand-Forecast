"""
recommend.py: a bike rebalancing plan for a morning rush.
1. Predict net flow = arrivals - departures per station over the window
   (negative = the station loses bikes)
2. Fill each shortage from the NEAREST station with extra bikes, in truck loads

Run:  python -m src.recommend
"""
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, STATIONS_FILE
from src.train import load_dataset


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight line distance between two points on Earth, in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def forecast_window(date, start_hour=7, end_hour=10):
    """Predicted departures and arrivals per station between start_hour and end_hour."""
    df = load_dataset()
    day = df[(df["timestamp"].dt.date == pd.Timestamp(date).date())
             & df["hour"].between(start_hour, end_hour - 1)]
    if day.empty:
        raise ValueError(f"No data for {date}. Pick a date inside the season.")
    out = day[["station_id", "departures", "arrivals"]].copy()
    for target in ["departures", "arrivals"]:
        bundle = joblib.load(MODELS_DIR / f"{target}_model.joblib")
        out[f"pred_{target}"] = bundle["model"].predict(day[bundle["features"]])
    totals = out.groupby("station_id").sum()
    totals["pred_net"] = totals["pred_arrivals"] - totals["pred_departures"]
    totals["actual_net"] = totals["arrivals"] - totals["departures"]
    stations = pd.read_csv(STATIONS_FILE).set_index("station_id")
    return stations.join(totals).reset_index()


def plan_moves(window, truck_capacity=20, min_gap=3):
    """Greedy plan: biggest shortage first, filled from the nearest surplus station."""
    need = window[window["pred_net"] <= -min_gap].sort_values("pred_net")
    extra = window[window["pred_net"] >= min_gap].copy()
    extra["available"] = extra["pred_net"].round()

    moves = []
    for _, s in need.iterrows():
        shortage = round(-s["pred_net"])
        while shortage > 0 and (extra["available"] > 0).any():
            donors = extra[extra["available"] > 0]
            dist = haversine_km(s["lat"], s["lon"], donors["lat"], donors["lon"])
            d_idx = dist.idxmin()
            bikes = int(min(shortage, extra.at[d_idx, "available"], truck_capacity))
            moves.append({
                "from_station": extra.at[d_idx, "station"],
                "to_station": s["station"],
                "bikes": bikes,
                "distance_km": round(float(dist[d_idx]), 2),
            })
            extra.at[d_idx, "available"] -= bikes
            shortage -= bikes
    return pd.DataFrame(moves)


def summarize(moves, truck_capacity=20):
    """One line overview of a plan."""
    if moves.empty:
        return "No moves needed."
    trips = int(np.ceil(moves["bikes"] / truck_capacity).sum())
    return (f"{int(moves['bikes'].sum())} bikes moved in {trips} truck trips, "
            f"{moves['distance_km'].sum():.1f} km in total")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    date = "2024-10-08"   # a Tuesday in the test period
    window = forecast_window(date)
    cols = ["station", "pred_departures", "pred_arrivals", "pred_net", "actual_net"]
    print(f"Stations losing the most bikes, {date} 7 to 10 am:")
    print(window.sort_values("pred_net")[cols].head(10).round(1).to_string(index=False))

    # Focus the plan on the 10 worst shortages: that is what a crew can do in one morning.
    worst = window.nsmallest(10, "pred_net")
    donors = window[window["pred_net"] > 0]
    moves = plan_moves(pd.concat([worst, donors]))
    print("\nSuggested truck moves (10 worst stations):")
    print(moves.to_string(index=False) if not moves.empty else "No moves needed.")
    print("\n" + summarize(moves))