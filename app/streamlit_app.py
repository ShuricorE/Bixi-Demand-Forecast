"""
streamlit_app.py: interactive demo.   Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # so we can import src/

import altair as alt
import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, METRICS_FILE, STATIONS_FILE, TEST_PREDICTIONS
from src.recommend import forecast_window, plan_moves, summarize

st.set_page_config(page_title="BIXI Demand Forecast", layout="wide")
st.title("BIXI Demand Forecast and Rebalancing")
st.caption("Built on public BIXI Montréal 2024 trip data and Open-Meteo weather.")

if not TEST_PREDICTIONS.exists():
    st.error("No predictions found. Run `python -m src.train` first.")
    st.stop()


@st.cache_data
def load():
    preds = pd.read_csv(TEST_PREDICTIONS, parse_dates=["timestamp"])
    stations = pd.read_csv(STATIONS_FILE)
    return preds.merge(stations[["station_id", "station"]], on="station_id"), stations


@st.cache_data
def cached_window(date, start_hour, end_hour):
    return forecast_window(date, start_hour, end_hour)


def hourly_chart(day, actual_col, pred_col, title):
    """Actual (grey, solid) vs forecast (red, dashed), x axis fixed to hours 0 to 23."""
    data = day[["hour", actual_col, pred_col]].rename(
        columns={actual_col: "actual", pred_col: "forecast"})
    long = data.melt("hour", var_name="series", value_name="rides")
    series = ["actual", "forecast"]
    return (
        alt.Chart(long, title=title)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y("rides:Q", title="Rides", scale=alt.Scale(domainMin=0)),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=series, range=["#9e9e9e", "#ff4b4b"])),
            strokeDash=alt.StrokeDash("series:N", legend=None,
                                      scale=alt.Scale(domain=series, range=[[1, 0], [6, 3]])),
        )
        .properties(height=300)
    )


preds, stations = load()
dates = sorted(preds["timestamp"].dt.date.unique())
forecast_tab, rebalance_tab, model_tab = st.tabs(["Forecast", "Rebalance", "Model"])

# ---------------- Forecast ----------------
with forecast_tab:
    c1, c2 = st.columns(2)
    station = c1.selectbox("Station", sorted(preds["station"].unique()))
    date = c2.selectbox("Day (test period)", dates, index=min(7, len(dates) - 1))
    day = preds[(preds["station"] == station) & (preds["timestamp"].dt.date == date)]
    day = day.assign(hour=day["timestamp"].dt.hour)
    st.altair_chart(hourly_chart(day, "departures", "pred_departures", "Departures per hour"),
                    use_container_width=True)
    st.altair_chart(hourly_chart(day, "arrivals", "pred_arrivals", "Arrivals per hour"),
                    use_container_width=True)

# ---------------- Rebalance ----------------
with rebalance_tab:
    c1, c2, c3, c4 = st.columns(4)
    r_date = c1.selectbox("Day", dates, index=min(7, len(dates) - 1), key="r_date")
    start_hour, end_hour = c2.slider("Window (hours)", 0, 24, (7, 10))
    capacity = c3.number_input("Truck capacity (bikes)", 5, 60, 20)
    n_worst = c4.number_input("Stations to fix", 1, 30, 10)

    window = cached_window(r_date, start_hour, end_hour)
    worst = window.nsmallest(n_worst, "pred_net")
    donors = window[window["pred_net"] > 0]
    moves = plan_moves(pd.concat([worst, donors]), truck_capacity=capacity)

    st.metric("Plan", summarize(moves, capacity))
    left, right = st.columns(2)
    with left:
        st.subheader("Stations losing the most bikes")
        cols = ["station", "pred_departures", "pred_arrivals", "pred_net", "actual_net"]
        st.dataframe(worst[cols].round(1), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Suggested truck moves")
        if moves.empty:
            st.info("No station is short enough to need a truck in this window.")
        else:
            st.dataframe(moves, use_container_width=True, hide_index=True)

    map_df = window.copy()
    map_df["color"] = map_df["pred_net"].apply(lambda v: "#d62728" if v < 0 else "#1f77b4")
    map_df["size"] = map_df["pred_net"].abs() * 15 + 30
    st.subheader("Map: red stations empty out, blue stations fill up")
    st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")

# ---------------- Model ----------------
with model_tab:
    if METRICS_FILE.exists():
        metrics = json.loads(METRICS_FILE.read_text())
        for target, results in metrics.items():
            st.write(f"**{target}** (test set, October to November)")
            st.dataframe(pd.DataFrame({k: {m: v[m] for m in ("mae", "rmse", "r2")}
                                       for k, v in results.items()}).T)
    for name in ["shap_summary.png", "shap_waterfall_example.png"]:
        if (FIGURES_DIR / name).exists():
            st.image(str(FIGURES_DIR / name))
