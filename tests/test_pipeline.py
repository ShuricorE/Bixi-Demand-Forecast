import pandas as pd

from src.build_dataset import to_local_hour
from src.recommend import haversine_km, plan_moves


def test_local_hour_handles_milliseconds_and_seconds():
    # 2024-07-01 12:30 UTC is 08:30 in Montreal (summer time, UTC-4)
    ms = pd.Series([1719837000000])
    s = pd.Series([1719837000])
    assert to_local_hour(ms).iloc[0] == pd.Timestamp("2024-07-01 08:00")
    assert to_local_hour(s).iloc[0] == pd.Timestamp("2024-07-01 08:00")


def test_haversine_known_distance():
    # Montreal to Quebec City is about 233 km
    d = haversine_km(45.5017, -73.5673, 46.8139, -71.2080)
    assert 225 < d < 240


def test_plan_moves_fills_shortage_from_nearest():
    window = pd.DataFrame({
        "station": ["short", "near", "far"],
        "lat": [45.50, 45.501, 45.60],
        "lon": [-73.57, -73.57, -73.57],
        "pred_net": [-10.0, 6.0, 20.0],
    })
    moves = plan_moves(window, truck_capacity=20)
    assert moves.iloc[0]["from_station"] == "near"     # nearest donor first
    assert moves["bikes"].sum() == 10                  # shortage fully covered
