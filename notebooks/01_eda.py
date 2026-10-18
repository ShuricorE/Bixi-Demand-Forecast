# %% [markdown]
# # 01: Exploratory data analysis
# Under each chart, write one sentence about what you see.

# %%
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATASET, STATIONS_FILE

df = pd.read_csv(DATASET, parse_dates=["timestamp"])
stations = pd.read_csv(STATIONS_FILE)
print(df.shape)
df.head()

# %% 1. Daily rides over the season
daily = df.set_index("timestamp")["departures"].resample("D").sum()
daily.plot(figsize=(12, 4), title="Departures per day, top 100 stations")

# %% 2. The commute shape: weekday vs weekend, by hour
by_hour = df.groupby(["is_weekend", "hour"])["departures"].mean().unstack(0)
by_hour.columns = ["weekday", "weekend"]
by_hour.plot(figsize=(10, 4), title="Average departures per station by hour")

# %% 3. Temperature and rain
hourly_total = df.groupby("timestamp").agg(
    departures=("departures", "sum"), temperature=("temperature", "first"),
    precipitation=("precipitation", "first"), hour=("hour", "first"))
daytime = hourly_total[hourly_total["hour"].between(8, 20)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(daytime["temperature"], daytime["departures"], s=4, alpha=0.4)
axes[0].set(xlabel="Temperature (°C)", ylabel="Departures per hour")
daytime.assign(raining=daytime["precipitation"] > 0.5).boxplot(
    column="departures", by="raining", ax=axes[1])
plt.suptitle("")
plt.tight_layout()

# %% 4. Busiest stations
top = df.groupby("station_id")["departures"].sum().nlargest(10)
top.index = stations.set_index("station_id").loc[top.index, "station"]
top.sort_values().plot.barh(figsize=(8, 5), title="Top 10 stations by departures")

# %% 5. Weekday mornings: which stations empty out, which fill up?
morning = df[df["hour"].between(7, 9) & (df["is_weekend"] == 0)]
net = (morning.groupby("station_id")[["arrivals", "departures"]].mean()
       .assign(net=lambda x: x["arrivals"] - x["departures"]))
net = net.join(stations.set_index("station_id")[["station", "lat", "lon"]])
plt.figure(figsize=(7, 7))
from matplotlib.colors import TwoSlopeNorm
plt.scatter(net["lon"], net["lat"], c=net["net"], cmap="coolwarm_r", s=60,
            norm=TwoSlopeNorm(vcenter=0))
plt.colorbar(label="arrivals minus departures per hour (blue = fills up)")
plt.title("Weekday morning: red stations empty out, blue ones fill up")
# %%
