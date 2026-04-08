import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== USER INPUT ==================
FILE_PATH = "ITO_CPCB.csv"
STATION_NAME = "ITO"

# ================== LOAD DATA ==================
df = pd.read_csv(FILE_PATH, parse_dates=["datetimeLocal"])

df = df.rename(columns={
    "datetimeLocal": "datetime",
    "PM2.5": "pm25",
    "T2M": "temperature",
    "RH2M": "humidity",
    "WS2M": "wind_speed",
    "PRECTOTCORR": "precipitation"
})

df = df.sort_values("datetime")

# Filter range
df = df[(df["datetime"] >= "2022-01-01") & (df["datetime"] <= "2025-12-31")]

plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["font.size"] = 11

# ================== 1) TIME SERIES ==================

def plot_time(df, title):
    plt.figure()
    plt.plot(df["datetime"], df["pm25"], color="blue", label="PM2.5")

    plt.xlabel("Time")
    plt.ylabel("PM2.5 Concentration (µg/m³)")
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# (a) Full
plot_time(df, f"PM2.5 vs Time ({STATION_NAME}, 2022–2025)")

# (b–e) Year-wise
for year in [2022, 2023, 2024, 2025]:
    temp = df[df["datetime"].dt.year == year]
    plot_time(temp, f"PM2.5 vs Time ({year}, {STATION_NAME})")

# (f) Daily Avg
daily = df.resample("D", on="datetime").mean(numeric_only=True)

plt.figure()
plt.plot(daily.index, daily["pm25"], color="green", label="Daily Avg PM2.5")

plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration (µg/m³)")
plt.title(f"Daily Average PM2.5 ({STATION_NAME}, 2022–2025)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# (g) Monthly Avg
monthly = df.resample("M", on="datetime").mean(numeric_only=True)

plt.figure()
plt.plot(monthly.index, monthly["pm25"], color="purple", label="Monthly Avg PM2.5")

plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration (µg/m³)")
plt.title(f"Monthly Average PM2.5 ({STATION_NAME}, 2022–2025)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== 2) METEOROLOGICAL FEATURES ==================

features = {
    "temperature": "Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "wind_speed": "Wind Speed (m/s)",
    "precipitation": "Precipitation (mm)"
}

for col, label in features.items():
    plt.figure()

    x = df[col]
    y = df["pm25"]

    corr = np.corrcoef(x, y)[0,1]

    plt.scatter(x, y, alpha=0.3, label="Data Points")

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), color="red", linewidth=2, label="Trend Line")

    plt.xlabel(label)
    plt.ylabel("PM2.5 Concentration (µg/m³)")
    plt.title(f"PM2.5 vs {label} ({STATION_NAME})\nCorrelation: {corr:.3f}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================== 3) EVENT FEATURES ==================

def mark_events(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    diwali_dates = ["2022-10-24","2023-11-12","2024-11-01","2025-10-20"]
    df["diwali"] = df["datetime"].dt.date.astype(str).isin(diwali_dates)

    df["crop_burning"] = df["month"].isin([10,11])
    df["winter"] = df["month"].isin([12,1,2])

    return df

df = mark_events(df)

def event_bar(event_col, title):
    yearly = []

    for year in [2022, 2023, 2024, 2025]:
        temp = df[df["year"] == year]

        event_avg = temp[temp[event_col]]["pm25"].mean()
        non_event_avg = temp[~temp[event_col]]["pm25"].mean()

        yearly.append((event_avg, non_event_avg))

    x = np.arange(4)
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, [y[0] for y in yearly], width, label="Event")
    plt.bar(x + width/2, [y[1] for y in yearly], width, label="Non-Event")

    plt.xticks(x, ["2022","2023","2024","2025"])
    plt.xlabel("Year")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.title(f"{title} ({STATION_NAME})")

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

event_bar("diwali", "Diwali vs Rest PM2.5")
event_bar("crop_burning", "Crop Burning vs Rest PM2.5")
event_bar("winter", "Winter Inversion vs Rest PM2.5")

# ================== 5) CORRELATION HEATMAP ==================

plt.figure(figsize=(16,12))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 7},
    linewidths=0.5,
    square=True
)

plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.title(f"Correlation Heatmap ({STATION_NAME})", fontsize=16, pad=20)

plt.subplots_adjust(top=0.92, bottom=0.15, left=0.15, right=0.95)

plt.show()
plt.close()
