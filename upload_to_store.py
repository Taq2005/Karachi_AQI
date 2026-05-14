import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding for emoji output
sys.stdout.reconfigure(encoding="utf-8")

import hopsworks

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")             # always finds .env next to this script
HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
    raise ValueError(
        "Missing credentials. Make sure your .env file contains:\n"
        "  HOPSWORKS_API_KEY=your_key\n"
        "  HOPSWORKS_PROJECT=your_project_name"
    )

# ── Config ────────────────────────────────────────────────────────────────────
LAT, LON      = 24.8608, 67.0104
TIMEZONE      = "Asia/Karachi"
END_DATE      = date.today() - timedelta(days=1)        # yesterday
START_DATE    = END_DATE - timedelta(days=89)           # 90 days total
FEATURE_GROUP_NAME    = "karachi_aqi_weather"
FEATURE_GROUP_VERSION = 1

print(f"📅  Date range : {START_DATE}  →  {END_DATE}")

# ── 1. Fetch Air Quality ──────────────────────────────────────────────────────
print("📡  Fetching air quality data …")
aq_resp = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude"  : LAT,
        "longitude" : LON,
        "start_date": str(START_DATE),
        "end_date"  : str(END_DATE),
        "hourly"    : [
            "pm2_5", "pm10",
            "nitrogen_dioxide", "ozone",
            "sulphur_dioxide",  "carbon_monoxide",
            "us_aqi",
        ],
        "timezone": TIMEZONE,
    },
    timeout=30,
)
aq_resp.raise_for_status()
aq_df = pd.DataFrame(aq_resp.json()["hourly"])

# ── 2. Fetch Weather ──────────────────────────────────────────────────────────
print("📡  Fetching weather data …")
wx_resp = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude"  : LAT,
        "longitude" : LON,
        "start_date": str(START_DATE),
        "end_date"  : str(END_DATE),
        "hourly"    : [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "visibility",
            "shortwave_radiation",
        ],
        "timezone": TIMEZONE,
    },
    timeout=30,
)
wx_resp.raise_for_status()
wx_df = pd.DataFrame(wx_resp.json()["hourly"])

# ── 3. Merge ──────────────────────────────────────────────────────────────────
df = pd.merge(aq_df, wx_df, on="time", how="inner")
df["time"] = pd.to_datetime(df["time"])
df.sort_values("time", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✅  Merged shape : {df.shape}")

# ── 4. Clean ──────────────────────────────────────────────────────────────────
# forward-fill short gaps (up to 3 hours), then backward-fill edges
df.ffill(limit=3, inplace=True)
df.bfill(limit=3, inplace=True)

# fill any remaining with column median
for col in df.select_dtypes(include="number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# drop columns still >40% empty
bad = [c for c in df.columns if df[c].isnull().mean() > 0.4]
if bad:
    df.drop(columns=bad, inplace=True)
    print(f"🗑️   Dropped bad columns: {bad}")

# clip to valid physical ranges
clips = {
    "us_aqi"               : (0, 500),
    "pm2_5"                : (0, 999),
    "pm10"                 : (0, 999),
    "temperature_2m"       : (-10, 55),
    "relative_humidity_2m" : (0, 100),
    "wind_speed_10m"       : (0, 200),
}
for col, (lo, hi) in clips.items():
    if col in df.columns:
        df[col] = df[col].clip(lo, hi)

assert df.isnull().sum().sum() == 0, "❌  NaNs still present after cleaning!"
print(f"✅  Clean shape  : {df.shape}  |  nulls: {df.isnull().sum().sum()}")

# ── 5. Feature Engineering ────────────────────────────────────────────────────
df["hour"]            = df["time"].dt.hour
df["day_of_week"]     = df["time"].dt.dayofweek
df["month"]           = df["time"].dt.month
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
df["hour_sin"]        = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]        = np.cos(2 * np.pi * df["hour"] / 24)
df["aqi_lag_1h"]      = df["us_aqi"].shift(1)
df["aqi_lag_24h"]     = df["us_aqi"].shift(24)
df["aqi_roll_24h"]    = df["us_aqi"].rolling(24).mean()
df["aqi_change_rate"] = df["us_aqi"].diff()
df["pm_ratio"]        = df["pm2_5"] / (df["pm10"] + 1e-9)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✅  After features: {df.shape}")

# ── 6. Rename columns — Hopsworks requires lowercase, no spaces ───────────────
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Hopsworks needs the primary event time as a proper datetime column
df["time"] = pd.to_datetime(df["time"])

# ── 7. Connect to Hopsworks ───────────────────────────────────────────────────
print("\n🔗  Connecting to Hopsworks …")
project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=HOPSWORKS_PROJECT,
)
fs = project.get_feature_store()
print(f"✅  Connected to project: {project.name}")

# ── 8. Create or get Feature Group ───────────────────────────────────────────
fg = fs.get_or_create_feature_group(
    name              = FEATURE_GROUP_NAME,
    version           = FEATURE_GROUP_VERSION,
    primary_key       = ["time"],
    event_time        = "time",
    description       = "Hourly AQI + weather features for Karachi (Open-Meteo)",
    online_enabled    = False,         # set True if you want real-time serving
)

# ── 9. Insert data ────────────────────────────────────────────────────────────
print(f"⬆️   Uploading {len(df)} rows to feature group '{FEATURE_GROUP_NAME}' …")
fg.insert(df, write_options={"wait_for_job": True})

print("\n🎉  Done!")
print(f"   Rows uploaded : {len(df)}")
print(f"   Feature group : {FEATURE_GROUP_NAME}  (v{FEATURE_GROUP_VERSION})")
print(f"   Columns       : {list(df.columns)}")