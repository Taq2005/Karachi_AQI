"""
Upload 3 months of Karachi AQI + Weather data to MongoDB
=========================================================
pip install pymongo dnspython requests pandas numpy python-dotenv
"""

import os
import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LAT, LON   = 24.8608, 67.0104
TIMEZONE   = "Asia/Karachi"
END_DATE   = date.today() - timedelta(days=1)       # yesterday
START_DATE = END_DATE - timedelta(days=89)          # 90 days = ~3 months

print(f"📅  Date range : {START_DATE}  →  {END_DATE}")
print(f"📊  Expected rows : ~{(END_DATE - START_DATE).days * 24}")

# ── Connect to MongoDB ────────────────────────────────────────────────────────
print("\n🔗  Connecting to MongoDB …")
client     = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=10000)
client.admin.command("ping")
print("✅  Connected!")

db         = client[os.getenv("MONGO_DB", "karachi_aqi")]
collection = db[os.getenv("MONGO_COLLECTION", "hourly_features")]

# ── 1. Fetch Air Quality ──────────────────────────────────────────────────────
print("\n📡  Fetching air quality data …")
aq = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude"  : LAT,
        "longitude" : LON,
        "start_date": str(START_DATE),
        "end_date"  : str(END_DATE),
        "hourly"    : [
            "pm2_5", "pm10",
            "nitrogen_dioxide", "ozone",
            "sulphur_dioxide", "carbon_monoxide",
            "us_aqi",
        ],
        "timezone": TIMEZONE,
    },
    timeout=30,
)
aq.raise_for_status()
aq_df = pd.DataFrame(aq.json()["hourly"])
print(f"✅  Air quality rows: {len(aq_df)}")

# ── 2. Fetch Weather ──────────────────────────────────────────────────────────
print("📡  Fetching weather data …")
wx = requests.get(
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
            "shortwave_radiation",
        ],
        "timezone": TIMEZONE,
    },
    timeout=30,
)
wx.raise_for_status()
wx_df = pd.DataFrame(wx.json()["hourly"])
print(f"✅  Weather rows: {len(wx_df)}")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
df = pd.merge(aq_df, wx_df, on="time", how="inner")
df["time"] = pd.to_datetime(df["time"])
df.sort_values("time", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\n✅  Merged shape : {df.shape}")

# ── 4. Clean ──────────────────────────────────────────────────────────────────
df.ffill(limit=3, inplace=True)
df.bfill(limit=3, inplace=True)
for col in df.select_dtypes("number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

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
df.ffill(limit=3, inplace=True)
df.bfill(limit=3, inplace=True)

for col in df.select_dtypes("number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# drop columns still >40% empty
bad_cols = [c for c in df.columns if df[c].isnull().mean() > 0.4]
if bad_cols:
    df.drop(columns=bad_cols, inplace=True)
    print(f"🗑️  Dropped: {bad_cols}")

# fill anything remaining
df.fillna(0, inplace=True)

assert df.isnull().sum().sum() == 0, "❌ NaNs still present!"
print("✅  No nulls remaining")
print(f"✅  Clean shape  : {df.shape}  |  nulls: {df.isnull().sum().sum()}")

# ── 5. Feature Engineering ────────────────────────────────────────────────────
df["hour"]            = df["time"].dt.hour
df["day_of_week"]     = df["time"].dt.dayofweek
df["month"]           = df["time"].dt.month
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
df["hour_sin"]        = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]        = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]         = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]         = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["pm_ratio"]        = df["pm2_5"] / (df["pm10"] + 1e-9)
df["aqi_lag_1h"]      = df["us_aqi"].shift(1)
df["aqi_lag_24h"]     = df["us_aqi"].shift(24)
df["aqi_roll_24h"]    = df["us_aqi"].rolling(24).mean()
df["aqi_change_rate"] = df["us_aqi"].diff()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✅  After features : {df.shape}")

# ── 6. Upsert to MongoDB ──────────────────────────────────────────────────────
print(f"\n⬆️   Uploading {len(df)} rows to MongoDB …")

records = df.to_dict("records")

# convert timestamps to native python datetime for MongoDB
for r in records:
    r["time"] = pd.Timestamp(r["time"]).to_pydatetime()

ops = [
    UpdateOne(
        {"time": r["time"]},   # match on timestamp — no duplicates
        {"$set": r},
        upsert=True
    )
    for r in records
]

result     = collection.bulk_write(ops)
total_docs = collection.count_documents({})

print(f"\n🎉  Done!")
print(f"   Upserted  : {result.upserted_count} new rows")
print(f"   Modified  : {result.modified_count} existing rows")
print(f"   Total docs in MongoDB : {total_docs}")

# ── 7. Create index on time for fast queries ──────────────────────────────────
collection.create_index("time")
print("✅  Index created on 'time' field")

client.close()
