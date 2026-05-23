import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import requests

load_dotenv()

LAT, LON = 24.8608, 67.0104
TIMEZONE = "Asia/Karachi"

client     = MongoClient(os.getenv("MONGO_URI"))
db         = client[os.getenv("MONGO_DB", "karachi_aqi")]
collection = db[os.getenv("MONGO_COLLECTION", "hourly_features")]

try:
    # ── Smart date range ──────────────────────────────────────────────────────
    END_DATE = date.today() - timedelta(days=5)   # archive API has 2-5 day lag

    last_doc = collection.find_one(sort=[("time", -1)])
    if last_doc:
        last_date  = pd.Timestamp(last_doc["time"]).date()
        START_DATE = last_date - timedelta(days=30)
        print(f"📅  Last record : {last_date}")
    else:
        START_DATE = END_DATE - timedelta(days=90)
        print("📅  First run — fetching 90 days")

    if START_DATE >= END_DATE:
        print("✅  Already up to date.")
        exit(0)

    print(f"📅  Fetching  {START_DATE}  →  {END_DATE}")

    # ── Fetch Air Quality ─────────────────────────────────────────────────────
    print("📡  Fetching air quality data …")
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude"  : LAT, "longitude": LON,
            "start_date": str(START_DATE), "end_date": str(END_DATE),
            "hourly"    : ["pm2_5","pm10","nitrogen_dioxide","ozone",
                           "sulphur_dioxide","carbon_monoxide","us_aqi"],
            "timezone"  : TIMEZONE,
        }, timeout=30
    )
    if aq_resp.status_code != 200 or not aq_resp.text.strip():
        print(f"⚠️  Air quality API error {aq_resp.status_code}: {aq_resp.text[:200]}")
        exit(0)
    aq = aq_resp.json()["hourly"]
    print(f"✅  Air quality rows: {len(aq['time'])}")

    # ── Fetch Weather ─────────────────────────────────────────────────────────
    print("📡  Fetching weather data …")
    wx_resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude"  : LAT, "longitude": LON,
            "start_date": str(START_DATE), "end_date": str(END_DATE),
            "hourly"    : ["temperature_2m","relative_humidity_2m","dew_point_2m",
                           "apparent_temperature","wind_speed_10m","wind_direction_10m",
                           "wind_gusts_10m","precipitation","surface_pressure",
                           "cloud_cover","visibility","shortwave_radiation"],
            "timezone"  : TIMEZONE,
        }, timeout=30
    )
    if wx_resp.status_code != 200 or not wx_resp.text.strip():
        print(f"⚠️  Weather API error {wx_resp.status_code}: {wx_resp.text[:200]}")
        exit(0)
    wx = wx_resp.json()["hourly"]
    print(f"✅  Weather rows: {len(wx['time'])}")

    # ── Merge & clean ─────────────────────────────────────────────────────────
    df = pd.merge(pd.DataFrame(aq), pd.DataFrame(wx), on="time")
    df["time"] = pd.to_datetime(df["time"])
    df.ffill(limit=3, inplace=True)
    df.bfill(limit=3, inplace=True)
    for col in df.select_dtypes("number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    df.fillna(0, inplace=True)
    print(f"✅  Merged shape: {df.shape}")

    # ── Feature Engineering ───────────────────────────────────────────────────
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
    df.dropna(subset=["aqi_lag_24h", "aqi_roll_24h"], inplace=True)
    df.fillna(0, inplace=True)

    # ── Filter only new rows ──────────────────────────────────────────────────
    if last_doc:
        last_uploaded = pd.Timestamp(last_doc["time"]).tz_localize(None)
        df = df[df["time"] > last_uploaded]

    print(f"✅  New rows to upload: {len(df)}")

    if df.empty:
        print("✅  Already up to date — nothing to upload.")
        exit(0)

    # ── Upload ────────────────────────────────────────────────────────────────
    records = df.to_dict("records")
    for r in records:
        r["time"] = pd.Timestamp(r["time"]).to_pydatetime()

    ops    = [UpdateOne({"time": r["time"]}, {"$set": r}, upsert=True) for r in records]
    result = collection.bulk_write(ops)
    print(f"🎉  Upserted {result.upserted_count}  |  Modified {result.modified_count}"
          f"  |  Total: {collection.count_documents({})}")

except Exception as e:
    print(f"❌  Error: {e}")
    raise

finally:
    client.close()
    print("🔒  Connection closed.")
