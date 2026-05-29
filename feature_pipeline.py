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
db         = client[os.getenv("MONGO_DB", "karachi_aqi_weather")]
collection = db[os.getenv("MONGO_COLLECTION", "hourly_features")]

def get_latest_available_date():
    for days_back in range(1, 10):
        probe_date = date.today() - timedelta(days=days_back)
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": LAT, "longitude": LON,
                "start_date": str(probe_date), "end_date": str(probe_date),
                "hourly": ["temperature_2m"], "timezone": TIMEZONE,
            }, timeout=15
        )
        if resp.status_code == 200 and resp.text.strip():
            try:
                temps = resp.json()["hourly"].get("temperature_2m", [])
                if temps and any(v is not None for v in temps):
                    print(f"✅  Latest archive date: {probe_date}")
                    return probe_date
            except Exception:
                continue
    return date.today() - timedelta(days=7)

def apply_features(df):
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
    df.fillna(0, inplace=True)
    return df

try:
    END_DATE = get_latest_available_date()
    today    = date.today()

    last_doc = collection.find_one(sort=[("time", -1)])
    if last_doc:
        last_date  = pd.Timestamp(last_doc["time"]).date()
        START_DATE = last_date - timedelta(days=2)
        print(f"📅  Last record : {last_date}")
    else:
        last_date  = date.min
        START_DATE = END_DATE - timedelta(days=90)
        print("📅  First run — fetching 90 days")

    frames = []

    # ── 1. Archive fetch — only if behind ─────────────────────────────────────
    if last_date < END_DATE:
        print(f"📅  Fetching archive  {START_DATE}  →  {END_DATE}")
        aq_resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": LAT, "longitude": LON,
                "start_date": str(START_DATE), "end_date": str(END_DATE),
                "hourly": ["pm2_5","pm10","nitrogen_dioxide","ozone",
                           "sulphur_dioxide","carbon_monoxide","us_aqi"],
                "timezone": TIMEZONE,
            }, timeout=30
        )
        wx_resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": LAT, "longitude": LON,
                "start_date": str(START_DATE), "end_date": str(END_DATE),
                "hourly": ["temperature_2m","relative_humidity_2m","dew_point_2m",
                           "apparent_temperature","wind_speed_10m","wind_direction_10m",
                           "wind_gusts_10m","precipitation","surface_pressure",
                           "cloud_cover","shortwave_radiation"],
                "timezone": TIMEZONE,
            }, timeout=30
        )
        if (aq_resp.status_code == 200 and aq_resp.text.strip() and
            wx_resp.status_code == 200 and wx_resp.text.strip()):
            df_arch = pd.merge(
                pd.DataFrame(aq_resp.json()["hourly"]),
                pd.DataFrame(wx_resp.json()["hourly"]),
                on="time"
            )
            df_arch["time"] = pd.to_datetime(df_arch["time"])
            df_arch.ffill(limit=3, inplace=True)
            df_arch.bfill(limit=3, inplace=True)
            for c in df_arch.select_dtypes("number").columns:
                df_arch[c].fillna(df_arch[c].median(), inplace=True)
            df_arch = apply_features(df_arch)
            frames.append(df_arch)
            print(f"✅  Archive rows: {len(df_arch)}")
        else:
            print("⚠️  Archive API error — skipping")
    else:
        print(f"✅  Archive already up to date (last={last_date})")

    # ── 2. Today from forecast API — always fetch ─────────────────────────────
    print("📡  Fetching today's data from forecast API …")
    aq_today = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": str(today), "end_date": str(today),
            "hourly": ["pm2_5","pm10","nitrogen_dioxide","ozone",
                       "sulphur_dioxide","carbon_monoxide","us_aqi"],
            "timezone": TIMEZONE,
        }, timeout=30
    )
    wx_today = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": str(today), "end_date": str(today),
            "hourly": ["temperature_2m","relative_humidity_2m","dew_point_2m",
                       "apparent_temperature","wind_speed_10m","wind_direction_10m",
                       "wind_gusts_10m","precipitation","surface_pressure",
                       "cloud_cover","shortwave_radiation"],
            "timezone": TIMEZONE,
        }, timeout=30
    )
    if (aq_today.status_code == 200 and aq_today.text.strip() and
        wx_today.status_code == 200 and wx_today.text.strip()):
        df_today = pd.merge(
            pd.DataFrame(aq_today.json()["hourly"]),
            pd.DataFrame(wx_today.json()["hourly"]),
            on="time"
        )
        df_today["time"] = pd.to_datetime(df_today["time"])

        # only keep hours that have already passed
        now_hour = pd.Timestamp.now().floor("h")
        df_today = df_today[df_today["time"] <= now_hour]

        df_today.ffill(limit=3, inplace=True)
        df_today.bfill(limit=3, inplace=True)
        for c in df_today.select_dtypes("number").columns:
            df_today[c].fillna(df_today[c].median(), inplace=True)
        df_today = apply_features(df_today)
        frames.append(df_today)
        print(f"✅  Today's rows: {len(df_today)}")
    else:
        print("⚠️  Forecast API error — skipping today's fetch")

    # ── 3. Combine ────────────────────────────────────────────────────────────
    if not frames:
        print("✅  Nothing to upload.")
        exit(0)

    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(subset=["time"], keep="last", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 4. Filter only new rows ───────────────────────────────────────────────
    if last_doc:
        last_uploaded = pd.Timestamp(last_doc["time"]).tz_localize(None)
        df = df[df["time"] > last_uploaded]

    print(f"✅  New rows to upload: {len(df)}")

    if df.empty:
        print("✅  Already up to date — nothing to upload.")
        exit(0)

    # ── 5. Upload ─────────────────────────────────────────────────────────────
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
