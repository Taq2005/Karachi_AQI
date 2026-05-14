"""
Fetch 90 days of Karachi weather + air quality data from Open-Meteo.
- Date range  : (today - 90 days)  →  yesterday
- No API key  : Open-Meteo is free and open
- Output      : karachi_weather_90d.csv  (clean, no empty columns)
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
LAT, LON   = 24.8608, 67.0104
TIMEZONE   = "Asia/Karachi"
END_DATE   = date.today() - timedelta(days=1)          # yesterday
START_DATE = END_DATE - timedelta(days=89)             # 90 days total
OUT_FILE   = "karachi_weather_90d.csv"

print(f"📅  Fetching  {START_DATE}  →  {END_DATE}  ({(END_DATE - START_DATE).days + 1} days)")

# ── 1. Air Quality ────────────────────────────────────────────────────────────
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

# ── 2. Weather ────────────────────────────────────────────────────────────────
wx_resp = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",   # historical endpoint
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
            "shortwave_radiation",          # proxy for solar intensity
        ],
        "timezone": TIMEZONE,
    },
    timeout=30,
)
wx_resp.raise_for_status()
wx_df = pd.DataFrame(wx_resp.json()["hourly"])

# ── 3. Merge on timestamp ─────────────────────────────────────────────────────
df = pd.merge(aq_df, wx_df, on="time", how="inner")
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)
df.sort_index(inplace=True)

print(f"\n🔗  Merged shape  : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} cols)")

# ── 4. Inspect missingness BEFORE cleaning ────────────────────────────────────
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print("\n📋  Missing % per column (before cleaning):")
print(missing_pct[missing_pct > 0].to_string() if missing_pct.any() else "   None — all columns complete ✅")

# ── 5. Drop columns that are >40 % empty (unusable) ──────────────────────────
bad_cols = missing_pct[missing_pct > 40].index.tolist()
if bad_cols:
    df.drop(columns=bad_cols, inplace=True)
    print(f"\n🗑️   Dropped (>40% missing): {bad_cols}")

# ── 6. Fill remaining gaps ────────────────────────────────────────────────────
# Time-series safe: forward-fill up to 3 hours, then backward-fill any edge NaNs
df.ffill(limit=3, inplace=True)
df.bfill(limit=3, inplace=True)

# Any column still NaN after that → fill with column median (last resort)
for col in df.columns[df.isnull().any()]:
    df[col].fillna(df[col].median(), inplace=True)

# ── 7. Remove duplicate timestamps ───────────────────────────────────────────
dupes = df.index.duplicated().sum()
if dupes:
    df = df[~df.index.duplicated(keep="first")]
    print(f"🔄  Removed {dupes} duplicate timestamp(s)")

# ── 8. Enforce correct dtypes ─────────────────────────────────────────────────
df = df.apply(pd.to_numeric, errors="coerce")

# ── 9. Sanity-check ranges (flag but don't drop) ──────────────────────────────
checks = {
    "temperature_2m"       : (-10, 55),
    "relative_humidity_2m" : (0, 100),
    "us_aqi"               : (0, 500),
    "pm2_5"                : (0, 999),
    "wind_speed_10m"       : (0, 200),
}
print("\n🔎  Range checks:")
for col, (lo, hi) in checks.items():
    if col not in df.columns:
        continue
    out = ((df[col] < lo) | (df[col] > hi)).sum()
    status = "✅ OK" if out == 0 else f"⚠️  {out} out-of-range values"
    print(f"   {col:<28} [{lo}, {hi}]  →  {status}")
    # clip obvious sensor errors
    df[col] = df[col].clip(lower=lo, upper=hi)

# ── 10. Final validation ──────────────────────────────────────────────────────
assert df.isnull().sum().sum() == 0, "❌  Still has NaNs after cleaning!"
print(f"\n✅  Final dataset  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Date range     : {df.index.min()}  →  {df.index.max()}")
print(f"   Missing values : {df.isnull().sum().sum()}  (none)")

# ── 11. Save ──────────────────────────────────────────────────────────────────
df.to_csv(OUT_FILE)
print(f"\n💾  Saved →  {OUT_FILE}")
print("\n📊  Preview:")
print(df.head(3).to_string())
print("\n📈  Quick stats:")
print(df.describe().round(2).to_string())