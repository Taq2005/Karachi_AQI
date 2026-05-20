import joblib
import requests
import numpy as np
import pandas as pd

LAT, LON  = 24.8607, 67.0011
TIMEZONE  = "Asia/Karachi"
TODAY     = pd.Timestamp.now().strftime("%Y-%m-%d")
YESTERDAY = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")

# ─── 1. Load trained models, scaler, features ────────────────────
models   = joblib.load("saved_models/models.pkl")
scaler   = joblib.load("saved_models/scaler.pkl")
FEATURES = joblib.load("saved_models/features.pkl")

print("✅ Loaded models, scaler and features\n")

# ─── 2. Fetch last 2 days from Open-Meteo ────────────────────────
aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
    "latitude": LAT, "longitude": LON,
    "start_date": YESTERDAY, "end_date": TODAY,
    "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
               "sulphur_dioxide", "carbon_monoxide", "us_aqi"],
    "timezone": TIMEZONE,
}, timeout=30).json()["hourly"]

wx = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": LAT, "longitude": LON,
    "start_date": YESTERDAY, "end_date": TODAY,
    "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
               "apparent_temperature", "wind_speed_10m", "wind_direction_10m",
               "wind_gusts_10m", "precipitation", "surface_pressure",
               "cloud_cover", "visibility", "shortwave_radiation"],
    "timezone": TIMEZONE,
}, timeout=30).json()["hourly"]

# ─── 3. Merge & engineer features (same as training) ─────────────
today_df = pd.merge(pd.DataFrame(aq), pd.DataFrame(wx), on="time")
today_df["time"] = pd.to_datetime(today_df["time"])
today_df.ffill(limit=3, inplace=True)
today_df.bfill(limit=3, inplace=True)
today_df.fillna(0, inplace=True)

today_df["hour"]            = today_df["time"].dt.hour
today_df["day_of_week"]     = today_df["time"].dt.dayofweek
today_df["month"]           = today_df["time"].dt.month
today_df["is_weekend"]      = (today_df["day_of_week"] >= 5).astype(int)
today_df["hour_sin"]        = np.sin(2 * np.pi * today_df["hour"] / 24)
today_df["hour_cos"]        = np.cos(2 * np.pi * today_df["hour"] / 24)
today_df["dow_sin"]         = np.sin(2 * np.pi * today_df["day_of_week"] / 7)
today_df["dow_cos"]         = np.cos(2 * np.pi * today_df["day_of_week"] / 7)
today_df["pm_ratio"]        = today_df["pm2_5"] / (today_df["pm10"] + 1e-9)
today_df["aqi_change_rate"] = today_df["us_aqi"].diff().fillna(0)
today_df["aqi_lag_1h"]      = today_df["us_aqi"].shift(1)
today_df["aqi_lag_24h"]     = today_df["us_aqi"].shift(24)
today_df["aqi_roll_24h"]    = today_df["us_aqi"].rolling(24).mean()

today_df.dropna(inplace=True)

# ─── 4. Latest complete hour ──────────────────────────────────────
latest     = today_df.iloc[-1]
actual_aqi = latest["us_aqi"]

input_scaled = scaler.transform(latest[FEATURES].values.reshape(1, -1))

# ─── 5. All 4 models predict & compare ───────────────────────────
print(f"Time       : {latest['time']}")
print(f"Actual AQI : {round(actual_aqi)}\n")
print(f"{'─'*47}")
print(f"  {'Model':<20} {'Predicted':>9} {'Error':>7} {'Error%':>7}")
print(f"{'─'*47}")

for name, model in models.items():
    try:
        predicted = model.predict(input_scaled)[0]
        error     = abs(predicted - actual_aqi)
        pct       = (error / actual_aqi) * 100
        print(f"  {name:<20} {round(predicted):>9} {error:>7.2f} {pct:>6.1f}%")
    except Exception as e:
        print(f"  {name:<20} ❌ Error: {e}")
print(f"{'─'*47}")