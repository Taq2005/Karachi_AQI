import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import io
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
load_dotenv()

# ─── 1. Fetch data ────────────────────────────────────────────────
client   = MongoClient(os.getenv("MONGO_URI"))
db       = client[os.getenv("MONGO_DB", "karachi_aqi_weather")]
col      = db[os.getenv("MONGO_COLLECTION", "hourly_features")]
registry = db["model_registry"]
fc_col   = db["aqi_forecasts"]
fc_h_col = db["aqi_forecasts_hourly"]

df = pd.DataFrame(list(col.find({}, {"_id": 0})))
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"Loaded {len(df):,} rows")

# ─── 2. Resample to daily ─────────────────────────────────────────
daily = df.set_index("time").resample("D").mean(numeric_only=True).reset_index()
daily.fillna(daily.median(numeric_only=True), inplace=True)
print(f"Daily rows: {len(daily)}")

# ─── 3. Feature engineering ───────────────────────────────────────
for lag in [1, 2, 3, 7]:
    daily[f"aqi_lag_{lag}d"] = daily["us_aqi"].shift(lag)
daily["aqi_roll_7d"]     = daily["us_aqi"].rolling(7).mean()
daily["aqi_roll_14d"]    = daily["us_aqi"].rolling(14).mean()
daily["aqi_roll_std_7d"] = daily["us_aqi"].rolling(7).std()
daily["aqi_change_rate"] = daily["us_aqi"].diff()
daily["pm_ratio"]        = daily["pm2_5"] / (daily["pm10"] + 1e-9)
daily["dow_sin"]         = np.sin(2 * np.pi * daily["time"].dt.dayofweek / 7)
daily["dow_cos"]         = np.cos(2 * np.pi * daily["time"].dt.dayofweek / 7)
daily["month"]           = daily["time"].dt.month
daily["is_weekend"]      = (daily["time"].dt.dayofweek >= 5).astype(int)

# targets
daily["target_day1"] = daily["us_aqi"].shift(-1)
daily["target_day2"] = daily["us_aqi"].shift(-2)
daily["target_day3"] = daily["us_aqi"].shift(-3)

daily.dropna(subset=["target_day1","target_day2","target_day3"], inplace=True)
daily.fillna(daily.median(numeric_only=True), inplace=True)
daily.reset_index(drop=True, inplace=True)
print(f"After engineering: {daily.shape}")

# ─── 4. Features & split ─────────────────────────────────────────
EXCLUDE  = ["time","us_aqi","target_day1","target_day2","target_day3"]
FEATURES = [c for c in daily.columns if c not in EXCLUDE]
DAYS     = {"Day 1":"target_day1","Day 2":"target_day2","Day 3":"target_day3"}

split      = int(len(daily) * 0.8)
last_row   = daily[FEATURES].iloc[[-1]].values   # for forecast

# ─── 5. Train per day & save to MongoDB ───────────────────────────
all_results = []
forecast    = {}

for day, target_col in DAYS.items():
    print(f"\n── {day} ──────────────────────────────")
    X = daily[FEATURES].values
    y = daily[target_col].values

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest"    : (RandomForestRegressor(n_estimators=300, max_depth=12,
                                min_samples_leaf=2, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.05, random_state=42), True),
        "XGBoost"          : (XGBRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=42,
                                verbosity=0), True),
    }

    day_results = []
    print(f"  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>8}")
    print(f"  {'─'*48}")

    for name, (model, use_scale) in models.items():
        Xtr = X_train_sc if use_scale else X_train
        Xte = X_test_sc  if use_scale else X_test
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae  = float(mean_absolute_error(y_test, pred))
        r2   = float(r2_score(y_test, pred))
        print(f"  {name:<22} {rmse:>7.2f} {mae:>7.2f} {r2:>8.4f}")
        day_results.append({"model":name,"day":day,
                             "rmse":round(rmse,4),"mae":round(mae,4),"r2":round(r2,4)})

    # best model for this day
    best     = min(day_results, key=lambda x: x["rmse"])
    best_mdl, best_scale = models[best["model"]]
    print(f"\n  🏆 Best: {best['model']}  RMSE={best['rmse']}")

    # forecast next day
    inp      = scaler.transform(last_row) if best_scale else last_row
    pred_aqi = float(best_mdl.predict(inp)[0])
    forecast[day] = round(pred_aqi, 1)

    # save to MongoDB registry
    scaler_buf = io.BytesIO(); joblib.dump(scaler, scaler_buf)
    for name, (mdl, ns) in models.items():
        buf = io.BytesIO(); joblib.dump(mdl, buf)
        row = next(r for r in day_results if r["model"] == name)
        registry.update_one(
            {"model_name": name, "day": day},
            {"$set": {
                **row,
                "trained_at"  : datetime.now(timezone.utc),
                "features"    : FEATURES,
                "is_best"     : (name == best["model"]),
                "is_active"   : True,
                "needs_scale" : ns,
                "model_bytes" : buf.getvalue(),
                "scaler_bytes": scaler_buf.getvalue(),
            }},
            upsert=True
        )
    all_results.extend(day_results)

# ─── 6. Write forecast to MongoDB ────────────────────────────────
now  = datetime.now(timezone.utc)
last = daily["time"].max()

daily_docs = []
for i, (day, aqi) in enumerate(forecast.items()):
    fc_date = last + pd.Timedelta(days=i+1)
    daily_docs.append({
        "date"         : fc_date.date().isoformat(),
        "predicted_aqi": aqi,
        "hourly_min"   : round(aqi * 0.88, 1),
        "hourly_max"   : round(aqi * 1.12, 1),
        "model_used"   : "ensemble",
        "generated_at" : now,
    })

fc_col.delete_many({})
fc_col.insert_many(daily_docs)
print(f"\n✅  Forecast written to aqi_forecasts:")
for d in daily_docs:
    print(f"   {d['date']}  AQI={d['predicted_aqi']}")

# ─── 7. Save training_run for comparison script ───────────────────
db["training_run"].replace_one({}, {
    "saved_at"   : now,
    "features"   : FEATURES,
    "all_results": all_results,
}, upsert=True)

client.close()
print("\n🔒 Done")
