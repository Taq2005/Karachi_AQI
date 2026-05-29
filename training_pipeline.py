import os, json, pickle, hashlib, warnings
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import gridfs

from sklearn.linear_model  import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble      import RandomForestRegressor
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
load_dotenv()

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not found  ->  pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not found   ->  pip install xgboost")

# ── MongoDB ───────────────────────────────────────────────────────────────────
client   = MongoClient(os.getenv("MONGO_URI"))
db       = client[os.getenv("MONGO_DB", "karachi_aqi_weather")]
fs       = gridfs.GridFS(db)
FEAT_COL = os.getenv("MONGO_COLLECTION", "hourly_features")
REG_COL  = "model_registry"

LOOKBACK   = 48       # hours of history used to build features
FORECAST_H = 72       # hours to predict (3 days)
TEST_HOURS = 24 * 20  # hold out last 20 days as test set

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print("\n📦  Loading feature store ...")
    docs = list(db[FEAT_COL].find({}, {"_id": 0}).sort("time", 1))
    if not docs:
        raise RuntimeError("Feature store is empty.")
    df = pd.DataFrame(docs)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    print(f"    {len(df):,} hourly rows  |  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (hourly level)
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "precipitation", "surface_pressure", "cloud_cover",
    "shortwave_radiation",
]

AQI_LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    aqi = df["us_aqi"]

    for h in AQI_LAG_HOURS:
        df[f"aqi_lag_{h}h"] = aqi.shift(h)

    for w in [6, 12, 24]:
        s = aqi.shift(1)
        df[f"aqi_roll{w}h_mean"] = s.rolling(w, min_periods=1).mean()
        df[f"aqi_roll{w}h_std"]  = s.rolling(w, min_periods=2).std()

    df["aqi_diff1h"]  = aqi.shift(1).diff()
    df["aqi_diff24h"] = aqi.shift(1) - aqi.shift(25)

    df["hour"]       = df.index.hour
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"]        = df.index.dayofweek
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"]      = df.index.month

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    for col in df.select_dtypes("number").columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != "us_aqi"]


def make_xy(df: pd.DataFrame, feature_cols: list):
    feat   = df[feature_cols].values[LOOKBACK:-1]
    target = df["us_aqi"].values[LOOKBACK + 1:]
    return feat.astype(np.float32), target.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"    [{name:<15}]  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def make_pipeline(estimator, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_ridge(X_tr, y_tr, X_te, y_te):
    print("\n🔵  Ridge Regression ...")
    pipe = make_pipeline(Ridge(alpha=10.0), scale=True)
    pipe.fit(X_tr, y_tr)
    return evaluate("Ridge", y_te, pipe.predict(X_te)), pipe


def train_rf(X_tr, y_tr, X_te, y_te):
    print("\n🌲  Random Forest ...")
    pipe = make_pipeline(
        RandomForestRegressor(n_estimators=300, max_depth=10,
                              min_samples_leaf=3, n_jobs=-1, random_state=42)
    )
    pipe.fit(X_tr, y_tr)
    return evaluate("RandomForest", y_te, pipe.predict(X_te)), pipe


def train_lgb(X_tr, y_tr, X_te, y_te):
    if not HAS_LGB:
        return None, None
    print("\n💡  LightGBM ...")
    pipe = make_pipeline(
        lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                          max_depth=6, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=1.0,
                          min_child_samples=5,
                          n_jobs=-1, random_state=42, verbosity=-1)
    )
    pipe.fit(X_tr, y_tr)
    return evaluate("LightGBM", y_te, pipe.predict(X_te)), pipe


def train_xgb(X_tr, y_tr, X_te, y_te):
    if not HAS_XGB:
        return None, None
    print("\n⚡  XGBoost ...")
    pipe = make_pipeline(
        xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                         max_depth=6, subsample=0.8,
                         colsample_bytree=0.8,
                         reg_alpha=0.1, reg_lambda=1.0,
                         min_child_weight=3,
                         tree_method="hist", random_state=42, verbosity=0)
    )
    pipe.fit(X_tr, y_tr)
    return evaluate("XGBoost", y_te, pipe.predict(X_te)), pipe


# ══════════════════════════════════════════════════════════════════════════════
# 6.  RECURSIVE 72-HOUR FORECAST  ->  3-DAY DAILY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def recursive_forecast(pipeline, df: pd.DataFrame,
                       feature_cols: list):
    max_lag  = max(AQI_LAG_HOURS)
    seed_df  = df.tail(max(LOOKBACK, max_lag) + 2).copy()
    aqi_hist = list(seed_df["us_aqi"].values)

    wx_cols      = [c for c in WEATHER_COLS if c in df.columns]
    last_weather = seed_df[wx_cols].iloc[-1].to_dict()
    last_ts      = seed_df.index[-1]
    hourly_preds = []
    hist_mean      = float(df["us_aqi"].tail(24 * 7).mean())
    REVERSION_RATE = 0.015
    for step in range(FORECAST_H):
        next_ts = last_ts + pd.Timedelta(hours=step + 1)
        row = {col: last_weather[col] for col in wx_cols}

        for h in AQI_LAG_HOURS:
            row[f"aqi_lag_{h}h"] = aqi_hist[-h] if len(aqi_hist) >= h else np.nan

        hist_arr = np.array(aqi_hist)
        for w in [6, 12, 24]:
            window = hist_arr[-w:] if len(hist_arr) >= w else hist_arr
            row[f"aqi_roll{w}h_mean"] = float(window.mean())
            row[f"aqi_roll{w}h_std"]  = float(window.std()) if len(window) > 1 else 0.0

        row["aqi_diff1h"]  = (aqi_hist[-1] - aqi_hist[-2])  if len(aqi_hist) >= 2  else 0.0
        row["aqi_diff24h"] = (aqi_hist[-1] - aqi_hist[-25]) if len(aqi_hist) >= 25 else 0.0

        row["hour_sin"]   = np.sin(2 * np.pi * next_ts.hour / 24)
        row["hour_cos"]   = np.cos(2 * np.pi * next_ts.hour / 24)
        row["dow_sin"]    = np.sin(2 * np.pi * next_ts.dayofweek / 7)
        row["dow_cos"]    = np.cos(2 * np.pi * next_ts.dayofweek / 7)
        row["is_weekend"] = int(next_ts.dayofweek >= 5)
        row["month"]      = next_ts.month

        x = np.array([row.get(c, 0.0) for c in feature_cols],
                     dtype=np.float32).reshape(1, -1)

        raw_pred = max(0.0, float(pipeline.predict(x)[0]))

        # Mean reversion: nudge toward 7-day historical mean to prevent drift
        reversion = REVERSION_RATE * step * (hist_mean - raw_pred)
        pred_aqi  = max(0.0, raw_pred + reversion)
        aqi_hist.append(pred_aqi)
        hourly_preds.append({
            "datetime":            next_ts.isoformat(),
            "predicted_aqi_hourly": round(pred_aqi, 1),
        })

    hourly_df = pd.DataFrame(hourly_preds)
    hourly_df["_dt"] = pd.to_datetime(hourly_df["datetime"])

    last_date  = last_ts.normalize()
    daily_rows = []
    for d in range(1, 4):
        day  = last_date + pd.Timedelta(days=d)
        mask = (hourly_df["_dt"] >= day) & (hourly_df["_dt"] < day + pd.Timedelta(days=1))
        vals = hourly_df.loc[mask, "predicted_aqi_hourly"]
        daily_rows.append({
            "date":          day.date().isoformat(),
            "predicted_aqi": round(float(vals.mean()), 1),
            "hourly_min":    round(float(vals.min()),  1),
            "hourly_max":    round(float(vals.max()),  1),
        })

    hourly_df.drop(columns=["_dt"], inplace=True)
    return daily_rows, hourly_df.to_dict("records")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def save_to_registry(name: str, pipeline, metrics: dict, feature_cols: list):
    print(f"\n💾  Saving '{name}' to MongoDB model registry ...")
    payload = {
        "model_name":   name,
        "feature_cols": feature_cols,
        "lookback":     LOOKBACK,
        "forecast_h":   FORECAST_H,
        "pipeline":     pipeline,
    }
    raw   = pickle.dumps(payload, protocol=5)
    sha   = hashlib.sha256(raw).hexdigest()[:12]
    fname = f"aqi_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pkl"

    file_id = fs.put(raw, filename=fname, content_type="application/octet-stream")
    db[REG_COL].update_many({"is_active": True}, {"$set": {"is_active": False}})
    db[REG_COL].insert_one({
        "model_name":      name,
        "version":         datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "trained_at":      datetime.now(timezone.utc),
        "metrics":         metrics,
        "feature_cols":    feature_cols,
        "lookback_hours":  LOOKBACK,
        "forecast_hours":  FORECAST_H,
        "gridfs_file_id":  file_id,
        "gridfs_filename": fname,
        "sha256":          sha,
        "is_active":       True,
    })
    print(f"    ✅  Saved  |  file_id={file_id}  |  SHA={sha}  |  {len(raw)/1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  WRITE forecast.json  (picked up by Streamlit app)
# ══════════════════════════════════════════════════════════════════════════════

def write_forecast_json(daily: list, hourly: list,
                        best_name: str, metrics: dict):
    os.makedirs("data", exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used":   best_name,
        "metrics":      metrics,
        "daily":        daily,
        "hourly":       hourly,
    }
    with open("data/forecast.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("✅  data/forecast.json written")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data()
    df = build_features(df)
    feature_cols = get_feature_cols(df)
    print(f"    Features ({len(feature_cols)}): {feature_cols[:8]} ...")

    X, y = make_xy(df, feature_cols)
    print(f"    Dataset : X={X.shape}  y={y.shape}")

    if len(X) < TEST_HOURS + 100:
        raise RuntimeError(
            f"Only {len(X)} samples. Need at least {TEST_HOURS + 100}."
        )

    split = len(X) - TEST_HOURS
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    print(f"    Train: {len(X_tr):,} hrs  |  Test: {len(X_te):,} hrs")

    print("\n" + "=" * 58)
    print("  MODEL EXPERIMENTS  (single-step hourly prediction)")
    print("=" * 58)

    results, pipelines = [], {}

    m, p = train_ridge(X_tr, y_tr, X_te, y_te)
    results.append(m); pipelines["Ridge"] = p

    m, p = train_rf(X_tr, y_tr, X_te, y_te)
    results.append(m); pipelines["RandomForest"] = p

    m, p = train_lgb(X_tr, y_tr, X_te, y_te)
    if m: results.append(m); pipelines["LightGBM"] = p

    m, p = train_xgb(X_tr, y_tr, X_te, y_te)
    if m: results.append(m); pipelines["XGBoost"] = p

    lb = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    lb.index += 1
    print("\n" + "=" * 58)
    print("  LEADERBOARD")
    print("=" * 58)
    print(lb.to_string())

    best_name    = lb.iloc[0]["model"]
    best_metrics = lb.iloc[0].to_dict()
    print(f"\n🏆  Best model : {best_name}  (RMSE={best_metrics['rmse']:.3f})")

    save_to_registry(best_name, pipelines[best_name], best_metrics, feature_cols)

    print("\n" + "=" * 58)
    print("  RECURSIVE 72-HOUR FORECAST")
    print("=" * 58)

    daily_fc, hourly_fc = recursive_forecast(pipelines[best_name], df, feature_cols)

    print("\n  3-Day Daily Averages:")
    for row in daily_fc:
        print(f"    {row['date']}  avg={row['predicted_aqi']}  "
              f"min={row['hourly_min']}  max={row['hourly_max']}")

    # ── MongoDB ───────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)

    daily_docs = [{**r, "generated_at": now, "model_used": best_name}
                  for r in daily_fc]
    db["aqi_forecasts"].delete_many({})
    db["aqi_forecasts"].insert_many(daily_docs)

    hourly_docs = [{**r,
                    "datetime":     pd.Timestamp(r["datetime"]).to_pydatetime(),
                    "generated_at": now,
                    "model_used":   best_name}
                   for r in hourly_fc]
    db["aqi_forecasts_hourly"].delete_many({})
    db["aqi_forecasts_hourly"].insert_many(hourly_docs)

    print("✅  Daily forecast  -> 'aqi_forecasts'")
    print("✅  Hourly forecast -> 'aqi_forecasts_hourly'")

    # ── JSON for Streamlit ────────────────────────────────────────────────────
    write_forecast_json(daily_fc, hourly_fc, best_name, best_metrics)

    client.close()
    print("🔒  Connection closed.")


if __name__ == "__main__":
    main()
