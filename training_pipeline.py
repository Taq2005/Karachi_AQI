"""
AQI Forecast Model Training Pipeline
=====================================
Models : Ridge Regression, Random Forest, LightGBM, XGBoost
Target : mean daily AQI for each of the next 3 days
Storage: best model → MongoDB GridFS + model_registry collection
Run    : python train_aqi_model.py
"""

import os, pickle, hashlib, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import gridfs

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.multioutput   import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
load_dotenv()

# ── Optional boosting libraries ───────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not found  →  pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not found   →  pip install xgboost")

# ── MongoDB ───────────────────────────────────────────────────────────────────
client   = MongoClient(os.getenv("MONGO_URI"))
db       = client[os.getenv("MONGO_DB", "karachi_aqi_weather")]
fs       = gridfs.GridFS(db)
FEAT_COL = os.getenv("MONGO_COLLECTION", "hourly_features")
REG_COL  = "model_registry"

HORIZON   = 3
TEST_FRAC = 0.15

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print("\n📦  Loading feature store …")
    docs = list(db[FEAT_COL].find({}, {"_id": 0}).sort("time", 1))
    if not docs:
        raise RuntimeError("Feature store is empty — run the pipeline first.")
    df = pd.DataFrame(docs)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    print(f"    {len(df):,} hourly rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (daily level, hand-crafted)
# ══════════════════════════════════════════════════════════════════════════════

# Raw columns to carry into daily aggregation (exclude hourly-only engineered fields)
RAW_WEATHER = [
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide",
    "carbon_monoxide", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "surface_pressure", "cloud_cover", "visibility",
]


def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly → daily, then build meaningful lag/rolling features
    at the daily level so they carry real signal.
    """
    # Keep only raw sensor columns + target
    keep = [c for c in RAW_WEATHER + ["us_aqi"] if c in df.columns]
    daily = df[keep].resample("D").mean(numeric_only=True)
    daily.dropna(subset=["us_aqi"], inplace=True)

    aqi = daily["us_aqi"]

    # ── Lag features (previous N days of AQI) ────────────────────────────────
    for lag in [1, 2, 3, 7]:
        daily[f"aqi_lag_{lag}d"] = aqi.shift(lag)

    # ── Rolling stats ─────────────────────────────────────────────────────────
    daily["aqi_roll7_mean"] = aqi.shift(1).rolling(7).mean()
    daily["aqi_roll7_std"]  = aqi.shift(1).rolling(7).std()
    daily["aqi_roll3_mean"] = aqi.shift(1).rolling(3).mean()

    # ── Trend (yesterday minus 3-day-ago) ─────────────────────────────────────
    daily["aqi_trend3d"]    = aqi.shift(1) - aqi.shift(3)

    # ── Calendar ──────────────────────────────────────────────────────────────
    daily["month"]      = daily.index.month
    daily["day_of_week"] = daily.index.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    # ── PM ratio at daily level ───────────────────────────────────────────────
    daily["pm_ratio"] = daily["pm2_5"] / (daily["pm10"] + 1e-9)

    daily.dropna(inplace=True)
    return daily


def build_dataset(daily: pd.DataFrame):
    """
    Each row is one day; X = all features except us_aqi.
    y = next-3-day mean AQI (one row per starting day).
    No flattened window needed — features already encode history via lags.
    """
    feature_cols = [c for c in daily.columns if c != "us_aqi"]
    target_arr   = daily["us_aqi"].values
    feat_arr     = daily[feature_cols].values

    X_rows, y_rows = [], []
    for i in range(len(daily) - HORIZON):
        X_rows.append(feat_arr[i])
        y_rows.append(target_arr[i + 1 : i + 1 + HORIZON])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows,  dtype=np.float32)
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"    [{name:<15}]  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SKLEARN PIPELINE BUILDER
#     Each model gets:  SimpleImputer → (optional scaler) → estimator
# ══════════════════════════════════════════════════════════════════════════════

def make_pipeline(estimator, scale: bool = False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_ridge(X_tr, y_tr, X_te, y_te):
    print("\n🔵  Ridge Regression …")
    pipe = make_pipeline(
        MultiOutputRegressor(Ridge(alpha=10.0)), scale=True
    )
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    metrics = evaluate("Ridge", y_te, y_pred)
    return metrics, pipe


def train_rf(X_tr, y_tr, X_te, y_te):
    print("\n🌲  Random Forest …")
    pipe = make_pipeline(
        MultiOutputRegressor(
            RandomForestRegressor(n_estimators=400, max_depth=8,
                                  min_samples_leaf=3, n_jobs=-1, random_state=42)
        )
    )
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    metrics = evaluate("RandomForest", y_te, y_pred)
    return metrics, pipe


def train_lgb(X_tr, y_tr, X_te, y_te):
    if not HAS_LGB:
        return None, None
    print("\n💡  LightGBM …")
    pipe = make_pipeline(
        MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               max_depth=5, num_leaves=20,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.1, reg_lambda=1.0,
                               min_child_samples=5,
                               n_jobs=-1, random_state=42, verbosity=-1)
        )
    )
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    metrics = evaluate("LightGBM", y_te, y_pred)
    return metrics, pipe


def train_xgb(X_tr, y_tr, X_te, y_te):
    if not HAS_XGB:
        return None, None
    print("\n⚡  XGBoost …")
    pipe = make_pipeline(
        MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                              max_depth=5, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.1,
                              reg_lambda=1.0, min_child_weight=3,
                              tree_method="hist", random_state=42, verbosity=0)
        )
    )
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    metrics = evaluate("XGBoost", y_te, y_pred)
    return metrics, pipe


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def save_to_registry(name: str, pipeline, metrics: dict, feature_cols: list):
    print(f"\n💾  Saving '{name}' to MongoDB model registry …")

    payload = {
        "model_name":   name,
        "feature_cols": feature_cols,
        "horizon":      HORIZON,
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
        "horizon":         HORIZON,
        "gridfs_file_id":  file_id,
        "gridfs_filename": fname,
        "sha256":          sha,
        "is_active":       True,
    })
    print(f"    ✅  Saved  |  file_id={file_id}  |  SHA={sha}  |  size={len(raw)/1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def forecast_next_3_days(daily: pd.DataFrame, feature_cols: list,
                         pipeline) -> pd.DataFrame:
    # Use the most recent daily row as the input features
    last_row = daily[feature_cols].iloc[[-1]].values.astype(np.float32)
    pred     = pipeline.predict(last_row)[0]

    last  = daily.index[-1]
    dates = [last + pd.Timedelta(days=i + 1) for i in range(HORIZON)]
    return pd.DataFrame({"date": dates, "predicted_aqi": np.round(pred, 1)})


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load & prepare ────────────────────────────────────────────────────────
    df    = load_data()
    daily = make_daily_features(df)
    print(f"    {len(daily)} daily rows after feature engineering")

    X, y, feature_cols = build_dataset(daily)
    print(f"    Dataset : X={X.shape}  y={y.shape}  features={len(feature_cols)}")

    split = int(len(X) * (1 - TEST_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  MODEL EXPERIMENTS")
    print("═" * 58)

    results, pipelines = [], {}

    m, p = train_ridge(X_tr, y_tr, X_te, y_te)
    results.append(m); pipelines["Ridge"] = p

    m, p = train_rf(X_tr, y_tr, X_te, y_te)
    results.append(m); pipelines["RandomForest"] = p

    m, p = train_lgb(X_tr, y_tr, X_te, y_te)
    if m: results.append(m); pipelines["LightGBM"] = p

    m, p = train_xgb(X_tr, y_tr, X_te, y_te)
    if m: results.append(m); pipelines["XGBoost"] = p

    # ── Leaderboard ───────────────────────────────────────────────────────────
    lb = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    lb.index += 1
    print("\n" + "═" * 58)
    print("  LEADERBOARD  (↑ lower RMSE = better)")
    print("═" * 58)
    print(lb.to_string())

    best_name    = lb.iloc[0]["model"]
    best_metrics = lb.iloc[0].to_dict()
    print(f"\n🏆  Best model : {best_name}  (RMSE={best_metrics['rmse']:.3f})")

    # ── Save winner ───────────────────────────────────────────────────────────
    save_to_registry(best_name, pipelines[best_name], best_metrics, feature_cols)

    # ── Forecast ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  3-DAY AQI FORECAST")
    print("═" * 58)
    fc = forecast_next_3_days(daily, feature_cols, pipelines[best_name])
    print(fc.to_string(index=False))

    fc_docs = fc.to_dict("records")
    for d in fc_docs:
        d["date"]         = pd.Timestamp(d["date"]).to_pydatetime()
        d["generated_at"] = datetime.now(timezone.utc)
        d["model_used"]   = best_name

    db["aqi_forecasts"].delete_many({})
    db["aqi_forecasts"].insert_many(fc_docs)
    print("\n✅  Forecast written to 'aqi_forecasts' collection.")

    client.close()
    print("🔒  Connection closed.")


if __name__ == "__main__":
    main()