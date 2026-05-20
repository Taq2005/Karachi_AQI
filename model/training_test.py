import os
import warnings
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
load_dotenv()

# ─── 1. Fetch data from MongoDB ───────────────────────────────────
client = MongoClient(os.getenv("MONGO_URI"))
df = pd.DataFrame(list(
    client[os.getenv("MONGO_DB", "karachi_aqi")]
    [os.getenv("MONGO_COLLECTION", "hourly_features")]
    .find({}, {"_id": 0})
))
client.close()

df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"Loaded {len(df):,} rows\n")

# ─── 2. Features & Target ────────────────────────────────────────
EXCLUDE  = {"time", "us_aqi", "time_id"}
FEATURES = [c for c in df.columns if c not in EXCLUDE]
TARGET   = "us_aqi"

X = df[FEATURES].values
y = df[TARGET].values

# ─── 3. Train / Test split (80/20 chronological) ─────────────────
split   = int(len(X) * 0.80)
X_train = X[:split]
X_test  = X[split:]
y_train = y[:split]
y_test  = y[split:]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {split:,}  |  Test: {len(X)-split:,}  |  Features: {len(FEATURES)}\n")

# ─── 4. Define 4 models ──────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=15,
        min_samples_leaf=2, random_state=42, n_jobs=-1),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=5,
        learning_rate=0.05, random_state=42),

    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbosity=0, n_jobs=-1),

    "LightGBM": LGBMRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbosity=-1, n_jobs=-1),
}

# ─── 5. Train, test, compare ─────────────────────────────────────
print(f"{'─'*55}")
print(f"  {'Model':<20} {'RMSE':>7} {'MAE':>7} {'R²':>7}")
print(f"{'─'*55}")

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)

    print(f"  {name:<20} {rmse:>7.2f} {mae:>7.2f} {r2:>7.4f}")
    results.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2})

print(f"{'─'*55}")

# ─── 6. Best model ───────────────────────────────────────────────
best = min(results, key=lambda x: x["rmse"])
print(f"\n🏆  Best Model : {best['name']}")
print(f"    RMSE       : {best['rmse']:.2f}")
print(f"    MAE        : {best['mae']:.2f}")
print(f"    R²         : {best['r2']:.4f}")

# ─── 7. Save models, scaler, features to disk ────────────────────
os.makedirs("saved_models", exist_ok=True)

joblib.dump(models,    "saved_models/models.pkl")
joblib.dump(scaler,    "saved_models/scaler.pkl")
joblib.dump(FEATURES,  "saved_models/features.pkl")

print("\n✅ Saved models, scaler and features to saved_models/")