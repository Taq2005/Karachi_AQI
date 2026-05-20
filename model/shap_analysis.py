"""
Read features from MongoDB → Train RF → SHAP Analysis
pip install pymongo dnspython scikit-learn shap matplotlib pandas numpy python-dotenv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

load_dotenv()

# ── 1. Fetch from MongoDB ─────────────────────────────────────────────────────
print("🔗  Connecting to MongoDB …")
client     = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=10000)
db         = client[os.getenv("MONGO_DB", "karachi_aqi")]
collection = db[os.getenv("MONGO_COLLECTION", "hourly_features")]

print("📥  Reading data …")
df = pd.DataFrame(list(collection.find({}, {"_id": 0})))  # exclude _id field
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)
client.close()
print(f"✅  {len(df)} rows loaded  |  {df['time'].min()}  →  {df['time'].max()}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
for lag in [1, 3, 6, 24]:
    df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)
df["aqi_roll_24h"]    = df["us_aqi"].rolling(24).mean()
df["aqi_change_rate"] = df["us_aqi"].diff()
df["pm_ratio"]        = df["pm2_5"] / (df["pm10"] + 1e-9)
df["hour_sin"]        = np.sin(2 * np.pi * df["time"].dt.hour / 24)
df["hour_cos"]        = np.cos(2 * np.pi * df["time"].dt.hour / 24)
df.dropna(inplace=True)
print(f"✅  After feature engineering: {df.shape}")

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
FEATURES = [c for c in df.columns if c not in ["time", "us_aqi"]]
TARGET   = "us_aqi"

X, y  = df[FEATURES], df[TARGET]
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("\n🌲  Training Random Forest …")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"   RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.2f}  R²={r2_score(y_test, y_pred):.4f}")

# ── 5. SHAP ───────────────────────────────────────────────────────────────────
print("\n🔍  Computing SHAP values …")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Global bar chart
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
plt.title("SHAP — Feature Importance"); plt.tight_layout()
plt.savefig("shap_importance.png", bbox_inches="tight"); plt.close()
print("💾  Saved: shap_importance.png")

# Beeswarm
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title("SHAP — Beeswarm"); plt.tight_layout()
plt.savefig("shap_beeswarm.png", bbox_inches="tight"); plt.close()
print("💾  Saved: shap_beeswarm.png")

# Waterfall for worst prediction
worst = int(np.argmax(y_pred))

# fix: extract scalar base value
base_val = explainer.expected_value
if hasattr(base_val, "__len__"):
    base_val = float(base_val[0])
else:
    base_val = float(base_val)

shap.waterfall_plot(shap.Explanation(
    values        = shap_values[worst],
    base_values   = base_val,
    data          = X_test.iloc[worst].values,
    feature_names = FEATURES
), max_display=12, show=False)
plt.tight_layout()
plt.savefig("shap_waterfall.png", bbox_inches="tight"); plt.close()
print("💾  Saved: shap_waterfall.png")
# ── 6. Top features table ─────────────────────────────────────────────────────
top = pd.DataFrame({
    "feature"      : FEATURES,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\n🏆  Top 10 features by SHAP:")
print(top.head(10).to_string(index=False))