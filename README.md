# 🌫️ Aab-O-Hawa — Karachi Air Quality Intelligence

A fully automated, end-to-end air quality monitoring and forecasting system built for **Karachi, Pakistan**. Tracks real-time AQI, forecasts the next 3 days using machine learning, and displays everything on a live Streamlit dashboard.

[![Feature Pipeline](https://github.com/Taq2005/Karachi_AQI/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/Taq2005/Karachi_AQI/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/Taq2005/Karachi_AQI/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/Taq2005/Karachi_AQI/actions/workflows/training_pipeline.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![MongoDB](https://img.shields.io/badge/Database-MongoDB_Atlas-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 🖥️ Live Dashboard

> **[View Live App →](https://your-app-url.streamlit.app)**  
> Updates automatically every hour · Forecast retrained daily
![Dashboard Preview](assets/dashboard.png)
![Forecast Cards](assets/forecast.png)
![Trends](assets/trend.png)
---

## 📸 Preview

| Current AQI | 3-Day Forecast | Pollutant Trends |
|-------------|----------------|------------------|
| Live AQI badge with health category | Day 1 / Day 2 / Day 3 predictions | PM2.5, PM10, O₃, NO₂ charts |

---

## 🏗️ Architecture

```
Open-Meteo API
├── Air Quality API  ──┐
└── Archive/Forecast ──┤
                       ▼
              Feature Pipeline          ← runs every hour
              (feature_pipeline.py)
                       │
                       ▼
              MongoDB Atlas
              ├── hourly_features       ← raw + engineered features
              ├── model_registry        ← trained models (serialized)
              ├── aqi_forecasts         ← 3-day daily forecast
              └── aqi_forecasts_hourly  ← 72-hour hourly forecast
                       │
              Training Pipeline         ← runs every day
              (train_aqi_model.py)
                       │
                       ▼
              Streamlit Dashboard       ← live on Streamlit Cloud
              (app.py)
```

All pipelines run automatically via **GitHub Actions** — no server required.

---

## ✨ Features

- 🔴 **Live AQI monitoring** — current readings updated every hour
- 🔮 **3-day AQI forecast** — Day 1, Day 2, Day 3 predictions with min/max range
- 🚨 **Health alerts** — automatic warnings for Unhealthy / Hazardous conditions
- 📊 **Pollutant breakdown** — PM2.5, PM10, Ozone, NO₂, SO₂, CO trends
- 🌡️ **Atmospheric conditions** — temperature, humidity, wind, pressure
- 📈 **Historical trend** — 14-day AQI trend with 24h rolling average
- 🤖 **Model transparency** — active model name, RMSE, R² shown in dashboard
- 🔄 **Fully automated** — data + training pipelines run on schedule

---

## 🧠 ML Models

Three models are trained and compared for each forecast horizon. The best performer (lowest RMSE) is automatically selected and saved to MongoDB.

| Model | Type | Notes |
|-------|------|-------|
| Random Forest | Ensemble (trees) | No scaling needed, handles non-linearity |
| Ridge Regression | Linear | Fast baseline, good for linear trends |
| XGBoost | Gradient Boosting | Best overall for tabular time-series |
| Gradient Boosting | Ensemble | Strong regularisation, robust to noise |

**Evaluation metrics:** RMSE · MAE · R²

**Explainability:** SHAP (SHapley Additive exPlanations) is used to identify which features drive AQI predictions.

---

## 📁 Project Structure

```
Karachi_AQI/
│
├── app.py                      # Streamlit dashboard
├── feature_pipeline.py         # Hourly data fetch → MongoDB
├── train_aqi_model.py          # Daily model training → MongoDB
├── requirements.txt            # Python dependencies
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Hourly GitHub Actions cron
│       └── training_pipeline.yml  # Daily GitHub Actions cron
│
├── model/
│   └── training_pipeline.py    # Model comparison script
│
└── eda/
    └── karachi_eda.ipynb       # Exploratory Data Analysis notebook
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- MongoDB Atlas account (free M0 tier)
- GitHub account

### 1. Clone the repo

```bash
git clone https://github.com/Taq2005/Karachi_AQI.git
cd Karachi_AQI
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows CMD
.venv\Scripts\activate.bat

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGO_DB=karachi_aqi_weather
MONGO_COLLECTION=hourly_features
```

> ⚠️ Never commit `.env` to GitHub. It is already in `.gitignore`.

### 5. Run the initial backfill (90 days of data)

```bash
python feature_pipeline.py
```

### 6. Train the models

```bash
python training_pipeline.py
```

### 7. Launch the dashboard

```bash
streamlit run app.py
```

---

## ⚙️ Automated Pipelines

### Feature Pipeline — runs every hour

Fetches the latest AQI + weather data from Open-Meteo and uploads new rows to MongoDB.

```yaml
# .github/workflows/feature_pipeline.yml
on:
  schedule:
    - cron: "0 * * * *"   # every hour
  workflow_dispatch:        # manual trigger
```

### Training Pipeline — runs every day

Reads all features from MongoDB, trains 4 models, saves the best to the model registry.

```yaml
# .github/workflows/training_pipeline.yml
on:
  schedule:
    - cron: "0 2 * * *"   # 2:00 AM UTC daily
  workflow_dispatch:
```

### GitHub Secrets required

Add these in **Settings → Secrets → Actions**:

| Secret | Description |
|--------|-------------|
| `MONGO_URI` | MongoDB Atlas connection string |
| `MONGO_DB` | Database name (`karachi_aqi_weather`) |
| `MONGO_COLLECTION` | Collection name (`hourly_features`) |

---

## 📦 Data Source

All data is fetched from **[Open-Meteo](https://open-meteo.com/)** — free, no API key required.

| Endpoint | Variables |
|----------|-----------|
| Air Quality API | PM2.5, PM10, NO₂, O₃, SO₂, CO, US AQI |
| Archive API | Temperature, humidity, wind, pressure, cloud cover |
| Forecast API | Same weather variables (current day, real-time) |

**Location:** Karachi, Pakistan · 24.8608°N, 67.0104°E · Timezone: Asia/Karachi

---

## 🌐 Deploying to Streamlit Cloud

1. Push all files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and `app.py` as the main file
4. Go to **Settings → Secrets** and add:

```toml
MONGO_URI        = "mongodb+srv://..."
MONGO_DB         = "karachi_aqi_weather"
MONGO_COLLECTION = "hourly_features"
```

5. Click **Deploy** — the app auto-deploys on every GitHub push

---

## 📊 AQI Scale Reference

| AQI Range | Category | Health Implication |
|-----------|----------|--------------------|
| 0–50 | 🟢 Good | Air quality is satisfactory |
| 51–100 | 🟡 Moderate | Acceptable for most people |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Elderly, children, asthma patients at risk |
| 151–200 | 🔴 Unhealthy | Everyone may experience health effects |
| 201–300 | 🟣 Very Unhealthy | Health alert — serious effects for everyone |
| 300+ | ⚫ Hazardous | Emergency conditions |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Data Source | Open-Meteo API |
| Feature Store | MongoDB Atlas |
| ML Models | scikit-learn, XGBoost |
| Explainability | SHAP |
| Automation | GitHub Actions |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Cloud |

---

## 📝 License

This project is open source under the [MIT License](LICENSE).

---

## 👤 Author

**Muhammad Taqee**  
Built as part of an end-to-end ML engineering project.

---

*آب و ہوا — Because every breath matters.*
