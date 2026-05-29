from datetime import datetime, timezone, timedelta
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aab-O-Hawa | Karachi AQI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — CSS custom properties adapt to Streamlit's own theme vars ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Tokens: light defaults, dark overrides via media query ── */
:root {
    --bg-base:        #F5F7FA;
    --bg-card:        #FFFFFF;
    --bg-card2:       #EEF2F8;
    --border:         #D8E2F0;
    --border-accent:  #3B82F6;
    --text-primary:   #0F1C2E;
    --text-secondary: #4A6080;
    --text-muted:     #3A5070;
    --accent:         #2563EB;
    --accent-glow:    rgba(37,99,235,0.12);
    --line-subtle:    rgba(37,99,235,0.15);
    --chart-grid:     #E4EAF4;
    --shadow:         0 2px 16px rgba(15,28,46,0.08);
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg-base:        #080C14;
        --bg-card:        #0F1A2E;
        --bg-card2:       #0A1220;
        --border:         #1A2E4A;
        --border-accent:  #2E6BE6;
        --text-primary:   #E8EDF5;
        --text-secondary: #5A7FA8;
        --text-muted:     #6A90B8;
        --accent:         #2E6BE6;
        --accent-glow:    rgba(46,107,230,0.10);
        --line-subtle:    rgba(46,107,230,0.18);
        --chart-grid:     #0F1E38;
        --shadow:         0 2px 24px rgba(0,0,0,0.4);
    }
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; }
#MainMenu, footer         { visibility: hidden; }

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-accent), transparent);
}

/* ── AQI hero ── */
.aqi-hero {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 24px;
    text-align: center;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.aqi-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.aqi-number {
    font-family: 'Syne', sans-serif;
    font-size: 88px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -4px;
}
.aqi-label {
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 6px;
}
.aqi-category {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}
.aqi-pills {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}
.aqi-pill-item .pill-label {
    font-size: 9px;
    letter-spacing: 2px;
    color: var(--text-muted);
    text-transform: uppercase;
}
.aqi-pill-item .pill-value {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: var(--shadow);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-label {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Forecast cards ── */
.forecast-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: var(--shadow);
    transition: border-color 0.2s, transform 0.2s;
}
.forecast-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
}
.forecast-date {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.forecast-aqi {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    line-height: 1.1;
    margin: 8px 0 4px;
}
.forecast-cat {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 10px;
}
.forecast-range {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 6px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--line-subtle), transparent);
}

/* ── Top bar ── */
.topbar-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}
.topbar-sub {
    font-size: 9px;
    letter-spacing: 4px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 2px;
}
.model-badge {
    display: inline-block;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
}
.status-live {
    font-size: 10px;
    letter-spacing: 2px;
    color: #22C55E;
    text-transform: uppercase;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22C55E;
    box-shadow: 0 0 6px #22C55E;
    margin-right: 5px;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 28px 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 9px;
    letter-spacing: 2px;
    color: var(--text-muted);
    padding-bottom: 24px;
    text-transform: uppercase;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def get_db():   # ← no @st.cache_resource
    uri     = st.secrets.get("MONGO_URI",  os.getenv("MONGO_URI"))
    db_name = st.secrets.get("MONGO_DB",   os.getenv("MONGO_DB", "karachi_aqi"))
    client  = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client[db_name]

@st.cache_data(ttl=300)
def load_current_aqi():
    return get_db()["hourly_features"].find_one(sort=[("time", -1)])

@st.cache_data(ttl=300)
def load_daily_forecast():
    docs = list(get_db()["aqi_forecasts"].find({}, {"_id": 0}).sort("date", 1))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

@st.cache_data(ttl=300)
def load_hourly_forecast():
    docs = list(get_db()["aqi_forecasts_hourly"].find({}, {"_id": 0}).sort("datetime", 1))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

@st.cache_data(ttl=300)
def load_historical(days=14):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    docs  = list(get_db()["hourly_features"].find(
        {"time": {"$gte": since}},
        {"_id": 0, "time": 1, "us_aqi": 1}
    ).sort("time", 1))
    return pd.DataFrame(docs) if docs else pd.DataFrame()

@st.cache_data(ttl=300)
def load_model_meta():
    return get_db()["model_registry"].find_one({"is_active": True})

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

AQI_BANDS = [
    (50,  "#16A34A", "Good"),
    (100, "#CA8A04", "Moderate"),
    (150, "#EA580C", "Unhealthy for Sensitive Groups"),
    (200, "#DC2626", "Unhealthy"),
    (300, "#9333EA", "Very Unhealthy"),
    (999, "#7F1D1D", "Hazardous"),
]

def aqi_color(val):
    for t, c, _ in AQI_BANDS:
        if val <= t: return c
    return "#7F1D1D"

def aqi_category(val):
    for t, _, l in AQI_BANDS:
        if val <= t: return l
    return "Hazardous"

# Plotly layout — transparent bg so it inherits page theme
def chart_layout(height=280):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono", size=11),
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(showgrid=False, zeroline=False, showline=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)",
                   zeroline=False, showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
    )

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    current    = load_current_aqi()   or {}
    daily_fc   = load_daily_forecast()
    hourly_fc  = load_hourly_forecast()
    historical = load_historical(14)
    model_meta = load_model_meta()    or {}

    cur_aqi  = int(current.get("us_aqi", 0))
    cur_time = current.get("time", datetime.now(timezone.utc))

    # ── Top bar ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("""
        <div style="padding:6px 0 12px;">
            <div class="topbar-title">🌫️ Aab-O-Hawa</div>
            <div class="topbar-sub">Karachi Air Quality Intelligence</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        ts = pd.Timestamp(cur_time).strftime("%d %b %Y, %H:%M") if cur_time else "—"
        st.markdown(f"""
        <div style="text-align:right; padding-top:14px;">
            <span class="status-dot"></span>
            <span class="status-live">Live</span><br>
            <span style="font-size:10px; color:var(--text-muted);">{ts} PKT</span>
        </div>""", unsafe_allow_html=True)

    with c3:
        m = model_meta.get("metrics", {})
        st.markdown(f"""
        <div style="text-align:right; padding-top:14px;">
            <div class="model-badge">⚙ {model_meta.get('model_name','—')}</div>
            <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
                RMSE {m.get('rmse', 0):.2f} &nbsp;·&nbsp; R² {m.get('r2', 0):.3f}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Current AQI + conditions ──────────────────────────────────────────────
    col_hero, col_cond = st.columns([1, 2], gap="large")

    with col_hero:
        color = aqi_color(cur_aqi)
        cat   = aqi_category(cur_aqi)
        st.markdown(f"""
        <div class="aqi-hero">
            <div class="aqi-label">Current AQI · Karachi</div>
            <div class="aqi-number" style="color:{color};">{cur_aqi}</div>
            <div class="aqi-category" style="color:{color};">{cat}</div>
            <div class="aqi-pills">
                <div class="aqi-pill-item">
                    <div class="pill-label">PM2.5</div>
                    <div class="pill-value">{current.get('pm2_5', 0):.1f}</div>
                </div>
                <div class="aqi-pill-item">
                    <div class="pill-label">PM10</div>
                    <div class="pill-value">{current.get('pm10', 0):.1f}</div>
                </div>
                <div class="aqi-pill-item">
                    <div class="pill-label">O₃</div>
                    <div class="pill-value">{current.get('ozone', 0):.1f}</div>
                </div>
                <div class="aqi-pill-item">
                    <div class="pill-label">NO₂</div>
                    <div class="pill-value">{current.get('nitrogen_dioxide', 0):.1f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_cond:
        st.markdown('<div class="section-header">Atmospheric Conditions</div>',
                    unsafe_allow_html=True)
        row1 = st.columns(4)
        row2 = st.columns(4)
        cond_metrics = [
            (f"{current.get('temperature_2m', 0):.1f}°C",        "Temperature"),
            (f"{current.get('relative_humidity_2m', 0):.0f}%",   "Humidity"),
            (f"{current.get('wind_speed_10m', 0):.1f} km/h",     "Wind Speed"),
            (f"{current.get('surface_pressure', 0):.0f} hPa",    "Pressure"),
            (f"{current.get('sulphur_dioxide', 0):.1f} μg/m³",   "SO₂"),
            (f"{current.get('carbon_monoxide', 0):.0f} μg/m³",   "CO"),
            (f"{current.get('wind_gusts_10m', 0):.1f} km/h",      "Wind Gusts"),
            (f"{current.get('cloud_cover', 0):.0f}%",            "Cloud Cover"),
        ]
        for i, (val, label) in enumerate(cond_metrics):
            col = row1[i] if i < 4 else row2[i - 4]
            with col:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:10px;">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 3-Day forecast cards ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">3-Day Forecast</div>',
                unsafe_allow_html=True)

    if not daily_fc.empty:
        cols = st.columns(3, gap="medium")
        for i, (_, row) in enumerate(daily_fc.iterrows()):
            aqi_val  = int(row.get("predicted_aqi", 0))
            color    = aqi_color(aqi_val)
            cat      = aqi_category(aqi_val)
            date_str = pd.Timestamp(row["date"]).strftime("%A, %d %b")
            with cols[i]:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-date">{date_str}</div>
                    <div class="forecast-aqi" style="color:{color};">{aqi_val}</div>
                    <div class="forecast-cat" style="color:{color};">{cat}</div>
                    <div class="forecast-range">
                        ↓ {int(row.get('hourly_min', 0))} &nbsp;·&nbsp;
                        ↑ {int(row.get('hourly_max', 0))}
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No forecast data yet — run the pipeline first.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 72-hour hourly forecast chart ─────────────────────────────────────────
    st.markdown('<div class="section-header">72-Hour Hourly Forecast</div>',
                unsafe_allow_html=True)

    if not hourly_fc.empty:
        hourly_fc["datetime"] = pd.to_datetime(hourly_fc["datetime"])
        fig = go.Figure()

        # Subtle AQI band fills
        for lo, hi, color, label in [
            (0,   50,  "rgba(22,163,74,0.07)",   "Good"),
            (50,  100, "rgba(202,138,4,0.07)",    "Moderate"),
            (100, 150, "rgba(234,88,12,0.07)",    "Unhealthy·S"),
            (150, 200, "rgba(220,38,38,0.07)",    "Unhealthy"),
            (200, 300, "rgba(147,51,234,0.07)",   "Very Unhealthy"),
        ]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=color,
                          line_width=0, layer="below")

        fig.add_trace(go.Scatter(
            x=hourly_fc["datetime"],
            y=hourly_fc["predicted_aqi_hourly"],
            mode="lines",
            name="Predicted AQI",
            line=dict(color="#2563EB", width=2.5,
                      shape="spline", smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.07)",
            hovertemplate="<b>%{y:.0f}</b> AQI<extra></extra>",
        ))

        # Day separator lines
        start = hourly_fc["datetime"].iloc[0].normalize()
        for d in range(1, 4):
            fig.add_vline(
                x=(start + pd.Timedelta(days=d)).timestamp() * 1000,
                line_color="rgba(128,128,128,0.2)",
                line_dash="dot", line_width=1,
            )

        fig.update_layout(**chart_layout(280))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("No hourly forecast available.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Historical trend + distribution ───────────────────────────────────────
    col_hist, col_dist = st.columns([3, 2], gap="large")

    with col_hist:
        st.markdown('<div class="section-header">14-Day Historical Trend</div>',
                    unsafe_allow_html=True)
        if not historical.empty:
            historical["time"]    = pd.to_datetime(historical["time"])
            historical["roll24"]  = historical["us_aqi"].rolling(24).mean()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=historical["time"], y=historical["us_aqi"],
                mode="lines", name="AQI",
                line=dict(color="rgba(37,99,235,0.5)", width=1.2,
                          shape="spline", smoothing=0.6),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.05)",
                hovertemplate="<b>%{y:.0f}</b><extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=historical["time"], y=historical["roll24"],
                mode="lines", name="24h Avg",
                line=dict(color="#EA580C", width=2, dash="dot"),
                hovertemplate="<b>%{y:.1f}</b> avg<extra></extra>",
            ))
            fig2.update_layout(**chart_layout(240))
            st.plotly_chart(fig2, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No historical data available.")

    with col_dist:
        st.markdown('<div class="section-header">AQI Distribution</div>',
                    unsafe_allow_html=True)
        if not historical.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=historical["us_aqi"],
                nbinsx=20,
                marker=dict(
                    color=historical["us_aqi"].apply(aqi_color),
                    line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                ),
                hovertemplate="AQI %{x}: <b>%{y}</b> hrs<extra></extra>",
                name="",
            ))
            layout3 = chart_layout(240)
            layout3["bargap"]          = 0.05
            layout3["xaxis"]["title"]  = "AQI"
            layout3["yaxis"]["title"]  = "Hours"
            layout3["showlegend"]      = False
            fig3.update_layout(**layout3)
            st.plotly_chart(fig3, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No data available.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Model performance ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance</div>',
                unsafe_allow_html=True)

    m      = model_meta.get("metrics", {})
    trained = model_meta.get("trained_at", None)
    trained_str = (pd.Timestamp(trained).strftime("%d %b %Y, %H:%M")
                   if trained else "—")

    mc = st.columns(5)
    perf = [
        (model_meta.get("model_name", "—"), "Active Model"),
        (f"{m.get('rmse', 0):.3f}",         "RMSE"),
        (f"{m.get('mae',  0):.3f}",         "MAE"),
        (f"{m.get('r2',   0):.4f}",         "R²"),
        (trained_str,                        "Last Trained"),
    ]
    for col, (val, label) in zip(mc, perf):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:18px; color:var(--accent);">
                    {val}
                </div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        Aab-O-Hawa &nbsp;·&nbsp; Karachi Air Quality Intelligence
        &nbsp;·&nbsp; Data: Open-Meteo &nbsp;·&nbsp; Model retrained daily
    </div>""", unsafe_allow_html=True)


main()
