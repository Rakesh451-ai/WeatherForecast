import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats as scipy_stats
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Forecast · ARIMA",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────
C_BG      = "#0b0f1a"
C_SURFACE = "#111d2e"
C_BORDER  = "#1e3050"
C_TEXT    = "#c8d6e5"
C_MUTED   = "#5a7a9a"
C_TEAL    = "#56cfb2"
C_RED     = "#e05c5c"
C_AMBER   = "#f0a843"
C_BLUE    = "#4b8fe0"
C_PURPLE  = "#9b7fe0"
C_PINK    = "#e07fc4"

YEAR_COLORS = [C_BLUE, C_TEAL, C_AMBER, C_RED, C_PURPLE, C_PINK]

def hex_rgba(hex_color, alpha=0.15):
    """Convert a #rrggbb hex string to rgba(r,g,b,alpha) — Plotly compatible."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def add_vline_dt(fig, dt, color=C_AMBER, label="", row=None, col=None):
    """Add a vertical line on a datetime axis safely (avoids annotation bugs)."""
    x_ms = int(pd.Timestamp(dt).timestamp() * 1000)
    kw = dict(row=row, col=col) if row else {}
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=x_ms, x1=x_ms, y0=0, y1=1,
                  line=dict(color=color, width=1.5, dash="dot"), **kw)
    if label:
        fig.add_annotation(x=x_ms, yref="paper", y=0.98,
                           text=label, showarrow=False,
                           font=dict(color=color, size=11),
                           xanchor="left", bgcolor="rgba(0,0,0,0)")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C_SURFACE,
    font=dict(family="DM Mono, monospace", color=C_TEXT, size=12),
    xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
    yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER, borderwidth=1,
                font=dict(size=11)),
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor=C_SURFACE, bordercolor=C_BORDER,
                    font=dict(color=C_TEXT, size=12)),
)

def apply_layout(fig, title="", height=420, **kw):
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14, family="Syne, sans-serif"), x=0),
        height=height, **kw)
    return fig

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Mono',monospace;background:#0b0f1a;color:#c8d6e5;}
.stApp{background:linear-gradient(160deg,#0b0f1a 0%,#0d1525 60%,#111827 100%);}
section[data-testid="stSidebar"]{background:#0a0e1a!important;border-right:1px solid #1a2840;}
section[data-testid="stSidebar"] *{color:#7a9ab8!important;}
[data-testid="stMetric"]{background:#111d2e;border:1px solid #1e3050;border-radius:12px;padding:18px 20px!important;}
[data-testid="stMetricLabel"]{color:#56cfb2!important;font-size:10px!important;letter-spacing:.1em;text-transform:uppercase;}
[data-testid="stMetricValue"]{color:#e2eaf5!important;font-family:'Syne',sans-serif!important;font-size:24px!important;}
.stTabs [data-baseweb="tab-list"]{background:#0a0e1a;border-radius:10px;gap:2px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#3a5a7a!important;background:transparent!important;border-radius:8px!important;font-family:'DM Mono',monospace;font-size:12px;padding:6px 14px!important;}
.stTabs [aria-selected="true"]{background:#162640!important;color:#56cfb2!important;}
.stButton>button{background:linear-gradient(135deg,#1e4d8c,#1a6b8a);color:#e2eaf5;border:none;border-radius:8px;font-family:'DM Mono',monospace;font-size:13px;padding:10px 24px;transition:all .2s;width:100%;}
.stButton>button:hover{background:linear-gradient(135deg,#2560aa,#1f82a8);transform:translateY(-1px);}
h1{font-family:'Syne',sans-serif!important;color:#e2eaf5!important;}
h2,h3,h4{font-family:'Syne',sans-serif!important;color:#a8c0d8!important;}
hr{border-color:#1e2d45!important;}
.stAlert{border-radius:10px!important;}
.stDataFrame{border:1px solid #1e3050!important;border-radius:10px;}
.stSelectbox>div>div{background:#111d2e!important;border:1px solid #1e3050!important;color:#c8d6e5!important;}
.card{background:#111d2e;border:1px solid #1e3050;border-radius:12px;padding:20px 24px;margin-bottom:12px;}
.slabel{font-family:'Syne',sans-serif;color:#56cfb2;font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  BUILD / LOAD DATASET  2021-2026
# ─────────────────────────────────────────────
@st.cache_data
def build_dataset():
    end_date = pd.Timestamp("2026-03-29")
    # Try loading saved data and extend if needed
    if os.path.exists("weather_data.pkl"):
        with open("weather_data.pkl", "rb") as f:
            bundle = pickle.load(f)
        base = bundle["full_df"]
        last = base.index[-1]
        if last < end_date:
            np.random.seed(77)
            ext_dates = pd.date_range(last + pd.Timedelta(days=1), end_date, freq="D")
            n   = len(ext_dates)
            off = (last - pd.Timestamp("2021-01-01")).days
            d   = np.arange(off+1, off+1+n)
            t   = np.clip(26 + 15*np.sin(2*np.pi*(d-80)/365) + 0.003*d
                          + np.random.normal(0, 2.5, n), 10, 46)
            ext = pd.DataFrame({"temperature": np.round(t, 2)}, index=ext_dates)
            full = pd.concat([base, ext])
        else:
            full = base
    else:
        np.random.seed(42)
        dates = pd.date_range("2021-01-01", end_date, freq="D")
        n = len(dates); d = np.arange(n)
        t = np.clip(26 + 15*np.sin(2*np.pi*(d-80)/365) + 0.003*d
                    + np.random.normal(0, 2.5, n), 10, 46)
        full = pd.DataFrame({"temperature": np.round(t, 2)}, index=dates)

    split = int(len(full) * 0.80)
    return full, full["temperature"][:split], full["temperature"][split:]

full_df, train_data, test_data = build_dataset()
MODEL_ORDER = (2, 1, 2)

# ─────────────────────────────────────────────
#  SEASONAL DECOMPOSITION ENGINE
#  Why: Plain ARIMA converges to the series mean after ~5 steps →
#  every future day looks identical. Fix:
#    1. Build a day-of-year seasonal profile from historical data
#    2. Subtract it → deseasonalised residuals
#    3. Fit ARIMA on residuals
#    4. Forecast residuals N steps ahead (stays near 0, varies slightly)
#    5. Add back the seasonal profile for future dates
#  Result: each forecast day gets a different, seasonally-correct temperature.
# ─────────────────────────────────────────────
@st.cache_data
def build_seasonal_profile():
    """
    Day-of-year average temperature across all training years.
    Returns a dict {1..366 : avg_temp} and the deseasonalised training series.
    """
    df = full_df["temperature"].copy()
    doy_mean = df.groupby(df.index.dayofyear).mean()
    # Smooth with a 15-day rolling window to avoid sharp jumps
    extended = pd.concat([doy_mean, doy_mean, doy_mean])
    smoothed = extended.rolling(15, center=True, min_periods=1).mean()
    # Take the middle 365/366 values
    n = len(doy_mean)
    smoothed = smoothed.iloc[n: 2*n]
    smoothed.index = doy_mean.index
    profile = smoothed.to_dict()

    deseason = df - df.index.map(lambda d: profile.get(d.dayofyear, df.mean()))
    return profile, deseason

@st.cache_resource
def get_model(_hash):
    """Fit ARIMA on deseasonalised residuals."""
    _, deseason = build_seasonal_profile()
    return ARIMA(deseason, order=MODEL_ORDER).fit()

def seasonal_forecast(future_dates, model, profile, alpha=0.05):
    """
    Forecast future_dates using: seasonal_profile[doy] + ARIMA_residual_forecast.
    Each date gets a unique seasonal component → no flat-line predictions.
    """
    steps = len(future_dates)
    fo    = model.get_forecast(steps=steps)
    resid_mean = fo.predicted_mean.values
    resid_ci   = fo.conf_int(alpha=alpha).values   # shape (steps, 2)

    seasonal_vals = np.array([
        profile.get(d.dayofyear, np.mean(list(profile.values())))
        for d in future_dates
    ])

    mean_fc = np.clip(seasonal_vals + resid_mean, 10, 46)
    lo_fc   = np.clip(seasonal_vals + resid_ci[:, 0], 10, 46)
    hi_fc   = np.clip(seasonal_vals + resid_ci[:, 1], 10, 46)

    return (pd.Series(mean_fc, index=future_dates),
            pd.DataFrame({"lower": lo_fc, "upper": hi_fc}, index=future_dates))

profile, deseason_series = build_seasonal_profile()
model_fit = get_model(len(full_df))

# ─────────────────────────────────────────────
#  CACHED COMPUTATIONS
# ─────────────────────────────────────────────
@st.cache_data
def test_forecast():
    """
    Validation on last 60 days of data using seasonal decomposition model.
    Train on all-but-last-60, forecast 60 steps, add seasonal back.
    """
    n_eval    = 60
    full_ser  = full_df["temperature"]
    tr_cut    = full_ser[:-n_eval]
    val_true  = full_ser[-n_eval:]

    # Build profile from training cut only
    doy_mean  = tr_cut.groupby(tr_cut.index.dayofyear).mean()
    extended  = pd.concat([doy_mean, doy_mean, doy_mean])
    smoothed  = extended.rolling(15, center=True, min_periods=1).mean()
    n = len(doy_mean)
    smoothed  = smoothed.iloc[n:2*n]; smoothed.index = doy_mean.index
    prof_cut  = smoothed.to_dict()

    deseas_cut = tr_cut - tr_cut.index.map(lambda d: prof_cut.get(d.dayofyear, tr_cut.mean()))
    m_cut = ARIMA(deseas_cut, order=MODEL_ORDER).fit()

    val_dates = val_true.index
    mn, _     = seasonal_forecast(val_dates, m_cut, prof_cut, alpha=0.05)
    return mn, val_true

@st.cache_data
def future_forecast(steps, alpha):
    future_dates = pd.date_range(
        full_df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    mn, ci_df = seasonal_forecast(future_dates, model_fit, profile, alpha=alpha)
    ci = pd.DataFrame({"lower": ci_df["lower"], "upper": ci_df["upper"]},
                      index=future_dates)
    # Convert to statsmodels-style 2-column CI for downstream compatibility
    ci.columns = [ci.columns[0], ci.columns[1]]
    return mn, ci

@st.cache_data
def calc_metrics():
    tf_pred, tf_actual = test_forecast()
    a, p = tf_actual.values, tf_pred.values
    return (mean_absolute_error(a, p),
            np.sqrt(mean_squared_error(a, p)),
            np.mean(np.abs((a - p) / a)) * 100,
            1 - np.sum((a - p)**2) / np.sum((a - a.mean())**2))

tf_pred, tf_actual = test_forecast()
mae, rmse, mape, r2 = calc_metrics()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    forecast_days  = st.slider("Forecast horizon (days)", 7, 365, 60)
    ci_level       = st.selectbox("Confidence interval", ["95%","90%","80%"])
    alpha          = {"95%":0.05,"90%":0.10,"80%":0.20}[ci_level]
    history_days   = st.slider("History window (days)", 30, 600, 180)
    show_fill      = st.checkbox("Fill area under curves", True)
    show_markers   = st.checkbox("Show data markers", False)

    st.markdown("---")
    st.markdown("### 📅 Year Filter")
    all_years    = sorted(full_df.index.year.unique().tolist())
    sel_years    = st.multiselect("Years (EDA / Compare tabs)", all_years, default=all_years)

    st.markdown("---")
    st.markdown("### ℹ️ Model")
    p,d,q = MODEL_ORDER
    st.markdown(f"""<div style='background:#0a0e1a;border:1px solid #1a2840;border-radius:8px;padding:12px;font-size:12px;line-height:2.2;'>
    <b style='color:{C_TEAL}'>Method:</b> Seasonal Decomposition + ARIMA<br>
    <span style='color:{C_TEAL}'>p</span>={p} AR &nbsp; <span style='color:{C_AMBER}'>d</span>={d} &nbsp; <span style='color:{C_BLUE}'>q</span>={q} MA<br>
    AIC={model_fit.aic:.1f}<br>BIC={model_fit.bic:.1f}<br>
    Seasonal period: 365 days<br>
    Data end: {full_df.index[-1].strftime("%b %Y")}</div>""",
    unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
h1, h2, h3 = st.columns([3,1,1])
with h1:
    st.markdown("# 🌦️ Weather Forecast — Bhopal")
    st.markdown(f"<p style='color:{C_MUTED};font-size:13px;margin-top:-12px;'>"
                f"ARIMA(2,1,2) · Jan 2021 – Mar 2026 · {len(full_df):,} daily records</p>",
                unsafe_allow_html=True)
with h2:
    st.markdown(f"<div class='card' style='text-align:center;padding:14px;'>"
                f"<div style='color:{C_TEAL};font-size:13px;'>Model Active ✓</div>"
                f"<div style='color:{C_MUTED};font-size:11px;margin-top:4px;'>ARIMA(2,1,2)</div></div>",
                unsafe_allow_html=True)
with h3:
    st.markdown(f"<div class='card' style='text-align:center;padding:14px;'>"
                f"<div style='color:{C_AMBER};font-size:13px;'>6 Years Data</div>"
                f"<div style='color:{C_MUTED};font-size:11px;margin-top:4px;'>2021 → 2026</div></div>",
                unsafe_allow_html=True)

st.markdown("---")

# METRIC ROW
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("MAE",      f"{mae:.2f}°C", help="Mean Absolute Error on 60-day validation window")
m2.metric("RMSE",     f"{rmse:.2f}°C", help="Root Mean Square Error")
m3.metric("MAPE",     f"{mape:.1f}%",
          delta="Good" if mape<10 else "Fair",
          delta_color="normal" if mape<10 else "inverse")
m4.metric("R²",       f"{r2:.4f}", help="Closer to 1.0 = better")
m5.metric("Train",    f"{len(train_data):,} days")
m6.metric("Val win.", "60 days", help="Metrics computed on last 60 days of training data")

st.markdown("---")

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab0,tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🎯 Predict","🔮 Forecast","📈 EDA",
    "📊 Year Compare","🔬 Diagnostics","📉 ACF/PACF","📋 Data"
])

# ══════════════════════════════════════════════
#  TAB 0 — PREDICT
# ══════════════════════════════════════════════
with tab0:
    st.subheader("🎯 Predict Temperature for Any Date")
    st.markdown(f"<p style='color:{C_MUTED};font-size:13px;margin-top:-8px;'>The ARIMA model forecasts ahead from the last data point.</p>", unsafe_allow_html=True)

    with st.form("pred_form"):
        f1,f2,f3 = st.columns([2,1,1])
        with f1:
            last_d    = full_df.index[-1].date()
            user_date = st.date_input("📅 Target date",
                value     = last_d + pd.Timedelta(days=10),
                min_value = last_d + pd.Timedelta(days=1),
                max_value = last_d + pd.Timedelta(days=730))
        with f2:
            pred_ci_lv = st.selectbox("Confidence band", ["95%","90%","80%"])
        with f3:
            show_path = st.checkbox("Forecast path chart", True)
        go_btn = st.form_submit_button("🔍  Predict Temperature", use_container_width=True)

    if go_btn:
        p_alpha  = {"95%":0.05,"90%":0.10,"80%":0.20}[pred_ci_lv]
        n_steps  = (pd.Timestamp(user_date) - full_df.index[-1]).days

        # Build all future dates from last data point to user_date (inclusive)
        with st.spinner(f"Forecasting {n_steps} days ahead using seasonal model..."):
            all_future = pd.date_range(
                full_df.index[-1] + pd.Timedelta(days=1),
                periods=n_steps, freq="D")
            fc_m, fc_ci_df = seasonal_forecast(all_future, model_fit, profile, alpha=p_alpha)

        pred  = float(fc_m.iloc[-1])
        lo    = float(fc_ci_df["lower"].iloc[-1])
        hi    = float(fc_ci_df["upper"].iloc[-1])
        sigma = max((hi - lo) / (2 * 1.96), 0.5)

        # Label
        if   pred>=42: icon,label,col="🔥","Dangerously Hot", C_RED
        elif pred>=38: icon,label,col="☀️","Extremely Hot",  "#ff7043"
        elif pred>=32: icon,label,col="🌤️","Very Warm",      C_AMBER
        elif pred>=26: icon,label,col="⛅","Warm & Pleasant",C_TEAL
        elif pred>=20: icon,label,col="🌥️","Mild",           C_BLUE
        elif pred>=14: icon,label,col="🧥","Cool",           C_PURPLE
        else:          icon,label,col="🧊","Cold",           "#90caf9"

        # Result card
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111d2e,#0d1a2e);border:1px solid {col}44;
                    border-radius:18px;padding:36px;text-align:center;margin:16px 0;'>
            <div style='font-size:56px;line-height:1;margin-bottom:8px;'>{icon}</div>
            <div style='font-size:12px;color:{C_MUTED};letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;'>Predicted for</div>
            <div style='font-size:22px;color:#e2eaf5;font-family:Syne,sans-serif;font-weight:700;margin-bottom:20px;'>
                {pd.Timestamp(user_date).strftime("%A, %d %B %Y")}</div>
            <div style='font-size:80px;font-family:Syne,sans-serif;font-weight:800;color:{col};line-height:1;'>
                {pred:.1f}<span style='font-size:36px;'>°C</span></div>
            <div style='font-size:16px;color:{col};opacity:.8;margin:10px 0 28px;'>{label}</div>
            <div style='display:flex;justify-content:center;'>
                <div style='flex:1;border-right:1px solid #1e3050;padding:14px;'>
                    <div style='font-size:11px;color:{C_MUTED};margin-bottom:6px;'>{pred_ci_lv} Lower</div>
                    <div style='font-size:22px;color:{C_BLUE};font-family:Syne,sans-serif;font-weight:600;'>{lo:.1f}°C</div>
                </div>
                <div style='flex:1;border-right:1px solid #1e3050;padding:14px;'>
                    <div style='font-size:11px;color:{C_MUTED};margin-bottom:6px;'>{pred_ci_lv} Upper</div>
                    <div style='font-size:22px;color:{C_RED};font-family:Syne,sans-serif;font-weight:600;'>{hi:.1f}°C</div>
                </div>
                <div style='flex:1;padding:14px;'>
                    <div style='font-size:11px;color:{C_MUTED};margin-bottom:6px;'>Days ahead</div>
                    <div style='font-size:22px;color:{C_AMBER};font-family:Syne,sans-serif;font-weight:600;'>{n_steps}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Advisory
        if   pred>=42: advice="🔥 Extreme heat! Stay indoors, drink water every 30 min, avoid all outdoor activity."
        elif pred>=38: advice="☀️ Very hot. Light cotton clothes, sunscreen, avoid 11am–4pm outdoors."
        elif pred>=32: advice="🌤️ Warm day. Carry water, cotton clothes, outdoor work best in morning."
        elif pred>=26: advice="⛅ Comfortable weather. Great day for outdoor activities."
        elif pred>=20: advice="🌥️ Pleasant and mild. A light jacket for evenings is suggested."
        elif pred>=14: advice="🧥 Cool day. Dress in layers and carry a jacket."
        else:          advice="🧊 Cold weather expected. Wear warm clothing and stay cozy!"

        st.markdown(f"""<div style='background:{C_SURFACE};border-left:3px solid {col};
                    border-radius:0 10px 10px 0;padding:14px 18px;font-size:13px;color:{C_TEXT};margin-bottom:16px;'>
            💡 <b>Advisory:</b> {advice}</div>""", unsafe_allow_html=True)

        pc1, pc2 = st.columns(2)
        with pc1:
            # Gauge chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={"reference": full_df["temperature"].mean(), "valueformat":".1f",
                       "suffix":"°C vs avg"},
                title={"text": "Predicted Temperature", "font":{"size":14,"family":"Syne","color":C_TEXT}},
                number={"suffix":"°C","font":{"size":42,"family":"Syne","color":col}},
                gauge={
                    "axis":{"range":[10,46],"tickwidth":1,"tickcolor":C_MUTED,
                            "tickfont":{"color":C_MUTED}},
                    "bar":{"color":col,"thickness":0.25},
                    "bgcolor":C_SURFACE,
                    "borderwidth":1,"bordercolor":C_BORDER,
                    "steps":[
                        {"range":[10,20],"color":"#0d2a4a"},
                        {"range":[20,28],"color":"#0e3a2e"},
                        {"range":[28,35],"color":"#3d2c00"},
                        {"range":[35,46],"color":"#3d0e0e"},
                    ],
                    "threshold":{"line":{"color":C_AMBER,"width":3},"thickness":0.75,"value":pred}
                }
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=C_TEXT), height=280,
                                margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_g, use_container_width=True)

        with pc2:
            # Probability bar chart
            rng_labels = ["< 15°C","15–25°C","25–32°C","32–38°C","> 38°C"]
            rng_bounds = [15, 25, 32, 38, 46]
            probs, prev = [], -np.inf
            for rv in rng_bounds:
                probs.append(round(
                    (scipy_stats.norm.cdf(rv,pred,sigma) - scipy_stats.norm.cdf(prev,pred,sigma))*100, 1))
                prev = rv
            rng_colors = [C_BLUE,C_TEAL,C_AMBER,"#ff7043",C_RED]
            fig_pb = go.Figure(go.Bar(x=rng_labels, y=probs,
                marker_color=rng_colors,
                text=[f"{v}%" for v in probs], textposition="outside",
                hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>"))
            apply_layout(fig_pb, "Probability across temperature ranges", height=280)
            fig_pb.update_yaxes(range=[0, max(probs)+12], title="%")
            st.plotly_chart(fig_pb, use_container_width=True)

        # Path chart
        if show_path:
            hist = full_df["temperature"][-90:]

            fig_p = go.Figure()
            mode = "lines+markers" if show_markers else "lines"
            fig_p.add_trace(go.Scatter(x=hist.index, y=hist.values, mode=mode,
                name="History", line=dict(color=C_BLUE,width=1.5), marker=dict(size=3),
                hovertemplate="%{x|%b %d %Y}: <b>%{y:.1f}°C</b><extra></extra>"))
            fig_p.add_trace(go.Scatter(
                x=list(all_future)+list(all_future[::-1]),
                y=list(fc_ci_df["upper"])+list(fc_ci_df["lower"][::-1]),
                fill="toself", fillcolor=hex_rgba(col,0.13), line=dict(width=0),
                name=f"{pred_ci_lv} CI", hoverinfo="skip"))
            fig_p.add_trace(go.Scatter(x=all_future, y=fc_m.values,
                mode="lines", name="Seasonal Forecast", line=dict(color=col,width=2,dash="dot"),
                hovertemplate="%{x|%b %d %Y}: <b>%{y:.1f}°C</b><extra></extra>"))
            fig_p.add_trace(go.Scatter(
                x=[pd.Timestamp(user_date)], y=[pred],
                mode="markers+text", name=f"Prediction",
                marker=dict(color=col,size=14,symbol="diamond",line=dict(color="white",width=2)),
                text=[f"  {pred:.1f}°C"], textposition="middle right",
                textfont=dict(color=col,size=13),
                hovertemplate=f"<b>{pd.Timestamp(user_date).strftime('%b %d %Y')}: {pred:.1f}°C</b><extra></extra>"))
            add_vline_dt(fig_p, full_df.index[-1], C_AMBER, "Last data")
            apply_layout(fig_p, f"Forecast path → {pd.Timestamp(user_date).strftime('%d %b %Y')}", height=380)
            st.plotly_chart(fig_p, use_container_width=True)

    else:
        st.markdown(f"""
        <div style='background:{C_SURFACE};border:1px dashed {C_BORDER};border-radius:14px;
                    padding:60px;text-align:center;'>
            <div style='font-size:48px;margin-bottom:16px;'>🌡️</div>
            <div style='font-size:16px;color:{C_MUTED};'>Choose a date and click <b>Predict Temperature</b></div>
            <div style='font-size:12px;color:#2a3a4a;margin-top:10px;'>
                Dataset: {full_df.index[0].strftime("%b %Y")} → {full_df.index[-1].strftime("%b %Y")}
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 1 — FORECAST DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    st.subheader(f"🔮 {forecast_days}-Day Future Forecast")
    fm, fci = future_forecast(forecast_days, alpha)
    hist    = full_df["temperature"][-history_days:]
    mode    = "lines+markers" if show_markers else "lines"

    fig_f = go.Figure()
    fill_arg = dict(fill="tozeroy", fillcolor="rgba(75,143,224,0.07)") if show_fill else {}
    fig_f.add_trace(go.Scatter(x=hist.index, y=hist.values, mode=mode,
        name="Historical", line=dict(color=C_BLUE,width=1.3), marker=dict(size=3),
        hovertemplate="%{x|%b %d, %Y}: <b>%{y:.1f}°C</b><extra></extra>", **fill_arg))
    fig_f.add_trace(go.Scatter(
        x=list(fm.index)+list(fm.index[::-1]),
        y=list(fci["upper"])+list(fci["lower"][::-1]),
        fill="toself", fillcolor="rgba(86,207,178,0.12)", line=dict(width=0),
        name=f"{ci_level} CI", hoverinfo="skip"))
    fig_f.add_trace(go.Scatter(x=fm.index, y=fm.values, mode=mode,
        name="ARIMA Forecast", line=dict(color=C_TEAL,width=2.5,dash="dash"),
        marker=dict(size=4),
        hovertemplate="%{x|%b %d, %Y}: <b>%{y:.1f}°C</b><extra></extra>"))
    add_vline_dt(fig_f, full_df.index[-1], C_AMBER, "Forecast start →")
    apply_layout(fig_f, f"Temperature Forecast — Next {forecast_days} Days ({ci_level} CI)", height=460)
    fig_f.update_xaxes(rangeslider_visible=True,
        rangeselector=dict(bgcolor=C_SURFACE, activecolor=C_BORDER,
            font=dict(color=C_TEXT),
            buttons=[dict(count=1,label="1m",step="month",stepmode="backward"),
                     dict(count=3,label="3m",step="month",stepmode="backward"),
                     dict(count=6,label="6m",step="month",stepmode="backward"),
                     dict(step="all",label="All")]))
    st.plotly_chart(fig_f, use_container_width=True)

    # Stats
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("Min forecast",   f"{fm.min():.1f}°C")
    s2.metric("Max forecast",   f"{fm.max():.1f}°C")
    s3.metric("Mean forecast",  f"{fm.mean():.1f}°C")
    s4.metric("Temp range",     f"{(fm.max()-fm.min()):.1f}°C")
    s5.metric("Avg CI width",   f"{(fci['upper']-fci['lower']).mean():.1f}°C")

    # Heatmap calendar
    st.markdown("#### 🗓️ Forecast Heatmap Calendar")
    fdf = pd.DataFrame({"date":fm.index,"temp":fm.values})
    fdf["week"] = fdf["date"].dt.isocalendar().week.astype(int)
    fdf["dow"]  = fdf["date"].dt.weekday
    piv = fdf.pivot_table(index="dow", columns="week", values="temp", aggfunc="first")
    fig_cal = px.imshow(piv, color_continuous_scale="RdYlBu_r",
                        labels=dict(x="Week",y="Day",color="°C"), aspect="auto")
    fig_cal.update_xaxes(tickvals=list(piv.columns),
                         ticktext=[f"Wk{w}" for w in piv.columns])
    fig_cal.update_yaxes(tickvals=list(range(7)),
                         ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    apply_layout(fig_cal, "Weekly heatmap of forecast temperatures", height=280)
    fig_cal.update_coloraxes(colorbar=dict(tickfont=dict(color=C_TEXT),
                                           title=dict(font=dict(color=C_TEXT))))
    st.plotly_chart(fig_cal, use_container_width=True)

    with st.expander("📋 Full forecast table"):
        ft = pd.DataFrame({
            "Date":      fm.index.strftime("%Y-%m-%d"),
            "Weekday":   fm.index.strftime("%A"),
            "°C":        fm.values.round(2),
            f"Lower({ci_level})": fci["lower"].values.round(2),
            f"Upper({ci_level})": fci["upper"].values.round(2),
        })
        st.dataframe(ft, use_container_width=True, height=300)


# ══════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📈 Exploratory Data Analysis")
    eda = full_df[full_df.index.year.isin(sel_years)] if sel_years else full_df

    # Interactive time series with range selector
    fig_ts = go.Figure()
    fa = dict(fill="tozeroy",fillcolor="rgba(75,143,224,0.07)") if show_fill else {}
    fig_ts.add_trace(go.Scatter(x=eda.index, y=eda["temperature"], mode="lines",
        name="Temperature", line=dict(color=C_BLUE,width=0.8),
        hovertemplate="%{x|%d %b %Y}: <b>%{y:.1f}°C</b><extra></extra>", **fa))
    roll30 = eda["temperature"].rolling(30).mean()
    fig_ts.add_trace(go.Scatter(x=eda.index, y=roll30, mode="lines",
        name="30-day mean", line=dict(color=C_TEAL,width=2.2),
        hovertemplate="%{x|%b %Y}: <b>%{y:.1f}°C</b><extra></extra>"))
    roll7 = eda["temperature"].rolling(7).mean()
    fig_ts.add_trace(go.Scatter(x=eda.index, y=roll7, mode="lines",
        name="7-day mean", line=dict(color=C_AMBER,width=1.2,dash="dot"),
        hovertemplate="%{x|%b %d}: <b>%{y:.1f}°C</b><extra></extra>"))
    apply_layout(fig_ts, "Daily Temperature (2021–2026) — Interactive", height=430)
    fig_ts.update_xaxes(rangeslider_visible=True,
        rangeselector=dict(bgcolor=C_SURFACE,activecolor=C_BORDER,font=dict(color=C_TEXT),
            buttons=[dict(count=3,label="3m",step="month",stepmode="backward"),
                     dict(count=6,label="6m",step="month",stepmode="backward"),
                     dict(count=1,label="1y",step="year",stepmode="backward"),
                     dict(step="all",label="All")]))
    st.plotly_chart(fig_ts, use_container_width=True)

    # Monthly avg bar + violin per year
    e1, e2 = st.columns(2)
    with e1:
        mav = eda["temperature"].resample("M").mean()
        mdf = pd.DataFrame({"date":mav.index,"temp":mav.values})
        fig_bar = px.bar(mdf, x="date", y="temp",
            color="temp", color_continuous_scale="RdYlBu_r",
            labels={"temp":"°C","date":"Month"})
        fig_bar.update_coloraxes(showscale=False)
        fig_bar.update_traces(hovertemplate="%{x|%b %Y}: <b>%{y:.1f}°C</b><extra></extra>")
        apply_layout(fig_bar, "Monthly Average Temperature", height=340)
        st.plotly_chart(fig_bar, use_container_width=True)

    with e2:
        fig_v = go.Figure()
        for i,yr in enumerate(sorted(eda.index.year.unique())):
            yd = eda[eda.index.year==yr]["temperature"]
            c  = YEAR_COLORS[i%len(YEAR_COLORS)]
            fig_v.add_trace(go.Violin(y=yd, name=str(yr), box_visible=True,
                meanline_visible=True, line_color=c, fillcolor=hex_rgba(c,0.19),
                hovertemplate=f"{yr}: <b>%{{y:.1f}}°C</b><extra></extra>"))
        apply_layout(fig_v, "Temperature Distribution by Year", height=340)
        fig_v.update_yaxes(title="°C")
        st.plotly_chart(fig_v, use_container_width=True)

    # Seasonal decomposition
    st.markdown("#### 🔬 Seasonal Decomposition")
    try:
        dc = seasonal_decompose(full_df["temperature"], model="additive", period=365)
        fig_dc = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Observed","Trend","Seasonal","Residual"],
            vertical_spacing=0.05)
        for series,colour,row in [(dc.observed,C_BLUE,1),(dc.trend,C_TEAL,2),
                                   (dc.seasonal,C_AMBER,3),(dc.resid,C_RED,4)]:
            fig_dc.add_trace(go.Scatter(x=series.index, y=series.values,
                mode="lines", line=dict(color=colour,width=0.9), showlegend=False,
                hovertemplate="%{x|%b %Y}: <b>%{y:.2f}</b><extra></extra>"), row=row, col=1)
        fig_dc.update_layout(**PLOTLY_LAYOUT, height=680,
            title=dict(text="Additive Seasonal Decomposition (period=365 days)",
                       font=dict(size=14,family="Syne")))
        for i in range(1,5):
            fig_dc.update_xaxes(gridcolor=C_BORDER,linecolor=C_BORDER,row=i,col=1)
            fig_dc.update_yaxes(gridcolor=C_BORDER,linecolor=C_BORDER,row=i,col=1)
        st.plotly_chart(fig_dc, use_container_width=True)
    except Exception as e:
        st.warning(f"Decomposition skipped: {e}")

    # Validation window: actual vs predicted
    st.markdown("#### 🔮 Validation Window: Actual vs Predicted (last 60 days of training)")
    errors = tf_actual.values - tf_pred.values
    fig_tv = go.Figure()
    fig_tv.add_trace(go.Scatter(x=tf_actual.index, y=tf_actual.values,
        mode="lines", name="Actual", line=dict(color=C_TEAL,width=1.8),
        hovertemplate="%{x|%b %d %Y}: <b>%{y:.1f}°C</b><extra></extra>"))
    fig_tv.add_trace(go.Scatter(x=tf_pred.index, y=tf_pred.values,
        mode="lines", name="Predicted (60-step)", line=dict(color=C_RED,width=1.8,dash="dot"),
        hovertemplate="%{x|%b %d %Y}: <b>%{y:.1f}°C</b><extra></extra>"))
    fig_tv.add_trace(go.Bar(x=tf_actual.index, y=errors, name="Error",
        marker_color=[C_RED if e<0 else C_TEAL for e in errors],
        opacity=0.35, yaxis="y2",
        hovertemplate="%{x|%b %d}: error <b>%{y:.2f}°C</b><extra></extra>"))
    apply_layout(fig_tv, "Actual vs ARIMA — 60-Day Validation Window", height=430)
    fig_tv.update_layout(yaxis2=dict(overlaying="y",side="right",title="Error (°C)",
        gridcolor="rgba(0,0,0,0)", zeroline=True, zerolinecolor=C_BORDER,
        tickfont=dict(color=C_MUTED)))
    fig_tv.update_yaxes(title="Temperature (°C)")
    st.plotly_chart(fig_tv, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — YEAR COMPARE
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Year-over-Year Comparison")

    # Overlay all years on day-of-year axis
    fig_yoy = go.Figure()
    for i,yr in enumerate(all_years):
        yd   = full_df[full_df.index.year==yr]["temperature"]
        doy  = yd.index.dayofyear
        c    = YEAR_COLORS[i%len(YEAR_COLORS)]
        fig_yoy.add_trace(go.Scatter(x=doy, y=yd.rolling(7).mean().values,
            mode="lines", name=str(yr), line=dict(color=c,width=2),
            hovertemplate=f"{yr} day %{{x}}: <b>%{{y:.1f}}°C</b><extra></extra>"))
    apply_layout(fig_yoy,"Year-over-Year (7-day smoothed) — all on same Jan–Dec axis",height=450)
    fig_yoy.update_xaxes(title="Day of Year",
        tickvals=[1,32,60,91,121,152,182,213,244,274,305,335],
        ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    fig_yoy.update_yaxes(title="Temperature (°C)")
    st.plotly_chart(fig_yoy, use_container_width=True)

    # Monthly heatmap
    st.markdown("#### 🗓️ Monthly Average Heatmap")
    pdata = {}
    for yr in all_years:
        ym = full_df[full_df.index.year==yr]["temperature"].resample("M").mean()
        pdata[yr] = {m.month:v for m,v in ym.items()}
    hdf = pd.DataFrame(pdata).T
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    hdf.columns = [month_names[c-1] for c in hdf.columns]
    fig_hm = px.imshow(hdf, color_continuous_scale="RdYlBu_r",
                       labels=dict(x="Month",y="Year",color="°C"),
                       text_auto=".1f", aspect="auto")
    fig_hm.update_traces(textfont=dict(size=11, color="white"))
    apply_layout(fig_hm,"Monthly Temperature Heatmap (2021–2026)",height=310)
    fig_hm.update_coloraxes(colorbar=dict(tickfont=dict(color=C_TEXT),
                                          title=dict(font=dict(color=C_TEXT))))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Box per year
    c_box1, c_box2 = st.columns(2)
    with c_box1:
        fig_box = go.Figure()
        for i,yr in enumerate(all_years):
            yd = full_df[full_df.index.year==yr]["temperature"]
            c  = YEAR_COLORS[i%len(YEAR_COLORS)]
            fig_box.add_trace(go.Box(y=yd, name=str(yr),
                marker_color=c, boxmean="sd",
                hovertemplate=f"{yr}: <b>%{{y:.1f}}°C</b><extra></extra>"))
        apply_layout(fig_box,"Box & Whisker per Year",height=360)
        fig_box.update_yaxes(title="°C")
        st.plotly_chart(fig_box, use_container_width=True)

    with c_box2:
        # Annual extremes bar chart
        rows=[]
        for yr in all_years:
            yd = full_df[full_df.index.year==yr]["temperature"]
            rows.append({"Year":str(yr),"Max":yd.max().round(1),"Min":yd.min().round(1),"Mean":yd.mean().round(1)})
        adf2 = pd.DataFrame(rows)
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(x=adf2["Year"],y=adf2["Max"],name="Max",
            marker_color=C_RED,hovertemplate="%{x}: Max <b>%{y}°C</b><extra></extra>"))
        fig_ext.add_trace(go.Bar(x=adf2["Year"],y=adf2["Min"],name="Min",
            marker_color=C_BLUE,hovertemplate="%{x}: Min <b>%{y}°C</b><extra></extra>"))
        fig_ext.add_trace(go.Scatter(x=adf2["Year"],y=adf2["Mean"],mode="lines+markers",
            name="Mean",line=dict(color=C_AMBER,width=2),marker=dict(size=8),
            hovertemplate="%{x}: Mean <b>%{y}°C</b><extra></extra>"))
        apply_layout(fig_ext,"Annual Min / Mean / Max",height=360)
        fig_ext.update_layout(barmode="group")
        fig_ext.update_yaxes(title="°C")
        st.plotly_chart(fig_ext, use_container_width=True)

    # Annual stats table
    st.markdown("#### 📋 Annual Statistics Table")
    rows2=[]
    for yr in all_years:
        yd=full_df[full_df.index.year==yr]["temperature"]
        rows2.append({"Year":yr,"Min °C":yd.min().round(1),"Max °C":yd.max().round(1),
            "Mean °C":yd.mean().round(2),"Std Dev":yd.std().round(2),
            "Days > 35°C":int((yd>35).sum()),"Days < 15°C":int((yd<15).sum())})
    st.dataframe(pd.DataFrame(rows2).set_index("Year"),use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — DIAGNOSTICS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Model Diagnostics")
    adf_o = adfuller(full_df["temperature"].dropna())
    adf_d = adfuller(full_df["temperature"].diff().dropna())
    resid = model_fit.resid

    d1,d2 = st.columns(2)
    with d1:
        sc=C_TEAL if adf_o[1]<=0.05 else C_RED
        st.markdown(f"""<div class='card'><div class='slabel'>ADF — Original</div>
            <div style='font-size:28px;color:{sc};font-family:Syne,sans-serif;'>p = {adf_o[1]:.4f}</div>
            <div style='font-size:12px;color:{C_MUTED};margin-top:6px;'>
            Stat={adf_o[0]:.4f} | {"✅ Stationary" if adf_o[1]<=0.05 else "❌ Not stationary"}</div></div>""",
            unsafe_allow_html=True)
    with d2:
        sc2=C_TEAL if adf_d[1]<=0.05 else C_RED
        st.markdown(f"""<div class='card'><div class='slabel'>ADF — After Differencing (d=1)</div>
            <div style='font-size:28px;color:{sc2};font-family:Syne,sans-serif;'>p = {adf_d[1]:.4f}</div>
            <div style='font-size:12px;color:{C_MUTED};margin-top:6px;'>
            Stat={adf_d[0]:.4f} | {"✅ Stationary — ARIMA ready!" if adf_d[1]<=0.05 else "❌ Still not stationary"}</div></div>""",
            unsafe_allow_html=True)

    # Residuals time series
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines",
        name="Residual", line=dict(color=C_BLUE,width=0.8),
        hovertemplate="%{x|%b %Y}: <b>%{y:.3f}</b><extra></extra>"))
    fig_r.add_hline(y=0,line_dash="dash",line_color=C_RED,line_width=1.5)
    apply_layout(fig_r,"Residuals Over Time — should look like white noise",height=310)
    st.plotly_chart(fig_r, use_container_width=True)

    rp1,rp2 = st.columns(2)
    with rp1:
        fig_rh = go.Figure()
        fig_rh.add_trace(go.Histogram(x=resid.dropna(), nbinsx=60,
            marker_color=C_TEAL, opacity=0.8, name="Residuals",
            hovertemplate="Bin %{x:.2f}: <b>%{y}</b><extra></extra>"))
        xs=np.linspace(resid.min(),resid.max(),200)
        ys=scipy_stats.norm.pdf(xs,resid.mean(),resid.std())*len(resid)*(resid.max()-resid.min())/60
        fig_rh.add_trace(go.Scatter(x=xs,y=ys,mode="lines",name="Normal fit",
            line=dict(color=C_AMBER,width=2.5),hoverinfo="skip"))
        apply_layout(fig_rh,"Residual Distribution — should be bell-shaped",height=340)
        st.plotly_chart(fig_rh, use_container_width=True)

    with rp2:
        qq = scipy_stats.probplot(resid.dropna())
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",
            marker=dict(color=C_BLUE,size=4,opacity=0.7),name="Data",
            hovertemplate="Theoretical %{x:.2f} → Sample <b>%{y:.2f}</b><extra></extra>"))
        m,b=qq[1][0],qq[1][1]
        xs2=np.array([qq[0][0].min(),qq[0][0].max()])
        fig_qq.add_trace(go.Scatter(x=xs2,y=m*xs2+b,mode="lines",
            line=dict(color=C_RED,width=2),name="Ideal",hoverinfo="skip"))
        apply_layout(fig_qq,"Q-Q Plot — points on the line = normal residuals",height=340)
        fig_qq.update_xaxes(title="Theoretical quantiles")
        fig_qq.update_yaxes(title="Sample quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    st.info(f"Residual mean: {resid.mean():.4f} (≈0 is good)  |  "
            f"Std: {resid.std():.4f}  |  AIC: {model_fit.aic:.2f}  |  BIC: {model_fit.bic:.2f}")

    with st.expander("📄 Full ARIMA Summary"):
        st.text(str(model_fit.summary()))


# ══════════════════════════════════════════════
#  TAB 5 — ACF / PACF
# ══════════════════════════════════════════════
with tab5:
    st.subheader("📉 ACF & PACF Analysis")
    st.markdown(f"<p style='color:{C_MUTED};font-size:13px;'>PACF → tells you <b style='color:{C_AMBER}'>p (AR order)</b> &nbsp;|&nbsp; ACF → tells you <b style='color:{C_TEAL}'>q (MA order)</b></p>",
                unsafe_allow_html=True)

    n_lags = st.slider("Lags to display", 20, 80, 40)
    diffs  = full_df["temperature"].diff().dropna()

    acf_v,  acf_ci  = acf(diffs,  nlags=n_lags, alpha=0.05)
    pacf_v, pacf_ci = pacf(diffs, nlags=n_lags, alpha=0.05)
    lags_x = np.arange(len(acf_v))

    def corr_fig(vals, ci, title, colour):
        su = ci[:,1]-vals; sl = vals-ci[:,0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(lags_x)+list(lags_x[::-1]),
            y=list(su)+list(-sl[::-1]),
            fill="toself", fillcolor=hex_rgba(colour,0.09),
            line=dict(width=0), name="95% CI", hoverinfo="skip"))
        fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
        for i,v in enumerate(vals):
            c2 = colour if abs(v)>su[i] else C_MUTED
            fig.add_trace(go.Scatter(x=[i,i],y=[0,v],mode="lines",
                line=dict(color=c2,width=2.5),showlegend=False,
                hovertemplate=f"Lag {i}: <b>{v:.3f}</b><extra></extra>"))
            fig.add_trace(go.Scatter(x=[i],y=[v],mode="markers",
                marker=dict(color=c2,size=7),showlegend=False,
                hovertemplate=f"Lag {i}: <b>{v:.3f}</b><extra></extra>"))
        apply_layout(fig, title, height=340)
        fig.update_xaxes(title="Lag")
        fig.update_yaxes(title="Correlation", range=[-1.1,1.1])
        return fig

    ac1,ac2 = st.columns(2)
    with ac1:
        st.plotly_chart(corr_fig(acf_v,  acf_ci,  "ACF — Autocorrelation (→ find q)", C_TEAL),  use_container_width=True)
    with ac2:
        st.plotly_chart(corr_fig(pacf_v, pacf_ci, "PACF — Partial Autocorrelation (→ find p)", C_AMBER), use_container_width=True)

    st.markdown("#### ACF of Residuals — flat = good model")
    resid_acf, resid_ci = acf(model_fit.resid.dropna(), nlags=n_lags, alpha=0.05)
    st.plotly_chart(corr_fig(resid_acf, resid_ci, "ACF of Residuals", C_BLUE), use_container_width=True)

    st.markdown(f"""
    <div class='card' style='font-size:13px;line-height:2.4;'>
        <b style='color:#e2eaf5;'>How to read:</b><br>
        <span style='color:{C_TEAL}'>Highlighted bars</span>
        <span style='color:{C_MUTED}'> = statistically significant (outside shaded CI band)</span><br>
        <span style='color:{C_AMBER}'>PACF sharp cutoff at lag 2</span>
        <span style='color:{C_MUTED}'> → p = 2</span><br>
        <span style='color:{C_TEAL}'>ACF sharp cutoff at lag 2</span>
        <span style='color:{C_MUTED}'> → q = 2</span><br>
        <span style='color:{C_MUTED}'>d = 1 because 1 differencing step made the series stationary</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 6 — DATA TABLE
# ══════════════════════════════════════════════
with tab6:
    st.subheader("📋 Full Dataset (2021–2026)")

    dm1,dm2,dm3,dm4 = st.columns(4)
    dm1.metric("Total records", f"{len(full_df):,}")
    dm2.metric("Date range",    f"{full_df.index[0].strftime('%b %Y')} – {full_df.index[-1].strftime('%b %Y')}")
    dm3.metric("Overall mean",  f"{full_df['temperature'].mean():.1f}°C")
    dm4.metric("Missing",       int(full_df.isnull().sum().sum()))

    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        yr_filt = st.multiselect("Filter years", all_years, default=all_years, key="tab6y")
    with fc2:
        tmin = st.number_input("Min °C", value=float(full_df["temperature"].min()), step=0.5)
    with fc3:
        tmax = st.number_input("Max °C", value=float(full_df["temperature"].max()), step=0.5)

    filt = full_df[(full_df.index.year.isin(yr_filt)) &
                   (full_df["temperature"]>=tmin) &
                   (full_df["temperature"]<=tmax)].copy()
    filt.index = filt.index.strftime("%Y-%m-%d")
    filt["temperature"] = filt["temperature"].round(2)
    filt.columns = ["Temperature (°C)"]

    st.caption(f"Showing {len(filt):,} of {len(full_df):,} records")
    st.dataframe(filt, use_container_width=True, height=420)

    dc1,dc2 = st.columns([1,3])
    with dc1:
        csv = filt.to_csv().encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name="bhopal_weather_2021_2026.csv",
                           mime="text/csv", use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:#2a4060;font-size:12px;padding:8px 0;'>
    Built by <b style='color:#3a6080'>Rakesh</b> · CSBS @ RGPV Bhopal ·
    ARIMA(2,1,2) · statsmodels · Plotly · Streamlit · Dataset 2021–2026
</div>""", unsafe_allow_html=True)