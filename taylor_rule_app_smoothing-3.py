
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Taylor Rule with Smoothing", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
REQUIRED_HIST_COLS = ["date", "fed_funds_actual", "inflation_used", "unemployment"]

def coerce_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=False)

def verify_and_map_columns(df, required_cols, title="Map columns"):
    """
    Tries to auto-map columns; if ambiguous, asks the user to map via selectboxes.
    Returns a dict: {required_name: actual_name}
    """
    candidates = {c.lower(): c for c in df.columns}
    auto = {}

    # common aliases to make upload easier
    aliases = {
        "date": ["date", "month", "period", "observation_date"],
        "fed_funds_actual": ["fed_funds_actual", "actual fed funds rate", "fed funds rate", "ffr", "effective federal funds rate", "effr"],
        "inflation_used": ["inflation_used", "headline inflation", "cpi", "core cpi", "core pce", "inflation", "pce inflation"],
        "unemployment": ["unemployment", "unemployment rate", "u3", "ur", "unemp"]
    }

    # attempt auto map
    for req in required_cols:
        found = None
        if req in candidates:
            found = candidates[req]
        else:
            # try aliases in order
            for alt in aliases.get(req, []):
                if alt in candidates:
                    found = candidates[alt]
                    break
        auto[req] = found

    with st.expander(title, expanded=False):
        st.write("Select the columns that correspond to each required field:")
        mapping = {}
        for req in required_cols:
            options = [None] + list(df.columns)
            default = auto.get(req, None)
            if default is not None and default in options:
                default_index = options.index(default)
            else:
                default_index = 0
            chosen = st.selectbox(f"{req} →", options, index=default_index, key=f"map_{req}")
            mapping[req] = chosen

    # final check
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        st.warning(f"Please finish mapping columns: missing {missing}")
        return None

    return mapping

def compute_taylor_with_smoothing(base_term: pd.Series, i_lag_init: float, rho: float) -> pd.Series:
    """
    Recursive smoothing: i_t = rho * i_{t-1} + (1 - rho) * base_t
    Uses provided initial lag value.
    """
    modeled = []
    i_prev = i_lag_init
    for val in base_term:
        i_t = rho * i_prev + (1.0 - rho) * float(val)
        modeled.append(i_t)
        i_prev = i_t
    return pd.Series(modeled, index=base_term.index, name="fed_funds_modeled")

def style_forecast_rows(df, forecast_flag_col="is_forecast"):
    if forecast_flag_col not in df.columns:
        return df
    def highlight(row):
        if bool(row.get(forecast_flag_col, False)):
            return ["background-color: #ffe5e5; color: #a40000"] * len(row)
        return [""] * len(row)
    return df.style.apply(highlight, axis=1)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Model Parameters")

# Choice of inflation target and neutral rate, etc.
r_star = st.sidebar.number_input("r* (neutral real rate, %)", value=0.5, step=0.1, format="%.2f")
pi_star = st.sidebar.number_input("π* (inflation target, %)", value=2.0, step=0.1, format="%.2f")
u_star = st.sidebar.number_input("u* (NAIRU, %)", value=4.0, step=0.1, format="%.2f")

# Sliders for a and b
a = st.sidebar.slider("a (inflation-gap coefficient)", min_value=-2.0, max_value=3.0, value=1.0, step=0.1)
b = st.sidebar.slider("b (unemployment-gap coef.)", min_value=-2.0, max_value=2.0, value=-0.5, step=0.1)

# Smoothing parameter (ρ) — per request, do NOT print/display it elsewhere.
rho = st.sidebar.slider("ρ (interest-rate smoothing)", min_value=0.0, max_value=0.99, value=0.8, step=0.01)

st.sidebar.header("Data")
st.sidebar.write("Upload your historical data (CSV) or use a demo dataset. Required columns:")
st.sidebar.code(", ".join(REQUIRED_HIST_COLS))

demo_btn = st.sidebar.button("Load demo dataset")

hist_file = st.sidebar.file_uploader("Historical data CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Optional: Upload forecasts")
st.sidebar.write("Provide a CSV with future **inflation** and **unemployment** forecasts. Required columns: `date`, plus either `inflation_used` or an inflation alias, and `unemployment`.")
forecast_file = st.sidebar.file_uploader("Forecast CSV", type=["csv"])

# -----------------------------
# Load data
# -----------------------------
if demo_btn and (hist_file is None):
    # simple demo
    demo_csv = io.StringIO()
    demo_csv.write("date,fed_funds_actual,inflation_used,unemployment\n")
    # a tiny illustrative demo series
    dates = pd.date_range("2018-01-01", periods=48, freq="MS")
    rng = np.random.default_rng(0)
    ffr = np.clip(1.5 + 0.5*np.sin(np.linspace(0, 5, len(dates))) + rng.normal(0, 0.15, len(dates)), 0.0, None)
    infl = np.clip(2.0 + 0.6*np.sin(np.linspace(0, 6, len(dates))) + rng.normal(0, 0.2, len(dates)), -1, None)
    unemp = np.clip(4.0 + 0.3*np.cos(np.linspace(0, 4, len(dates))) + rng.normal(0, 0.1, len(dates)), 2.5, None)
    for d, i1, i2, u in zip(dates, ffr, infl, unemp):
        demo_csv.write(f"{d.date()},{i1:.2f},{i2:.2f},{u:.2f}\n")
    demo_csv.seek(0)
    hist_df = pd.read_csv(demo_csv)
else:
    if hist_file is None:
        st.info("Upload a historical CSV to begin, or click 'Load demo dataset' in the sidebar.")
        st.stop()
    hist_df = pd.read_csv(hist_file)

# Column mapping for historical data
hist_map = verify_and_map_columns(hist_df, REQUIRED_HIST_COLS, title="Map historical data columns")
if hist_map is None:
    st.stop()

# Clean and prepare historical data
df = hist_df.rename(columns={hist_map[k]: k for k in hist_map})
df["date"] = coerce_datetime(df["date"])
df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

# Build base term for historical period
base_term_hist = r_star + pi_star + a*(df["inflation_used"] - pi_star) + b*(df["unemployment"] - u_star)

# Initialize with first available actual fed funds if present, otherwise use base term
i_init = df["fed_funds_actual"].iloc[0]
if pd.isna(i_init):
    i_init = base_term_hist.iloc[0]

# Compute modeled historical series
df["fed_funds_modeled"] = compute_taylor_with_smoothing(base_term_hist, i_init, rho)

# Compute MSE over overlap where both actual and modeled exist
valid = df[["fed_funds_actual", "fed_funds_modeled"]].dropna()
mse_value = float(((valid["fed_funds_actual"] - valid["fed_funds_modeled"])**2).mean()) if not valid.empty else np.nan

# Forecast handling
forecast_df = None
forecast_series = None

if forecast_file is not None:
    raw_fore = pd.read_csv(forecast_file)

    # Attempt auto mapping using a reduced required set for forecasts
    # We re-use the same mapping helper but only need date, inflation_used, unemployment
    req_fore_cols = ["date", "inflation_used", "unemployment"]
    fore_map = verify_and_map_columns(raw_fore, req_fore_cols, title="Map forecast data columns")
    if fore_map is not None:
        forecast_df = raw_fore.rename(columns={fore_map[k]: k for k in fore_map})
        forecast_df["date"] = coerce_datetime(forecast_df["date"])
        forecast_df = forecast_df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

        # Build base term for forecast period
        base_term_fore = r_star + pi_star + a*(forecast_df["inflation_used"] - pi_star) + b*(forecast_df["unemployment"] - u_star)

        # Start from the last modeled historical value to maintain continuity
        last_modeled = df["fed_funds_modeled"].iloc[-1]
        forecast_series = compute_taylor_with_smoothing(base_term_fore, last_modeled, rho)
        forecast_series.name = "fed_funds_forecast"

# -----------------------------
# Layout
# -----------------------------
st.title("Taylor Rule with Smoothing")
left, right = st.columns((2, 1))

with left:
    st.subheader("Policy Rate: Actual vs. Taylor Rule (Modeled)")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Plot historical actual & modeled
    ax.plot(df["date"], df["fed_funds_actual"], label="Fed Funds (Actual)")
    ax.plot(df["date"], df["fed_funds_modeled"], label="Taylor Rule (Modeled)")

    # Optional forecast
    if forecast_series is not None:
        ax.plot(forecast_df["date"], forecast_series.values, linestyle="--", label="Taylor Rule (Forecast)")

    # Zero lower bound dashed line
    ax.axhline(0.0, linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.grid(True, which="both", axis="y", alpha=0.3)
    st.pyplot(fig)

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("MSE (Actual vs Modeled)", f"{mse_value:.4f}" if not math.isnan(mse_value) else "—")
    m2.metric("Observations", f"{len(df):d}")
    m3.metric("Forecast points", f"{len(forecast_series):d}" if forecast_series is not None else "0")

with right:
    st.subheader("Data Preview")

    # Merge a preview, tagging forecast rows
    preview = df[["date", "fed_funds_actual", "fed_funds_modeled", "inflation_used", "unemployment"]].copy()
    preview["is_forecast"] = False

    if forecast_series is not None:
        fore_preview = forecast_df[["date", "inflation_used", "unemployment"]].copy()
        fore_preview["fed_funds_actual"] = np.nan
        fore_preview["fed_funds_modeled"] = forecast_series.values
        fore_preview["is_forecast"] = True
        preview = pd.concat([preview, fore_preview], ignore_index=True)
        preview = preview.sort_values("date").reset_index(drop=True)

    # Style forecast rows in red
    styled = style_forecast_rows(preview, "is_forecast")
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption("Tip: Upload your own data with columns named (or mapped to) "
           "`date`, `fed_funds_actual`, `inflation_used`, and `unemployment`. "
           "Optionally upload a forecast CSV with `date`, `inflation_used`, and `unemployment`.")
