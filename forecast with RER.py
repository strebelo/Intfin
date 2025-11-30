# forecast with RER.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="Structured FX Forecasting (RER Mean Reversion)", layout="wide")

st.title("Nominal Exchange Rate Forecasts Exploiting RER Mean Reversion")

st.markdown(
    r"""
This app implements a **structural-style forecasting approach**:

1. It reads an Excel file with **Date**, **Spot**, and **Real Exchange Rate (RER)** in the first three columns.  
2. It defines (in logs) the nominal rate \( s_t \), the real exchange rate \( q_t \), and the implied price-differential \( d_t = s_t - q_t \).  
3. It estimates, **recursively and out-of-sample**, for each forecast origin \( t \):
   - A **mean-reverting AR(1)** for the RER,
   - A **random walk with drift** for \( d_t \),
   - Then constructs \( h \)-step forecasts of the nominal rate using  
     \[
     \hat s_{t+h|t}^{\text{struct}} = \mathbb{E}_t[q_{t+h}] + \mathbb{E}_t[d_{t+h}].
     \]
4. It compares out-of-sample **RMSFE** of this structured model with a **random walk** benchmark:
   \[
   \hat s_{t+h|t}^{\text{RW}} = s_t.
   \]
5. You can visualize forecast paths from any chosen origin date.

All estimation at a given origin uses **only data up to that origin** (no look-ahead bias).
"""
)

# --- Sidebar controls ---
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file (Date in col 1, Spot in col 2, RER in col 3)",
    type=["xlsx", "xls"]
)

min_window = st.sidebar.number_input(
    "Minimum estimation window (months)",
    min_value=36,
    max_value=240,
    value=60,
    step=6,
    help="Number of initial observations used before starting recursive out-of-sample forecasts."
)

default_horizons = [1, 3, 6, 12, 24, 36, 60]
horizon_str = st.sidebar.text_input(
    "Forecast horizons (months, comma-separated)",
    value="1,3,6,12,24,36,60",
    help="Horizons at which RMSFE is computed (max 60 months)."
)

MAX_H = 60  # 5 years

use_log_spot = st.sidebar.checkbox(
    "Use log(spot) (recommended)",
    value=True,
    help="If checked, models and forecasts are built on log spot rates."
)

use_log_rer = st.sidebar.checkbox(
    "Use log(RER) if positive",
    value=True,
    help="If checked and RER>0, mean reversion is estimated on log(RER)."
)

# Parse horizons
def parse_horizons(hstr):
    try:
        hs = sorted(set(int(h.strip()) for h in hstr.split(",") if h.strip() != ""))
        hs = [h for h in hs if 1 <= h <= 60]
        return hs
    except Exception:
        return default_horizons

horizons = parse_horizons(horizon_str)
if not horizons:
    horizons = default_horizons

if uploaded_file is None:
    st.info("Upload an Excel file in the sidebar to begin.")
    st.stop()

# --- Load and prepare data ---
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

if df.shape[1] < 3:
    st.error("The Excel file must have at least three columns: Date, Spot, and RER.")
    st.stop()

df = df.iloc[:, :3].copy()
df.columns = ["date", "spot", "rer"]

# Parse dates and sort
try:
    df["date"] = pd.to_datetime(df["date"])
except Exception as e:
    st.error(f"Could not parse dates in the first column as dates: {e}")
    st.stop()

df = df.sort_values("date").dropna(subset=["spot", "rer"])
df.set_index("date", inplace=True)

if len(df) < min_window + MAX_H:
    st.warning(
        f"You have {len(df)} observations. With a minimum window of {min_window} "
        f"and a max horizon of {MAX_H} months, out-of-sample evaluation may be limited."
    )

# --- Transformations: logs and series definitions ---
df["spot"] = df["spot"].astype(float)
df["rer"] = df["rer"].astype(float)

# Log spot
if use_log_spot:
    df["s"] = np.log(df["spot"])
else:
    df["s"] = df["spot"]

# RER for q_t
if use_log_rer:
    if (df["rer"] <= 0).any():
        st.warning("RER has non-positive values; using RER levels (no log) for mean-reversion.")
        df["q"] = df["rer"]
    else:
        df["q"] = np.log(df["rer"])
else:
    df["q"] = df["rer"]

# Price differential proxy: d_t = s_t - q_t
df["d"] = df["s"] - df["q"]

n = len(df)
dates = df.index

st.subheader("Data Overview")
st.write(f"Number of monthly observations: **{n}**")
st.dataframe(df[["spot", "rer"]].tail())

# Separate plots for spot and RER
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Spot exchange rate (levels)**")
    st.line_chart(df["spot"])
with col2:
    st.markdown("**Real exchange rate (RER)**")
    st.line_chart(df["rer"])

# --- Helper: RER AR(1) forecast and d_t drift forecast ---
def forecast_structural(origin_idx, h):
    """
    Build h-step-ahead structural forecast for s at origin index origin_idx,
    using only data up to origin_idx (inclusive).
    Returns float s_hat_struct.
    """
    # Data up to origin
    q_hist = df["q"].iloc[:origin_idx + 1].values  # indices 0..origin_idx
    d_hist = df["d"].iloc[:origin_idx + 1].values

    # Need at least 2 points to estimate AR(1) and drift; checked earlier via min_window
    # --- RER AR(1): q_t = alpha + rho q_{t-1} + u_t ---
    y_q = q_hist[1:]          # q_1..q_origin
    x_q = q_hist[:-1]         # q_0..q_{origin-1}
    X_q = sm.add_constant(x_q)
    ar_model = sm.OLS(y_q, X_q).fit()
    alpha_hat, rho_hat = ar_model.params

    # Long-run mean of q_t
    if abs(1 - rho_hat) > 1e-4 and abs(rho_hat) < 1.5:
        mu_hat = alpha_hat / (1 - rho_hat)
    else:
        # Fallback: sample mean
        mu_hat = np.mean(q_hist)

    q_t = q_hist[-1]
    # h-step forecast for q
    q_fore = mu_hat + (rho_hat ** h) * (q_t - mu_hat)

    # --- d_t random walk with drift: Δd_t = kappa + e_t ---
    d_t = d_hist[-1]
    d_diff = np.diff(d_hist)  # Δd_1 .. Δd_origin
    if len(d_diff) > 0:
        kappa_hat = np.mean(d_diff)
    else:
        kappa_hat = 0.0

    d_fore = d_t + h * kappa_hat

    # s_fore = q_fore + d_fore
    return q_fore + d_fore

# --- Out-of-sample evaluation ---
errors_struct = {h: [] for h in horizons}
errors_rw = {h: [] for h in horizons}
origins_used = []

for origin_idx in range(min_window, n - 1):  # leave at least 1 obs after origin
    origins_used.append(origin_idx)
    s_origin = df["s"].iloc[origin_idx]

    for h in horizons:
        if origin_idx + h >= n:
            continue

        s_actual = df["s"].iloc[origin_idx + h]

        # Structural forecast: uses data up to origin_idx only (inside function)
        s_hat_struct = forecast_structural(origin_idx, h)

        # Random walk forecast
        s_hat_rw = s_origin

        errors_struct[h].append(s_actual - s_hat_struct)
        errors_rw[h].append(s_actual - s_hat_rw)

origins_used = np.array(origins_used)
if len(origins_used) == 0:
    st.error("Could not perform recursive out-of-sample evaluation. Check the window size or data length.")
    st.stop()

# --- RMSFE comparison ---
st.subheader("Out-of-Sample Forecast Performance (Structured vs Random Walk)")

results = []
for h in horizons:
    es = np.array(errors_struct[h], dtype=float)
    er = np.array(errors_rw[h], dtype=float)
    if len(es) == 0 or len(er) == 0:
        continue
    rmsfe_struct = np.sqrt(np.mean(es**2))
    rmsfe_rw = np.sqrt(np.mean(er**2))
    rel = (rmsfe_struct / rmsfe_rw) if rmsfe_rw > 0 else np.nan
    results.append(
        {
            "Horizon (months)": h,
            "RMSFE - Structured model": rmsfe_struct,
            "RMSFE - Random walk": rmsfe_rw,
            "Ratio (Struct / RW)": rel,
            "Improvement (%)": (1 - rel) * 100 if np.isfinite(rel) else np.nan,
        }
    )

if results:
    res_df = pd.DataFrame(results).set_index("Horizon (months)")
    st.dataframe(
        res_df.style.format({
            "RMSFE - Structured model": "{:.6f}",
            "RMSFE - Random walk": "{:.6f}",
            "Ratio (Struct / RW)": "{:.3f}",
            "Improvement (%)": "{:+.1f}"
        })
    )
else:
    st.warning("No valid forecast errors were computed for the chosen horizons.")

# --- Forecast paths from a selected origin date ---
st.subheader("Forecast Paths from Selected Origin Date")

# Choose a reasonable default origin: max(last-60, min_window)
if n > (min_window + MAX_H):
    default_idx = n - MAX_H - 1
    if default_idx < min_window:
        default_idx = min_window
else:
    default_idx = min_window

origin_date_default = dates[default_idx]

origin_date = st.selectbox(
    "Choose forecast origin date",
    options=list(dates[min_window:-1]),
    index=list(dates[min_window:-1]).index(origin_date_default),
    format_func=lambda d: d.strftime("%Y-%m")
)

origin_idx = df.index.get_loc(origin_date)
s0 = df["s"].iloc[origin_idx]

freq = pd.infer_freq(df.index)
if freq is None:
    freq = "MS"

forecast_index = pd.date_range(start=origin_date, periods=MAX_H + 1, freq=freq)

# Build forecast paths
struct_forecast = []
rw_forecast = []

for h in range(0, MAX_H + 1):
    if h == 0:
        struct_forecast.append(s0)
        rw_forecast.append(s0)
    else:
        struct_forecast.append(forecast_structural(origin_idx, h))
        rw_forecast.append(s0)

fc_df = pd.DataFrame(
    {
        "Structured model": struct_forecast,
        "Random walk": rw_forecast,
    },
    index=forecast_index,
)

# Convert forecasts to levels for plotting
if use_log_spot:
    fc_plot = np.exp(fc_df)
    spot_plot = df["spot"]
else:
    fc_plot = fc_df
    spot_plot = df["s"]

# Merge actuals and forecasts
combined = pd.DataFrame(index=spot_plot.index.union(fc_plot.index))
combined["Actual spot"] = spot_plot
combined["Structured model"] = fc_plot["Structured model"]
combined["Random walk"] = fc_plot["Random walk"]

# Plot a 5-year window around the origin
plot_start = max(combined.index[0], origin_date - pd.DateOffset(years=5))
plot_end = min(combined.index[-1], origin_date + pd.DateOffset(years=5))
combined_window = combined.loc[plot_start:plot_end]

st.line_chart(combined_window)

st.markdown(
    r"""
**Interpretation:**

- The **structured model** exploits the **mean reversion of the RER** plus a **drifting price differential**:
  - When the RER is very depreciated relative to its long-run mean, the AR(1) pulls its forecast back, generating an expected **appreciation** of the nominal rate.
  - The drift in \( d_t \) captures **systematic inflation differentials** (PPP drift).
- The **random walk** ignores these forces and simply keeps \( s_t \) flat in expectation.

If the real exchange rate truly is mean-reverting and inflation differentials are persistent, you should see **gains at medium-to-long horizons** (e.g., 2–5 years) in the RMSFE table.
"""
)
