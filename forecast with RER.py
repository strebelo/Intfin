# forecast with RER.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="Exchange Rate Forecasting", layout="wide")

st.title("Monthly Spot Exchange Rate Forecasts (Random Walk vs RER-Augmented Model)")

st.markdown(
    """
This app:
1. Reads an Excel file with **Date**, **Spot**, and **Real Exchange Rate (RER)** (in the first three columns).  
2. Estimates, recursively over time, a model of the form  
   \\( \Delta S_t = \phi \cdot \frac{\text{RER}_{t-1}}{\overline{\text{RER}}_{1:t-1}} + \varepsilon_t \\).  
3. Uses \\( \hat\phi_t \\) at each date to forecast the spot rate up to **5 years ahead (60 months)**.  
4. Compares out-of-sample **RMSFE** of this model against a simple **random walk** at multiple horizons.  
5. Plots the forecast paths from any chosen origin date.
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
    min_value=24,
    max_value=240,
    value=60,
    step=6,
    help="Number of initial observations used before starting recursive estimation."
)

default_horizons = [1, 3, 6, 12, 24, 36, 60]
horizon_str = st.sidebar.text_input(
    "Forecast horizons (months, comma-separated)",
    value="1,3,6,12,24,36,60",
    help="Horizons at which RMSFE is computed (max 60 months)."
)

use_logs = st.sidebar.checkbox(
    "Use log(spot) in place of levels", 
    value=False,
    help="If checked, modeling and forecasts are done on log spot rates."
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

MAX_H = 60  # 5 years

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
    st.error(f"Could not parse dates in the first column: {e}")
    st.stop()

df = df.sort_values("date").dropna(subset=["spot", "rer"])
df.set_index("date", inplace=True)

if df.empty or len(df) < min_window + MAX_H:
    st.warning(
        f"Not enough data. You have {len(df)} monthly observations. "
        f"You requested a minimum estimation window of {min_window} and up to "
        f"{MAX_H} months ahead for forecasts. Consider reducing the window size."
    )

# Optionally log-transform spot
if use_logs:
    df["spot_trans"] = np.log(df["spot"].astype(float))
else:
    df["spot_trans"] = df["spot"].astype(float)

df["rer"] = df["rer"].astype(float)

# Compute ratio RER_t / avg_RER_1:t
df["avg_rer"] = df["rer"].expanding().mean()
df["ratio"] = df["rer"] / df["avg_rer"]

# Compute ΔS_t
df["dS"] = df["spot_trans"].diff()

n = len(df)
dates = df.index

st.subheader("Data Overview")
st.write(f"Number of monthly observations: **{n}**")
st.dataframe(df[["spot", "rer"]].tail())

st.line_chart(df[["spot", "rer"]])

# --- Recursive estimation and forecast errors ---
# Storage
phi_hat = pd.Series(index=dates, dtype=float)
errors_model = {h: [] for h in horizons}
errors_rw = {h: [] for h in horizons}

origins_used = []

# Main recursive loop
for origin_idx in range(min_window, n - 1):  # origin at index origin_idx
    # Regression sample: up to origin_idx (inclusive)
    # y_t = ΔS_t for t = 1..origin_idx
    # x_t = ratio_{t-1} for t = 1..origin_idx  (lagged ratio)
    y = df["dS"].iloc[1:origin_idx + 1]       # positions 1..origin_idx
    x = df["ratio"].iloc[0:origin_idx]        # positions 0..origin_idx-1

    # Align indices
    y, x = y.align(x, join="inner")

    if len(y) < min_window:
        continue

    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    phi = model.params[1]  # coefficient on ratio
    phi_hat.iloc[origin_idx] = phi

    S_origin = df["spot_trans"].iloc[origin_idx]
    ratio_origin = df["ratio"].iloc[origin_idx]
    origins_used.append(origin_idx)

    # Forecast errors for each horizon
    for h in horizons:
        if origin_idx + h >= n:
            continue
        S_actual = df["spot_trans"].iloc[origin_idx + h]

        # RER-augmented model: S_t+h|t = S_t + h * phi_hat_t * ratio_t
        S_hat_model = S_origin + h * phi * ratio_origin

        # Random walk: S_t+h|t = S_t
        S_hat_rw = S_origin

        errors_model[h].append(S_actual - S_hat_model)
        errors_rw[h].append(S_actual - S_hat_rw)

origins_used = np.array(origins_used)
if len(origins_used) == 0:
    st.error("Could not perform recursive estimation. Check the window size or data length.")
    st.stop()

# --- RMSFE comparison ---
results = []
for h in horizons:
    em = np.array(errors_model[h], dtype=float)
    er = np.array(errors_rw[h], dtype=float)
    if len(em) == 0 or len(er) == 0:
        continue
    rmsfe_model = np.sqrt(np.mean(em**2))
    rmsfe_rw = np.sqrt(np.mean(er**2))
    rel = (rmsfe_model / rmsfe_rw) if rmsfe_rw > 0 else np.nan
    results.append(
        {
            "Horizon (months)": h,
            "RMSFE - RER model": rmsfe_model,
            "RMSFE - Random walk": rmsfe_rw,
            "Ratio (RER / RW)": rel,
            "Improvement (%)": (1 - rel) * 100 if np.isfinite(rel) else np.nan,
        }
    )

st.subheader("Out-of-Sample Forecast Performance")

if results:
    res_df = pd.DataFrame(results).set_index("Horizon (months)")
    st.dataframe(res_df.style.format({
        "RMSFE - RER model": "{:.6f}",
        "RMSFE - Random walk": "{:.6f}",
        "Ratio (RER / RW)": "{:.3f}",
        "Improvement (%)": "{:+.1f}"
    }))
else:
    st.warning("No valid forecast errors were computed for the chosen horizons.")

# --- Forecast paths from a selected origin date ---
st.subheader("Forecast Paths from Selected Origin Date")

# Only allow origins where phi_hat is available
valid_phi = phi_hat.dropna()
if valid_phi.empty:
    st.warning("No valid φ estimates to plot forecast paths.")
    st.stop()

origin_date_default = valid_phi.index[-MAX_H] if len(valid_phi) > MAX_H else valid_phi.index[-1]

origin_date = st.selectbox(
    "Choose forecast origin date",
    options=list(valid_phi.index),
    index=list(valid_phi.index).index(origin_date_default),
    format_func=lambda d: d.strftime("%Y-%m")
)

origin_idx = df.index.get_loc(origin_date)
phi0 = phi_hat.iloc[origin_idx]
ratio0 = df["ratio"].iloc[origin_idx]
S0 = df["spot_trans"].iloc[origin_idx]

freq = pd.infer_freq(df.index)
if freq is None:
    freq = "MS"

forecast_index = pd.date_range(start=origin_date, periods=MAX_H + 1, freq=freq)

# Build forecast paths
model_forecast = []
rw_forecast = []
for h in range(0, MAX_H + 1):
    model_forecast.append(S0 + h * phi0 * ratio0)
    rw_forecast.append(S0)

fc_df = pd.DataFrame(
    {
        "Model (with RER)": model_forecast,
        "Random walk": rw_forecast,
    },
    index=forecast_index,
)

# Bring back to levels if using logs
if use_logs:
    fc_df = np.exp(fc_df)
    spot_plot = df["spot"]
else:
    spot_plot = df["spot_trans"]

# Merge actuals and forecasts for plotting
combined = pd.DataFrame(index=spot_plot.index.union(fc_df.index))
combined["Actual spot"] = spot_plot
combined["Model (with RER)"] = fc_df["Model (with RER)"]
combined["Random walk"] = fc_df["Random walk"]

# Restrict to a window around the origin (e.g. 5 years before and 5 years after)
plot_start = max(combined.index[0], origin_date - pd.DateOffset(years=5))
plot_end = min(combined.index[-1], origin_date + pd.DateOffset(years=5))
combined_window = combined.loc[plot_start:plot_end]

st.line_chart(combined_window)

st.markdown(
    """
**Interpretation tips:**
- The **table** shows whether adding the RER term improves out-of-sample RMSFE relative to a pure random walk at each horizon
  (negative “Improvement (%)” means the RER model is worse).
- The **plot** shows, for the chosen origin date:
  - The realized spot rate.
  - The 5-year forecast path from the random walk.
  - The 5-year forecast path from the RER-augmented model.
"""
)
