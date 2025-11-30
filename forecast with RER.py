# forecast with RER.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="FX Forecasts with RER & CPI", layout="wide")

st.title("Using RER Mean Reversion and CPI Ratio to Forecast the Nominal Exchange Rate")

st.markdown(
    r"""
Upload an Excel file with **five columns** (in this order):

1. Date (monthly)  
2. Spot nominal exchange rate  
3. Real exchange rate (RER)  
4. Domestic CPI  
5. Foreign CPI  

The app will:

- Work in logs: \( s_t = \log(\text{spot}_t) \), \( q_t = \log(\text{RER}_t) \),
  \( c_t = \log(\text{CPI}^{\text{dom}}_t / \text{CPI}^{\text{for}}_t) \).
- At each forecast origin \(t\), estimate:
  - AR(1) with constant for \(q_t\) on the full history up to \(t\),
  - AR(1) with constant for \(c_t\) on the last 5 years (60 months) up to \(t\).
- Build multi-step forecasts \(\hat q_{t+h|t}\), \(\hat c_{t+h|t}\), and
  \(\hat s^{\text{struct}}_{t+h|t} = \hat q_{t+h|t} + \hat c_{t+h|t}\).
- Compare out-of-sample RMSFE with a **random walk** benchmark:
  \( \hat s^{RW}_{t+h|t} = s_t \).
"""
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file (5 columns: Date, Spot, RER, CPI_dom, CPI_for)",
    type=["xlsx", "xls"]
)

MIN_WINDOW_MONTHS = st.sidebar.number_input(
    "Minimum estimation window (months)",
    min_value=36,
    max_value=240,
    value=120,  # e.g. 10 years
    step=12,
    help="Sample length before we start out-of-sample forecasting."
)

CPI_WINDOW_MONTHS = st.sidebar.number_input(
    "CPI AR(1) window (months)",
    min_value=24,
    max_value=240,
    value=60,   # 5 years as you suggested
    step=12,
    help="Number of last months used to estimate AR(1) for the CPI ratio."
)

MAX_H = 60  # 5 years in months

horizon_str = st.sidebar.text_input(
    "Forecast horizons (months, comma-separated)",
    value="1,3,6,12,24,36,60",
    help="Horizons for RMSFE comparison, up to 60."
)

def parse_horizons(hstr):
    try:
        hs = sorted(set(int(h.strip()) for h in hstr.split(",") if h.strip() != ""))
        hs = [h for h in hs if 1 <= h <= 60]
        return hs
    except Exception:
        return [1, 3, 6, 12, 24, 36, 60]

HORIZONS = parse_horizons(horizon_str)
if not HORIZONS:
    HORIZONS = [1, 3, 6, 12, 24, 36, 60]

# ---------------------------
# Helper functions
# ---------------------------

def ar1_forecast_multi_step(y_hist, h):
    """
    Estimate AR(1) with constant: y_t = alpha + rho y_{t-1} + eps,
    using y_hist (1D array). Then generate an h-step-ahead forecast
    starting from the last value y_hist[-1], using only y_hist.
    """
    y_hist = np.asarray(y_hist, dtype=float)
    if len(y_hist) < 3:
        # Too little data: fallback to random walk
        return y_hist[-1], 0.0, 1.0

    y = y_hist[1:]       # y_1..y_{T-1}
    x_lag = y_hist[:-1]  # y_0..y_{T-2}
    X = sm.add_constant(x_lag)
    model = sm.OLS(y, X).fit()
    alpha_hat, rho_hat = model.params

    y_fore = y_hist[-1]
    for _ in range(h):
        y_fore = alpha_hat + rho_hat * y_fore

    return y_fore, alpha_hat, rho_hat


def compute_structural_forecast(df_log, origin_idx, h, cpi_window_months=60):
    """
    Structural forecast:
        s_{t+h|t}^{struct} = q_{t+h|t} + c_{t+h|t}

    - q_t (log RER): AR(1) with constant, full history up to origin_idx.
    - c_t (log CPI_dom / CPI_for): AR(1) with constant, last cpi_window_months up to origin_idx.
    """
    # RER AR(1) on full history
    q_hist = df_log["q"].iloc[:origin_idx + 1].values
    q_hat_h, _, _ = ar1_forecast_multi_step(q_hist, h)

    # CPI ratio AR(1) on last cpi_window_months
    c_hist_full = df_log["c"].iloc[:origin_idx + 1].values
    if len(c_hist_full) > cpi_window_months:
        c_hist = c_hist_full[-cpi_window_months:]
    else:
        c_hist = c_hist_full
    c_hat_h, _, _ = ar1_forecast_multi_step(c_hist, h)

    return q_hat_h + c_hat_h


# ---------------------------
# Main logic
# ---------------------------
if uploaded_file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# Load Excel from upload (no FileNotFound now)
try:
    df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

if df_raw.shape[1] < 5:
    st.error("Excel must have at least 5 columns: Date, Spot, RER, CPI_dom, CPI_for.")
    st.stop()

# Take first 5 columns and rename
df = df_raw.iloc[:, :5].copy()
df.columns = ["date", "spot", "rer", "cpi_dom", "cpi_for"]

# Parse dates and clean
try:
    df["date"] = pd.to_datetime(df["date"])
except Exception as e:
    st.error(f"Could not parse the Date column as dates: {e}")
    st.stop()

df.sort_values("date", inplace=True)
df.set_index("date", inplace=True)
df = df.dropna()

# Remove non-positive values that break logs
df = df[(df["spot"] > 0) & (df["rer"] > 0) &
        (df["cpi_dom"] > 0) & (df["cpi_for"] > 0)]

n = len(df)
if n < MIN_WINDOW_MONTHS + MAX_H + 1:
    st.warning(
        f"After cleaning, you have {n} observations. "
        f"Forecast evaluation may be limited with the chosen windows and horizons."
    )

st.subheader("Data overview")
st.write(f"Number of monthly observations after cleaning: **{n}**")
st.dataframe(df.tail())

# Separate plots for spot and RER
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Spot exchange rate (level)**")
    st.line_chart(df["spot"])
with col2:
    st.markdown("**Real exchange rate (RER)**")
    st.line_chart(df["rer"])

# ---------------------------
# Build log variables
# ---------------------------
df_log = pd.DataFrame(index=df.index)
df_log["s"] = np.log(df["spot"].astype(float))
df_log["q"] = np.log(df["rer"].astype(float))
df_log["c"] = np.log(df["cpi_dom"].astype(float) / df["cpi_for"].astype(float))

# ---------------------------
# Out-of-sample evaluation
# ---------------------------
errors_struct = {h: [] for h in HORIZONS}
errors_rw = {h: [] for h in HORIZONS}
origins = []

for origin_idx in range(MIN_WINDOW_MONTHS, n - 1):
    s_t = df_log["s"].iloc[origin_idx]
    origins.append(origin_idx)

    for h in HORIZONS:
        if origin_idx + h >= n:
            continue

        s_actual = df_log["s"].iloc[origin_idx + h]

        # Structural forecast (uses only data up to origin_idx)
        s_hat_struct = compute_structural_forecast(
            df_log, origin_idx, h, cpi_window_months=CPI_WINDOW_MONTHS
        )

        # Random walk forecast
        s_hat_rw = s_t

        errors_struct[h].append(s_actual - s_hat_struct)
        errors_rw[h].append(s_actual - s_hat_rw)

if not origins:
    st.error(
        "Not enough effective sample to run out-of-sample evaluation with the "
        "current MIN_WINDOW_MONTHS / horizons."
    )
    st.stop()

# ---------------------------
# RMSFE comparison
# ---------------------------
st.subheader("Out-of-sample RMSFE (logs of spot)")

results = []
for h in HORIZONS:
    es = np.array(errors_struct[h], dtype=float)
    er = np.array(errors_rw[h], dtype=float)
    if len(es) == 0 or len(er) == 0:
        continue

    rmsfe_struct = np.sqrt(np.mean(es ** 2))
    rmsfe_rw = np.sqrt(np.mean(er ** 2))
    rel = rmsfe_struct / rmsfe_rw if rmsfe_rw > 0 else np.nan
    improvement = (1 - rel) * 100 if np.isfinite(rel) else np.nan

    results.append(
        {
            "Horizon (months)": h,
            "Obs used": len(es),
            "RMSFE - Structural": rmsfe_struct,
            "RMSFE - Random walk": rmsfe_rw,
            "Ratio (Struct/RW)": rel,
            "Improvement (%)": improvement,
        }
    )

if results:
    res_df = pd.DataFrame(results).set_index("Horizon (months)")
    st.dataframe(
        res_df.style.format({
            "RMSFE - Structural": "{:.6f}",
            "RMSFE - Random walk": "{:.6f}",
            "Ratio (Struct/RW)": "{:.3f}",
            "Improvement (%)": "{:+.1f}",
        })
    )
else:
    st.warning("No valid forecast errors were computed for the selected horizons.")

# ---------------------------
# Forecast path from a chosen origin date
# ---------------------------
st.subheader("Example 5-year forecast path")

# Choose origins that allow a full 5-year forecast
valid_origin_indices = [
    idx for idx in range(MIN_WINDOW_MONTHS, n - MAX_H)
]
if not valid_origin_indices:
    st.warning("Not enough room at the end of the sample for a full 5-year forecast path.")
    st.stop()

default_origin_idx = valid_origin_indices[-1]
origin_date_default = df_log.index[default_origin_idx]

origin_date = st.selectbox(
    "Choose forecast origin date",
    options=[df_log.index[i] for i in valid_origin_indices],
    index=valid_origin_indices.index(default_origin_idx),
    format_func=lambda d: d.strftime("%Y-%m")
)

origin_idx = df_log.index.get_loc(origin_date)
s0 = df_log["s"].iloc[origin_idx]

freq = pd.infer_freq(df_log.index)
if freq is None:
    freq = "MS"

forecast_index = pd.date_range(start=origin_date, periods=MAX_H + 1, freq=freq)

struct_path = []
rw_path = []
for h in range(0, MAX_H + 1):
    if h == 0:
        struct_path.append(s0)
        rw_path.append(s0)
    else:
        s_hat_struct = compute_structural_forecast(
            df_log, origin_idx, h, cpi_window_months=CPI_WINDOW_MONTHS
        )
        struct_path.append(s_hat_struct)
        rw_path.append(s0)

struct_levels = np.exp(struct_path)
rw_levels = np.exp(rw_path)

fc_df = pd.DataFrame(
    {
        "Structured model": struct_levels,
        "Random walk": rw_levels,
    },
    index=forecast_index,
)

combined = pd.DataFrame(index=df.index.union(fc_df.index))
combined["Actual spot"] = df["spot"]
combined["Structured model"] = fc_df["Structured model"]
combined["Random walk"] = fc_df["Random walk"]

plot_start = max(combined.index[0], origin_date - pd.DateOffset(years=5))
plot_end = min(combined.index[-1], origin_date + pd.DateOffset(years=5))
combined_window = combined.loc[plot_start:plot_end]

st.line_chart(combined_window)
