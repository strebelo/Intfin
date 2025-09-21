# monetary policy rule
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monetary policy simulator", layout="wide")

st.title("Monetary policy simulator")
st.write(
    "Upload a CSV file with the Federal Funds Rate, three inflation series (Headline, Core CPI, Core PCE), and the unemployment rate. "
    "You can get these data from www.fred.com or upload the file provided"
)

st.latex(r"""
i_t = \rho \, i_{t-L} + (1 - \rho) \left[ r^{*} + \pi^{*} + a (\pi_t - \pi^{*}) + b (u_t - u^{*}) \right]
""")

# ---- Sidebar controls ----
st.sidebar.header("Model Parameters")
a = st.sidebar.slider("Inflation gap coefficient", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
b = st.sidebar.slider("Unemployment gap coefficient", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
r_star = st.sidebar.slider("Neutral real rate r* (%)", min_value=-1.0, max_value=3.0, value=0.0, step=0.1)
pi_star = st.sidebar.slider("Inflation target π* (%)", min_value=0.0, max_value=4.0, value=0.0, step=0.1)
u_star = st.sidebar.slider("Natural unemployment u* (%)", min_value=3.0, max_value=8.0, value=4.0, step=0.1)

rho = st.sidebar.slider("Interest rate smoothing ρ", min_value=0.0, max_value=0.99, value=0.0, step=0.01,
                        help="Weight on the previous period policy rate.")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Upload")
uploaded = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help=(
        "Required columns: date, fed_funds_actual, unemployment, plus at least one inflation series. "
        "Accepted inflation columns include: inflation, headline_inflation, cpi_headline, "
        "core_inflation/core_cpi, or core_pce."
    )
)

if uploaded is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ---- Load and normalize columns ----
df = pd.read_csv(uploaded)
df.columns = [c.strip().lower() for c in df.columns]

# Friendly renames
rename_map = {
    'fed_funds_rate': 'fed_funds_actual',
    'actual_fed_funds': 'fed_funds_actual',
    'federal_funds_rate': 'fed_funds_actual',
    'unemp': 'unemployment',
    'unemployment_rate': 'unemployment',
    'cpi': 'inflation',
    'headline_cpi': 'inflation',
    'cpi_headline': 'inflation',
    'headline_inflation': 'inflation',
    'pce': 'inflation',
    'corecpi': 'core_inflation',
    'core_cpi': 'core_inflation',
    'cpi_core': 'core_inflation',
    'core': 'core_inflation',
    'corepce': 'core_pce',
    'core_pce_inflation': 'core_pce',
    'core-pce': 'core_pce'
}
df = df.rename(columns=rename_map)

# Required basics
required_base = {'date', 'fed_funds_actual', 'unemployment'}
missing_base = required_base - set(df.columns)
if missing_base:
    st.error(f"CSV missing required columns: {sorted(missing_base)}")
    st.stop()

# Parse/sort date
try:
    df['date'] = pd.to_datetime(df['date'])
except Exception:
    st.error("Could not parse 'date' column. Ensure it is ISO-like (YYYY-MM-DD).")
    st.stop()
df = df.sort_values('date')

# Detect available inflation series
inflation_candidates = {
    "Headline CPI": ['inflation', 'headline', 'cpi', 'cpi_inflation'],
    "Core CPI": ['core_inflation', 'corecpi'],
    "Core PCE": ['core_pce']
}

available_options = []
option_to_col = {}
for label, cols in inflation_candidates.items():
    for c in cols:
        if c in df.columns:
            available_options.append(label)
            option_to_col[label] = c
            break

if not available_options:
    st.error(
        "No inflation series found. Include at least one of: "
        "`inflation` (headline), `core_inflation` (core CPI), or `core_pce` (core PCE)."
    )
    st.stop()

st.sidebar.subheader("Inflation Measure")
chosen_label = st.sidebar.selectbox("Choose inflation series", options=available_options, index=0)
infl_col = option_to_col[chosen_label]
df['inflation_used'] = pd.to_numeric(df[infl_col], errors='coerce')

# Ensure numeric
for col in ['fed_funds_actual', 'inflation_used', 'unemployment']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['fed_funds_actual', 'inflation_used', 'unemployment'])

if df.empty:
    st.error("After cleaning, the dataset is empty. Check for non-numeric values or column mismatches.")
    st.stop()

# ---- Taylor rule with smoothing ----
# i_t = rho * i_{t-L} + (1 - rho) * ( r* + pi* + a*(pi_t - pi*) + b*(u_t - u*) )
base_term = r_star + pi_star + a*(df['inflation_used'] - pi_star) + b*(df['unemployment'] - u_star)

L = 3  # lag length for smoothing: t-3

# 1) Base (unsmoothed) Taylor term each period
base_term = r_star + pi_star \
            + a * (df['inflation_used'] - pi_star) \
            + b * (df['unemployment'] - u_star)

n = len(df)
modeled = [None] * n

# 2) Seed the first L observations
for k in range(min(L, n)):
    v = df['fed_funds_actual'].iloc[k]
    modeled[k] = v if not pd.isna(v) else base_term.iloc[k]

# 3) Recursive smoothing using t-3
for t in range(L, n):
    # prefer actual at t-3 if available; else use modeled[t-3]
    i_lag_actual = df['fed_funds_actual'].iloc[t - L]
    i_lag = i_lag_actual if not pd.isna(i_lag_actual) else modeled[t - L]

    i_t = rho * i_lag + (1.0 - rho) * base_term.iloc[t]
    # Optional: zero lower bound
    # i_t = max(0.0, i_t)

    modeled[t] = i_t

df['fed_funds_modeled'] = modeled

# ---- Layout ----
col1, col2 = st.columns([2,1], vertical_alignment="top")

with col1:
    st.subheader("Actual vs. Modeled Policy Rate")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['date'], df['fed_funds_actual'], label='Actual Fed Funds')
    ax.plot(df['date'], df['fed_funds_modeled'], label='Modeled (Taylor Rule w/ smoothing)')
    ax.axhline(0.0, linestyle='--', linewidth=1, label='Zero lower bound')
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.legend(loc='best')
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(f"Inflation series in use: **{chosen_label}** (column: `{infl_col}`).")

with col2:
    st.subheader("Key Gaps (latest)")
    last = df.iloc[-1]
    st.metric("Latest inflation (π)", f"{last['inflation_used']:.2f}%", delta=f"{(last['inflation_used']-pi_star):+.2f} vs π*")
    st.metric("Latest unemployment (u)", f"{last['unemployment']:.2f}%", delta=f"{(last['unemployment']-u_star):+.2f} vs u*")
    st.metric("Smoothing ρ", f"{rho:.2f}")
    st.markdown("---")
    st.subheader("Download Results")
    out = df.copy()
    out['date'] = out['date'].dt.strftime("%Y-%m-%d")
    csv_bytes = out.to_csv(index=False).encode('utf-8')
    st.download_button("Download modeled CSV", data=csv_bytes, file_name="taylor_rule_modeled.csv", mime="text/csv")

st.markdown("---")
st.subheader("Data Preview")
st.dataframe(df[['date','fed_funds_actual','inflation_used','unemployment','fed_funds_modeled']], use_container_width=True)

st.markdown(r"""
**Model equation**
\[ i_t = \rho\, i_{t-1} + (1-\rho)\,\big(r^{*} + \pi^{*} + a(\pi_t-\pi^{*}) + b(u_t-u^{*})\big)\,. \]
""")
