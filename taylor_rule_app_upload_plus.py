
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Taylor Rule Simulator", layout="wide")

st.title("Taylor Rule Simulator")
st.write(
    "Upload a CSV and explore how the Taylor Rule behaves under different parameter settings "
    "and inflation definitions (Headline, Core, or Core PCE)."
)

# ---- Sidebar controls ----
st.sidebar.header("Model Parameters")
a = st.sidebar.slider("Inflation gap coefficient", min_value=-2.0, max_value=3.0, value=1.0, step=0.1)
b = st.sidebar.slider("Unemployment gap coefficient", min_value=-3.0, max_value=2.0, value=-0.5, step=0.1)
r_star = st.sidebar.slider("Neutral real rate r* (%)", min_value=-1.0, max_value=3.0, value=0.5, step=0.1)
pi_star = st.sidebar.slider("Inflation target π* (%)", min_value=0.0, max_value=4.0, value=2.0, step=0.1)
u_star = st.sidebar.slider("Natural unemployment u* (%)", min_value=3.0, max_value=8.0, value=4.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Upload")
uploaded = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help=(
        "Required columns: date, fed_funds_actual, unemployment, plus at least one inflation series. "
        "Accepted inflation columns include: inflation, headline_inflation, cpi_headline, "
        "core_inflation, core_cpi, core, corepce, core_pce."
    )
)

if uploaded is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ---- Load and normalize columns ----
df = pd.read_csv(uploaded)
df.columns = [c.strip().lower() for c in df.columns]

# Friendly renames for common variants
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
    'pce': 'inflation',  # if author meant headline PCE
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
except Exception as e:
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
        "No inflation series found. Please include at least one of the following columns "
        "(case-insensitive): "
        "`inflation`, `headline_inflation`, `cpi_headline`, `core_inflation`/`core_cpi`, or `core_pce`."
    )
    st.stop()

st.sidebar.subheader("Inflation Measure")
chosen_label = st.sidebar.selectbox("Choose inflation series", options=available_options, index=0)
infl_col = option_to_col[chosen_label]
df['inflation_used'] = df[infl_col]

# Ensure numeric
for col in ['fed_funds_actual', 'inflation_used', 'unemployment']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['fed_funds_actual', 'inflation_used', 'unemployment'])

if df.empty:
    st.error("After cleaning, the dataset is empty. Please check for non-numeric values or column mismatches.")
    st.stop()

# ---- Compute Taylor rule ----
# i_t = r* + π_t + a(π_t - π*) + b(u* - u_t)
df['inflation_gap'] = df['inflation_used'] - pi_star
df['unemployment_gap'] = u_star - df['unemployment']
df['fed_funds_modeled'] = r_star + df['inflation_used'] + a*df['inflation_gap'] + b*df['unemployment_gap']

# ---- Layout ----
col1, col2 = st.columns([2,1], vertical_alignment="top")

with col1:
    st.subheader("Actual vs. Modeled Policy Rate")

    # Matplotlib plot to add dashed zero lower bound
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['date'], df['fed_funds_actual'], label='Actual Fed Funds')
    ax.plot(df['date'], df['fed_funds_modeled'], label='Modeled (Taylor Rule)')
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
    st.metric("Latest unemployment (u)", f"{last['unemployment']:.2f}%", delta=f"{(u_star-last['unemployment']):+.2f} gap")
    st.markdown("---")
    st.subheader("Download Results")
    out = df.copy()
    out['date'] = out['date'].dt.strftime("%Y-%m-%d")
    csv_bytes = out.to_csv(index=False).encode('utf-8')
    st.download_button("Download modeled CSV", data=csv_bytes, file_name="taylor_rule_modeled.csv", mime="text/csv")

st.markdown("---")
st.subheader("Data Preview")
st.dataframe(df[['date','fed_funds_actual','inflation_used','unemployment','fed_funds_modeled']], use_container_width=True)

st.markdown("""
**Notes**
- Required columns: `date`, `fed_funds_actual`, `unemployment`, plus at least one inflation series.
- Accepted inflation columns include: `inflation` (Headline CPI), `core_inflation` (Core CPI), or `core_pce` (Core PCE).
- Taylor rule: \\( i_t = r^* + \\pi_t + a(\\pi_t - \\pi^*) + b(u^* - u_t) \\).
- Defaults: a=1.0, b=-0.5, r*=0.5, π*=2.0, u*=4.0.
""")
