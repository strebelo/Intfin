
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Taylor Rule Simulator", layout="wide")

st.title("Taylor Rule Simulator")
st.write(
    "Explore how the Taylor Rule works. Adjust the parameters with the sliders "
    "and see how the modeled policy rate compares to the actual Federal Funds rate."
)

# --- Sidebar controls ---
st.sidebar.header("Model Parameters")
a = st.sidebar.slider("a: inflation gap coefficient", min_value=-2.0, max_value=3.0, value=1.0, step=0.1)
b = st.sidebar.slider("b: unemployment gap coefficient", min_value=-3.0, max_value=2.0, value=-0.5, step=0.1)
r_star = st.sidebar.slider("r*: neutral real rate (%)", min_value=-1.0, max_value=3.0, value=0.5, step=0.1)
pi_star = st.sidebar.slider("π*: inflation target (%)", min_value=0.0, max_value=4.0, value=2.0, step=0.1)
u_star = st.sidebar.slider("u*: natural unemployment rate (%)", min_value=3.0, max_value=8.0, value=4.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Upload")
uploaded = st.sidebar.file_uploader(
    "Upload CSV file", type=["csv"],
    help="Must include columns: date, fed_funds_actual, inflation_used, unemployment"
)

if uploaded is None:
    st.warning("Please upload a CSV file to proceed. Required columns: date, fed_funds_actual, inflation_used, unemployment.")
    st.stop()

# Load and check data
df = pd.read_csv(uploaded)
df.columns = [c.strip().lower() for c in df.columns]
rename_map = {
    'fed_funds_rate': 'fed_funds_actual',
    'actual_fed_funds': 'fed_funds_actual',
    'inflation': 'inflation_used',
    'unemp': 'unemployment',
    'unemployment_rate': 'unemployment'
}
df = df.rename(columns=rename_map)
required_cols = {'date', 'fed_funds_actual', 'inflation_used', 'unemployment'}
if not required_cols.issubset(df.columns):
    st.error(f"CSV missing required columns: {sorted(required_cols - set(df.columns))}")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Compute Taylor Rule
df['inflation_gap'] = df['inflation_used'] - pi_star
df['unemployment_gap'] = u_star - df['unemployment']
df['fed_funds_modeled'] = r_star + df['inflation_used'] + a*df['inflation_gap'] + b*df['unemployment_gap']

# Layout
col1, col2 = st.columns([2,1], vertical_alignment="top")

with col1:
    st.subheader("Actual vs. Modeled Policy Rate")
    chart_df = df.set_index('date')[['fed_funds_actual','fed_funds_modeled']]
    st.line_chart(chart_df, height=380)

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
st.dataframe(df, use_container_width=True)

st.markdown("""
**Notes**
- Required columns (case-insensitive): `date`, `fed_funds_actual`, `inflation_used`, `unemployment`.
- The Taylor rule here is:  
  \\( i_t = r^* + \\pi_t + a(\\pi_t - \\pi^*) + b(u^* - u_t) \\).
""")
