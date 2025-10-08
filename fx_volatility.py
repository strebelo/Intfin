
# fx_volatility.py
# Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails
# Author: (your name here)
# Usage: `streamlit run streamlit_fx_exchange_app.py`
#
# Features
# - Upload monthly spot exchange rates (CSV or Excel)
# - Compute annual log changes: log(S_t) - log(S_{t-12})
# - Summary stats: mean, std, skew, kurtosis
# - Histogram with Normal and KDE overlays
# - Normality tests: Jarque–Bera, Shapiro–Wilk, KS
# - QQ plot vs Normal
# - Nonparametric PDF (KDE) vs Normal
# - Tail probabilities beyond μ ± 1.96σ:
#     * Normal model
#     * KDE model (numerical integration on a fine grid)
#     * Empirical proportion
#
# Notes
# - Ensure your file has at least a Date column (monthly) and a SpotRate column.
# - Dates should parse to a monotone time index; the app sorts by date safely.
#
# ---------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats

st.set_page_config(page_title="FX Annual Log Changes — Normal vs Fat Tails", layout="wide")

# -------------------------------
# Helpers
# -------------------------------

def parse_input_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        # Try CSV as fallback
        df = pd.read_csv(uploaded_file)
    return df

def coerce_datetime(series: pd.Series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def compute_annual_log_changes(df: pd.DataFrame, date_col: str, price_col: str) -> pd.DataFrame:
    out = df[[date_col, price_col]].copy()
    out[date_col] = coerce_datetime(out[date_col])
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col)
    out = out.set_index(date_col)
    out = out.loc[:, [price_col]].astype(float)
    out["log_price"] = np.log(out[price_col])
    out["annual_log_change"] = out["log_price"] - out["log_price"].shift(12)
    out = out.dropna(subset=["annual_log_change"])
    return out

def kde_fit(x: np.ndarray, bw_method: str | float = "scott"):
    # gaussian_kde supports "scott", "silverman", or scalar factor
    return stats.gaussian_kde(x, bw_method=bw_method)

def kde_pdf_on_grid(kde, grid: np.ndarray) -> np.ndarray:
    return kde(grid)

def kde_tail_probs(kde, grid: np.ndarray, pdf_vals: np.ndarray, lower: float, upper: float) -> tuple[float, float, float]:
    # Numerical integration via trapezoid rule on a dense grid
    dx = np.diff(grid)
    mid_pdf = (pdf_vals[:-1] + pdf_vals[1:]) * 0.5
    # Build CDF by integrating from left
    cdf_vals = np.concatenate([[0.0], np.cumsum(mid_pdf * dx)])
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

    # Interpolate CDF at arbitrary points
    def interp_cdf(x):
        return np.interp(x, grid, cdf_vals, left=0.0, right=1.0)

    p_lower = interp_cdf(lower)  # P(X <= lower)
    p_upper = 1.0 - interp_cdf(upper)  # P(X >= upper)
    p_two_tail = p_lower + p_upper
    return float(p_lower), float(p_upper), float(p_two_tail)

def normal_tail_probs(mu: float, sigma: float, lower: float, upper: float) -> tuple[float, float, float]:
    dist = stats.norm(loc=mu, scale=sigma)
    p_lower = dist.cdf(lower)
    p_upper = 1.0 - dist.cdf(upper)
    return float(p_lower), float(p_upper), float(p_lower + p_upper)

def empirical_tail_props(x: np.ndarray, lower: float, upper: float) -> tuple[float, float, float]:
    n = x.size
    if n == 0:
        return 0.0, 0.0, 0.0
    p_lower = float(np.mean(x <= lower))
    p_upper = float(np.mean(x >= upper))
    return p_lower, p_upper, p_lower + p_upper

def nice_num(n):
    return f"{n:,.6f}"

def add_v_line(ax, x, label):
    ax.axvline(x, linestyle="--", linewidth=1)
    ax.text(x, ax.get_ylim()[1]*0.95, label, rotation=90, va="top", ha="right", fontsize=9)

# -------------------------------
# Sidebar — Controls & Template
# -------------------------------
st.sidebar.header("Upload & Settings")

with st.sidebar.expander("Download template", expanded=False):
    sample = pd.DataFrame({
        "Date": pd.date_range("2000-01-01", periods=36, freq="MS"),
        "SpotRate": np.linspace(1.00, 1.30, 36) * (1.0 + 0.03*np.sin(np.linspace(0, 3*np.pi, 36)))
    })
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("Download CSV template", data=buf.getvalue(), file_name="fx_monthly_template.csv", mime="text/csv")

uploaded = st.sidebar.file_uploader("Upload monthly FX data (CSV or Excel)", type=["csv", "xlsx", "xls"])

date_col = st.sidebar.text_input("Date column name", value="Date")
price_col = st.sidebar.text_input("Spot rate column name", value="SpotRate")

bins = st.sidebar.slider("Histogram bins", min_value=10, max_value=120, value=50, step=5)
bw_choice = st.sidebar.selectbox("KDE bandwidth method", options=["scott", "silverman", "custom"], index=0)
bw_custom = None
if bw_choice == "custom":
    bw_custom = st.sidebar.number_input("Custom bandwidth factor (e.g., 0.5 to widen tails)", min_value=0.01, max_value=5.0, value=1.0, step=0.05)

grid_padding = st.sidebar.slider("Grid padding (σ units beyond data)", min_value=1.0, max_value=6.0, value=4.0, step=0.5)

st.sidebar.caption("Tip: Use the template to ensure your file schema.")

st.title("Annual Log FX Changes — Normal vs. Fat Tails")

if uploaded is None:
    st.info("Upload a CSV/Excel with monthly spot rates to begin. Use the template in the sidebar if needed.")
    st.stop()

# -------------------------------
# Data ingestion & processing
# -------------------------------
try:
    raw = parse_input_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if date_col not in raw.columns or price_col not in raw.columns:
    st.error(f"Columns not found. Available: {list(raw.columns)}")
    st.stop()

with st.expander("Preview data"):
    st.dataframe(raw.head(20), use_container_width=True)

try:
    panel = compute_annual_log_changes(raw, date_col=date_col, price_col=price_col)
except Exception as e:
    st.error(f"Failed to compute annual log changes: {e}")
    st.stop()

if panel.empty:
    st.warning("Not enough data to compute 12-month log changes. Provide at least 13 months of data.")
    st.stop()

x = panel["annual_log_change"].to_numpy()
mu = float(np.mean(x))
sigma = float(np.std(x, ddof=1))  # sample std

# Extra moments
skew = float(stats.skew(x, bias=False))
kurt = float(stats.kurtosis(x, fisher=False, bias=False))  # Pearson (normal=3)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Summary statistics")
    st.markdown(f"""
- **Observations:** {len(x)}
- **Mean (μ):** {nice_num(mu)}
- **Std. dev (σ):** {nice_num(sigma)}
- **Skewness:** {nice_num(skew)}
- **Kurtosis (Pearson):** {nice_num(kurt)} (Normal = 3)
    """)

    # Normality tests
    jb_stat, jb_p = stats.jarque_bera(x)
    sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)  # Shapiro limited to n<=5000
    # KS against N(mu, sigma)
    ks_stat, ks_p = stats.kstest((x - mu)/sigma, 'norm')

    st.subheader("Normality tests")
    st.markdown(f"""
- **Jarque–Bera:** statistic = {nice_num(jb_stat)}, p-value = {nice_num(jb_p)}
- **Shapiro–Wilk:** statistic = {nice_num(sh_stat)}, p-value = {nice_num(sh_p)} {'(omitted if n>5000)' if len(x)>5000 else ''}
- **Kolmogorov–Smirnov (vs Normal):** statistic = {nice_num(ks_stat)}, p-value = {nice_num(ks_p)}
    """)

with right_col:
    st.subheader("QQ plot vs Normal")
    fig = plt.figure(figsize=(6, 5))
    stats.probplot(x, dist="norm", sparams=(mu, sigma), plot=plt)
    plt.xlabel("Theoretical Quantiles (Normal)")
    plt.ylabel("Sample Quantiles")
    plt.title("QQ Plot: Annual Log Changes vs Normal")
    st.pyplot(fig, clear_figure=True)

# -------------------------------
# Histogram + Overlays
# -------------------------------
st.subheader("Distribution: Histogram, Normal PDF, KDE")

# Grid for overlays
std_span = grid_padding * sigma if sigma > 0 else 1.0
grid_min = float(np.min(x) - std_span)
grid_max = float(np.max(x) + std_span)
grid = np.linspace(grid_min, grid_max, 2000)

# Normal pdf
normal_pdf = stats.norm.pdf(grid, loc=mu, scale=sigma) if sigma > 0 else np.zeros_like(grid)

# KDE
bw = bw_custom if (bw_choice == "custom") else bw_choice
kde = kde_fit(x, bw_method=bw)
kde_pdf = kde_pdf_on_grid(kde, grid)

# Plot
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.hist(x, bins=bins, density=True, alpha=0.35, label="Histogram")
ax.plot(grid, normal_pdf, label="Normal PDF", linewidth=2)
ax.plot(grid, kde_pdf, label=f"KDE PDF (bw={bw})", linewidth=2)
add_v_line(ax, mu, "μ")
add_v_line(ax, mu - 1.96*sigma, "μ - 1.96σ")
add_v_line(ax, mu + 1.96*sigma, "μ + 1.96σ")
ax.set_xlabel("Annual log change")
ax.set_ylabel("Density")
ax.set_title("Annual Log FX Changes: Histogram with Normal & KDE Overlays")
ax.legend()
st.pyplot(fig, clear_figure=True)

# -------------------------------
# Tail probabilities
# -------------------------------
lower_thr = mu - 1.96 * sigma
upper_thr = mu + 1.96 * sigma

pN_low, pN_high, pN_two = normal_tail_probs(mu, sigma, lower_thr, upper_thr)

kde_p_low, kde_p_high, kde_p_two = kde_tail_probs(kde, grid, kde_pdf, lower_thr, upper_thr)

emp_p_low, emp_p_high, emp_p_two = empirical_tail_props(x, lower_thr, upper_thr)

st.subheader("Tail risk beyond μ ± 1.96σ")
st.markdown(f"""
| Model | P(X ≤ μ−1.96σ) | P(X ≥ μ+1.96σ) | Two-tail |
|---|---:|---:|---:|
| **Normal** | {pN_low:.4%} | {pN_high:.4%} | {pN_two:.4%} |
| **KDE (non-parametric)** | {kde_p_low:.4%} | {kde_p_high:.4%} | {kde_p_two:.4%} |
| **Empirical** | {emp_p_low:.4%} | {emp_p_high:.4%} | {emp_p_two:.4%} |
""")

st.caption("Under a true Normal, the two-tail probability at ±1.96σ is ≈ 5%. Deviations above this suggest fat tails.")

# -------------------------------
# Data export
# -------------------------------
with st.expander("Download transformed data (annual log changes)"):
    out = panel.copy()
    out = out.reset_index()
    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button("Download CSV", data=csv_buf.getvalue(), file_name="annual_log_changes.csv", mime="text/csv")

st.success("Done. Explore different KDE bandwidths to see how tail estimates change. Try crisis periods vs tranquil periods for contrast.")
