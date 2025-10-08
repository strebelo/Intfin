# fx_volatility.py
# Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails
# Author: (your name here)

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

def normalize_colname(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum())

def guess_columns(df: pd.DataFrame):
    # Try to guess date and price columns (handles e.g. "Spot rate")
    norm_map = {col: normalize_colname(col) for col in df.columns}
    inv_map = {v: k for k, v in norm_map.items()}
    # Candidates
    date_candidates = ["date", "month", "period", "observationdate"]
    price_candidates = [
        "spotrate", "spot", "rate", "pricereturn", "price", "exchangerate", "fx", "spotusd", "usdusd"
    ]
    date_col = None
    for cand in date_candidates:
        if cand in inv_map:
            date_col = inv_map[cand]
            break
    if date_col is None:
        # fallback: first column that parses to many datetimes
        for col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if dt.notna().mean() > 0.8:
                date_col = col
                break

    price_col = None
    for cand in price_candidates:
        if cand in inv_map:
            price_col = inv_map[cand]
            break
    if price_col is None:
        # fallback: first numeric-looking column not equal to date
        for col in df.columns:
            if col != date_col and pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.9:
                price_col = col
                break

    return date_col, price_col

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
    return stats.gaussian_kde(x, bw_method=bw_method)

def normal_pdf(x, mu, sigma):
    return stats.norm.pdf(x, loc=mu, scale=sigma)

def tail_probs_normal(mu, sigma, k=1.96):
    # P(|X - mu| > k*sigma) under Normal = 2*(1 - Phi(k))
    p = 2 * (1 - stats.norm.cdf(k))
    return p

def tail_probs_kde(kde, mu, sigma, grid):
    mask = (np.abs(grid - mu) > 1.96 * sigma)
    pdf_vals = kde(grid)
    # Numerical integral using trapezoid
    mass = np.trapz(pdf_vals[mask], grid[mask])
    return mass

# -------------------------------
# UI
# -------------------------------

st.title("Annual Log FX Changes — Normal vs. Fat Tails")

st.markdown(
    """
Upload **monthly** spot exchange rates (CSV or Excel).  
The app computes **annual log changes**: \\(\\Delta\\ell_t = \\log S_t - \\log S_{t-12}\\), then compares Normal vs. nonparametric (KDE) distributions and tail risks.
"""
)

uploaded = st.file_uploader("Upload CSV or Excel with a Date column and a Spot Rate column", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Awaiting file upload…")
    st.stop()

# Read and guess columns
try:
    raw = parse_input_file(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if raw.empty:
    st.error("The uploaded file is empty.")
    st.stop()

date_guess, price_guess = guess_columns(raw)

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Date column", options=list(raw.columns), index=(list(raw.columns).index(date_guess) if date_guess in raw.columns else 0))
with col2:
    price_col = st.selectbox("Spot rate column", options=list(raw.columns), index=(list(raw.columns).index(price_guess) if price_guess in raw.columns else 1 if len(raw.columns) > 1 else 0))

# Compute changes
try:
    panel = compute_annual_log_changes(raw, date_col, price_col)
except Exception as e:
    st.error(f"Problem computing annual log changes: {e}")
    st.stop()

if panel["annual_log_change"].empty:
    st.warning("Not enough data to compute 12-month log changes. Provide at least 13 monthly observations.")
    st.stop()

x = panel["annual_log_change"].dropna().values
mu = float(np.mean(x))
sigma = float(np.std(x, ddof=1))
skew = float(stats.skew(x, bias=False))
kurt = float(stats.kurtosis(x, fisher=True, bias=False))  # excess kurtosis

# Normality tests
jb_stat, jb_p = stats.jarque_bera(x)
sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)  # Shapiro is slow >5k
ks_stat, ks_p = stats.kstest((x - mu)/sigma, 'norm')

# KDE fit
kde = kde_fit(x, bw_method="scott")

st.subheader("Summary statistics")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Mean (μ)", f"{mu:.4f}")
mcol2.metric("Std (σ)", f"{sigma:.4f}")
mcol3.metric("Skewness (Normal = 0)", f"{skew:.4f}")
mcol4.metric("Excess Kurtosis (Normal = 0)", f"{kurt:.4f}")

st.subheader("Normality tests")
t1, t2, t3 = st.columns(3)
with t1:
    st.write("**Jarque–Bera**")
    st.write(f"stat = {jb_stat:.3f}, p = {jb_p:.3g}")
with t2:
    st.write("**Shapiro–Wilk**")
    st.write("n ≤ 5000 required" if np.isnan(sh_stat) else f"stat = {sh_stat:.3f}, p = {sh_p:.3g}")
with t3:
    st.write("**Kolmogorov–Smirnov (vs Normal)**")
    st.write(f"stat = {ks_stat:.3f}, p = {ks_p:.3g}")

# -------------------------------
# Plots (Histogram + Normal & KDE)
# -------------------------------
st.subheader("Distribution: Histogram with Normal and KDE overlays")

# Grid padding: fixed to 1 (one standard deviation each side), per request
# No slider.
pad_stds = 1.0
xmin = min(x)
xmax = max(x)
lo = min(mu - 4 * sigma, xmin) - pad_stds * sigma
hi = max(mu + 4 * sigma, xmax) + pad_stds * sigma
grid = np.linspace(lo, hi, 2000)

pdf_norm = normal_pdf(grid, mu, sigma)
pdf_kde = kde(grid)

fig1, ax1 = plt.subplots(figsize=(7, 4.25))
# histogram density
ax1.hist(x, bins="auto", density=True, alpha=0.6, edgecolor="black")
ax1.plot(grid, pdf_norm, linewidth=2, label="Normal PDF")
ax1.plot(grid, pdf_kde, linewidth=2, linestyle="--", label="KDE PDF")
ax1.set_xlabel("Annual log change")
ax1.set_ylabel("Density")
ax1.set_title("Histogram with Normal and KDE PDFs")
ax1.legend()
ax1.grid(True, linestyle=":", linewidth=0.8)
st.pyplot(fig1)

# -------------------------------
# Tail probabilities
# -------------------------------
st.subheader("Tail probabilities beyond μ ± 1.96σ")

# Normal
p_tail_normal = tail_probs_normal(mu, sigma, k=1.96)

# KDE numeric integral
p_tail_kde = tail_probs_kde(kde, mu, sigma, grid)

# Empirical
threshold = 1.96 * sigma
p_tail_emp = float(np.mean(np.abs(x - mu) > threshold))

tt1, tt2, tt3 = st.columns(3)
tt1.metric("Normal model", f"{p_tail_normal:.4f}")
tt2.metric("KDE model", f"{p_tail_kde:.4f}")
tt3.metric("Empirical proportion", f"{p_tail_emp:.4f}")

st.caption(
    "Note: Under a Normal distribution, P(|X−μ|>1.96σ) ≈ 0.0500. "
    "Differences between KDE and Normal highlight fat/thin tails in the data."
)

# -------------------------------
# Data preview
# -------------------------------
with st.expander("Show computed series"):
    st.dataframe(panel[["annual_log_change"]].rename(columns={"annual_log_change": "Annual log change"}), use_container_width=True)

st.success("Done.")
