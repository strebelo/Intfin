# fx_volatility.py
# Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails
# Gaussian KDE with cross-validated bandwidth (scikit-learn)
#
# Run: streamlit run fx_volatility.py
# Requirements: numpy, pandas, scipy, matplotlib, scikit-learn, streamlit

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

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
        df = pd.read_csv(uploaded_file)
    return df

def coerce_datetime(series: pd.Series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def normalize_colname(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum())

def guess_columns(df: pd.DataFrame):
    norm_map = {col: normalize_colname(col) for col in df.columns}
    inv_map = {v: k for k, v in norm_map.items()}

    date_candidates = ["date", "month", "period", "observationdate"]
    price_candidates = ["spotrate", "spot", "rate", "price", "exchangerate", "fx", "spotusd", "usdusd"]

    date_col = None
    for cand in date_candidates:
        if cand in inv_map:
            date_col = inv_map[cand]
            break
    if date_col is None:
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

# ===== KDE (Gaussian kernel) with cross-validated bandwidth =====

def fit_kde_cv_gaussian(x: np.ndarray, cv_folds: int = 5):
    """
    Fit 1D Gaussian KDE via scikit-learn with CV-chosen bandwidth.
    Standardize x -> z, tune bandwidth on z, then use change of variables.
    Returns (best_kde, mu, sigma, best_bandwidth_z)
    """
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0:
        raise ValueError("Standard deviation is zero; KDE cannot be fit.")

    z = ((x - mu) / sigma).reshape(-1, 1)

    bandwidths = np.linspace(0.1, 1.5, 25)  # broad, sensible range on z-scale

    n = len(z)
    k = max(2, min(cv_folds, max(2, n // 5)))  # at least 2 folds; ~5 obs/fold
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=cv,
        n_jobs=-1
    )
    grid.fit(z)

    best_kde = grid.best_estimator_
    best_bw = float(grid.best_params_["bandwidth"])
    return best_kde, mu, sigma, best_bw

def evaluate_kde_pdf_on_grid(best_kde: KernelDensity, grid_x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """f_X(x) = f_Z((x-mu)/sigma)/sigma"""
    zgrid = ((grid_x - mu) / sigma).reshape(-1, 1)
    logpdf_z = best_kde.score_samples(zgrid)
    return np.exp(logpdf_z) / sigma

def normal_pdf(x, mu, sigma):
    return stats.norm.pdf(x, loc=mu, scale=sigma)

def tail_prob_from_pdf(grid_x: np.ndarray, pdf_x: np.ndarray, mu: float, sigma: float, k: float = 1.96) -> float:
    mask = (np.abs(grid_x - mu) > k * sigma)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pdf_x[mask], grid_x[mask]))

def tail_prob_normal(k: float = 1.96) -> float:
    return float(2 * (1 - stats.norm.cdf(k)))

# -------------------------------
# UI
# -------------------------------

st.title("Annual Log FX Changes — Normal vs. Fat Tails (CV Gaussian KDE)")

st.markdown(
    """
Upload **monthly** spot exchange rates (CSV or Excel).  
We compute the **annual log change** \\(\\Delta \\ell_t = \\log S_t - \\log S_{t-12}\\), then compare a Normal model with a **Kernel Density Estimation (Gaussian kernel, bandwidth via cross-validation)** and report tail risks.
"""
)

uploaded = st.file_uploader("Upload CSV or Excel with a Date column and a Spot Rate column", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Awaiting file upload…")
    st.stop()

# Read and infer columns
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
    date_col = st.selectbox(
        "Date column",
        options=list(raw.columns),
        index=(list(raw.columns).index(date_guess) if date_guess in raw.columns else 0)
    )
with col2:
    price_col = st.selectbox(
        "Spot rate column",
        options=list(raw.columns),
        index=(list(raw.columns).index(price_guess) if price_guess in raw.columns else (1 if len(raw.columns) > 1 else 0))
    )

# Compute annual log changes
try:
    panel = compute_annual_log_changes(raw, date_col, price_col)
except Exception as e:
    st.error(f"Problem computing annual log changes: {e}")
    st.stop()

if panel["annual_log_change"].empty:
    st.warning("Not enough data to compute 12-month log changes. Provide at least 13 monthly observations.")
    st.stop()

x = panel["annual_log_change"].dropna().values

# Summary stats
mu = float(np.mean(x))
sigma = float(np.std(x, ddof=1))
skew = float(stats.skew(x, bias=False))
kurt = float(stats.kurtosis(x, fisher=False, bias=False))  # Pearson kurtosis; Normal = 3

# Normality tests (omit Kolmogorov–Smirnov per request)
jb_stat, jb_p = stats.jarque_bera(x)
sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)

# Fit CV Gaussian KDE
try:
    kde_cv, mu_kde, sigma_kde, best_bw = fit_kde_cv_gaussian(x, cv_folds=5)
except Exception as e:
    st.error(f"KDE fit failed: {e}")
    st.stop()

st.subheader("Summary statistics")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Mean (μ)", f"{mu:.4f}")
mcol2.metric("Std (σ)", f"{sigma:.4f}")
mcol3.metric("Skewness (Normal = 0)", f"{skew:.4f}")
mcol4.metric("Kurtosis (Normal = 3)", f"{kurt:.4f}")

st.caption(f"Kernel Density Estimation: Gaussian kernel with CV bandwidth on standardized data. Best bandwidth (z-scale) = {best_bw:.3f}.")

st.subheader("Normality tests")
t1, t2 = st.columns(2)
with t1:
    st.write("**Jarque–Bera**")
    st.write(f"stat = {jb_stat:.3f}, p = {jb_p:.3g}")
with t2:
    st.write("**Shapiro–Wilk**")
    st.write("n ≤ 5000 required" if np.isnan(sh_stat) else f"stat = {sh_stat:.3f}, p = {sh_p:.3g}")

# -------------------------------
# Distribution Plot (with toggles)
# -------------------------------
st.subheader("Distribution: Histogram with optional overlays")

# Overlays toggles
ocol1, ocol2, ocol3 = st.columns(3)
with ocol1:
    show_normal = st.checkbox("Show Normal distribution overlay", value=True)
with ocol2:
    show_kde = st.checkbox("Show Kernel Density Estimation (CV Gaussian) overlay", value=True)
with ocol3:
    show_ci = st.checkbox("Show 95% Normal interval (μ ± 1.96σ)", value=True)

# Grid: padding fixed to 1σ beyond min/max and ±4σ envelope around μ
pad_stds = 1.0
xmin, xmax = np.min(x), np.max(x)
lo = min(mu - 4 * sigma, xmin) - pad_stds * sigma
hi = max(mu + 4 * sigma, xmax) + pad_stds * sigma
grid = np.linspace(lo, hi, 2000)

# Evaluate overlays
pdf_norm = normal_pdf(grid, mu, sigma) if show_normal else None
pdf_kde = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde) if show_kde else None

fig1, ax1 = plt.subplots(figsize=(7, 4.25))
ax1.hist(x, bins="auto", density=True, alpha=0.6, edgecolor="black", label="Histogram")

legend_handles = []

# 95% Normal CI shading (elegant band)
if show_ci:
    left = mu - 1.96 * sigma
    right = mu + 1.96 * sigma
    band = ax1.axvspan(left, right, alpha=0.15, label="95% Normal interval (μ ± 1.96σ)")
    legend_handles.append(band)

if show_normal:
    (lnorm,) = ax1.plot(grid, pdf_norm, linewidth=2, label="Normal PDF")
    legend_handles.append(lnorm)
if show_kde:
    (lkde,) = ax1.plot(grid, pdf_kde, linewidth=2, linestyle="--", label="Kernel Density Estimation (CV Gaussian) PDF")
    legend_handles.append(lkde)

ax1.set_xlabel("Annual log change")
ax1.set_ylabel("Density")
ax1.set_title("Histogram with Optional Normal / KDE Overlays")
if legend_handles:
    ax1.legend()
ax1.grid(True, linestyle=":", linewidth=0.8)
st.pyplot(fig1)

# -------------------------------
# Tail probabilities beyond μ ± 1.96σ
# -------------------------------
st.subheader("Tail probabilities beyond μ ± 1.96σ")

p_tail_norm = tail_prob_normal(k=1.96)

# Always compute KDE tail on the same grid (even if overlay hidden)
pdf_kde_full = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde)
p_tail_kde = tail_prob_from_pdf(grid, pdf_kde_full, mu, sigma, k=1.96)

threshold = 1.96 * sigma
p_tail_emp = float(np.mean(np.abs(x - mu) > threshold))

tt1, tt2, tt3 = st.columns(3)
tt1.metric("Normal model", f"{p_tail_norm:.4f}")
tt2.metric("Kernel Density Estimation (CV Gaussian)", f"{p_tail_kde:.4f}")
tt3.metric("Empirical proportion", f"{p_tail_emp:.4f}")

st.caption("Under a Normal distribution, P(|Z|>1.96) ≈ 0.0500. Differences vs. CV Gaussian KDE highlight fat/thin tails.")
