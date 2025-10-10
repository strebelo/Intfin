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
    price_candidates = [
        "spotrate", "spot", "rate", "price", "exchangerate", "fx", "spotusd", "usdusd"
    ]

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
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0:
        raise ValueError("Standard deviation is zero; KDE cannot be fit.")

    z = ((x - mu) / sigma).reshape(-1, 1)
    bandwidths = np.linspace(0.1, 1.5, 25)

    n = len(z)
    k = max(2, min(cv_folds, max(2, n // 5)))
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
    zgrid = ((grid_x - mu) / sigma).reshape(-1, 1)
    logpdf_z = best_kde.score_samples(zgrid)
    pdf_x = np.exp(logpdf_z) / sigma
    return pdf_x

def normal_pdf(x, mu, sigma):
    return stats.norm.pdf(x, loc=mu, scale=sigma)

# --- FIXED TAIL INTEGRATION (Option A) ---
def tail_prob_from_pdf(grid_x: np.ndarray, pdf_x: np.ndarray, mu: float, sigma: float, k: float = 1.96) -> float:
    left = mu - k * sigma
    right = mu + k * sigma
    left_mask = grid_x < left
    right_mask = grid_x > right
    area_left = float(np.trapz(pdf_x[left_mask], grid_x[left_mask])) if np.any(left_mask) else 0.0
    area_right = float(np.trapz(pdf_x[right_mask], grid_x[right_mask])) if np.any(right_mask) else 0.0
    return area_left + area_right

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
    date_col = st.selectbox("Date column", options=list(raw.columns),
                            index=(list(raw.columns).index(date_guess) if date_guess in raw.columns else 0))
with col2:
    price_col = st.selectbox("Spot rate column", options=list(raw.columns),
                             index=(list(raw.columns).index(price_guess) if price_guess in raw.columns else (1 if len(raw.columns) > 1 else 0)))

try:
    panel = compute_annual_log_changes(raw, date_col, price_col)
except Exception as e:
    st.error(f"Problem computing annual log changes: {e}")
    st.stop()

if panel["annual_log_change"].empty:
    st.warning("Not enough data to compute 12-month log changes.")
    st.stop()

x = panel["annual_log_change"].dropna().values
mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
skew, kurt = float(stats.skew(x, bias=False)), float(stats.kurtosis(x, fisher=False, bias=False))

jb_stat, jb_p = stats.jarque_bera(x)
sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)

try:
    kde_cv, mu_kde, sigma_kde, best_bw = fit_kde_cv_gaussian(x, cv_folds=5)
except Exception as e:
    st.error(f"KDE fit failed: {e}")
    st.stop()

st.subheader("Summary statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean (μ)", f"{mu:.4f}")
c2.metric("Std (σ)", f"{sigma:.4f}")
c3.metric("Skewness (Normal = 0)", f"{skew:.4f}")
c4.metric("Kurtosis (Normal = 3)", f"{kurt:.4f}")
st.caption(f"Best KDE bandwidth (z-scale): {best_bw:.3f}")

st.subheader("Normality tests")
t1, t2 = st.columns(2)
t1.write("**Jarque–Bera**")
t1.write(f"stat = {jb_stat:.3f}, p = {jb_p:.3g}")
t2.write("**Shapiro–Wilk**")
t2.write("n ≤ 5000 required" if np.isnan(sh_stat) else f"stat = {sh_stat:.3f}, p = {sh_p:.3g}")

# -------------------------------
# Distribution Plot (with toggles)
# -------------------------------
st.subheader("Distribution: Histogram with optional overlays")
ocol1, ocol2, ocol3 = st.columns(3)
show_normal = ocol1.checkbox("Show Normal distribution overlay", True)
show_kde = ocol2.checkbox("Show Kernel Density Estimation (CV Gaussian) overlay", True)
show_ci = ocol3.checkbox("Show 95% Normal interval (μ ± 1.96σ)", True)

pad_stds = 1.0
xmin, xmax = np.min(x), np.max(x)
lo = min(mu - 4 * sigma, xmin) - pad_stds * sigma
hi = max(mu + 4 * sigma, xmax) + pad_stds * sigma
grid = np.linspace(lo - 0.5 * sigma, hi + 0.5 * sigma, 4000)

pdf_norm = normal_pdf(grid, mu, sigma) if show_normal else None
pdf_kde = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde) if show_kde else None

fig1, ax1 = plt.subplots(figsize=(7, 4.25))
ax1.hist(x, bins="auto", density=True, alpha=0.6, edgecolor="black", label="Histogram")

if show_ci:
    left, right = mu - 1.96 * sigma, mu + 1.96 * sigma
    ax1.axvspan(left, right, alpha=0.15, label="95% Normal interval (μ ± 1.96σ)")
if show_normal:
    ax1.plot(grid, pdf_norm, linewidth=2, label="Normal PDF")
if show_kde:
    ax1.plot(grid, pdf_kde, linewidth=2, linestyle="--", label="Kernel Density Estimation (CV Gaussian) PDF")

ax1.set_xlabel("Annual log change")
ax1.set_ylabel("Density")
ax1.set_title("Histogram with Optional Normal / KDE Overlays")
ax1.grid(True, linestyle=":", linewidth=0.8)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize="small")
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
st.pyplot(fig1)

# -------------------------------
# Tail probabilities beyond μ ± 1.96σ
# -------------------------------
st.subheader("Tail probabilities beyond μ ± 1.96σ")
p_tail_norm = tail_prob_normal(1.96)
pdf_kde_full = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde)
p_tail_kde = tail_prob_from_pdf(grid, pdf_kde_full, mu, sigma, 1.96)
threshold = 1.96 * sigma
p_tail_emp = float(np.mean(np.abs(x - mu) > threshold))

t1, t2, t3 = st.columns(3)
t1.metric("Normal model", f"{p_tail_norm:.4f}")
t2.metric("KDE (CV Gaussian)", f"{p_tail_kde:.4f}")
t3.metric("Empirical proportion", f"{p_tail_emp:.4f}")
st.caption("Under a Normal distribution, P(|Z|>1.96) ≈ 0.0500. Differences vs. CV Gaussian KDE highlight fat/thin tails.")

# -------------------------------
# Forecast: Spot path & 95% Normal CI by month
# -------------------------------
st.subheader("Forecast: 95% Normal-confidence interval for the spot by month")

# Inputs
default_spot = float(panel[price_col].iloc[-1]) if len(panel) else 1.0
spot_now = st.number_input("Current spot (S₀)", min_value=0.0, value=round(default_spot, 6), format="%.6f")
horizon_m = st.number_input("Horizon (months)", min_value=1, max_value=240, value=12, step=1)

# Compute only if valid
if spot_now <= 0.0:
    st.warning("Please enter a positive current spot to compute the forecast.")
else:
    # Annual -> monthly scaling under Normal i.i.d. log changes assumption
    mu_month = mu / 12.0
    sigma_month = sigma / np.sqrt(12.0)

    # For month h: cumulative mean and std of log change
    months = np.arange(1, int(horizon_m) + 1, dtype=int)
    mu_h = months * mu_month
    sigma_h = sigma * np.sqrt(months / 12.0)  # same as sigma_month * sqrt(months)

    # Point forecast and 95% CI for the SPOT (level), using log-normal mapping
    point = spot_now * np.exp(mu_h)
    lower = spot_now * np.exp(mu_h - 1.96 * sigma_h)
    upper = spot_now * np.exp(mu_h + 1.96 * sigma_h)

    # Future dates (month starts) based on last data timestamp, if available
    last_date = panel.index.max() if isinstance(panel.index, pd.DatetimeIndex) and len(panel.index) else None
    if pd.notna(last_date):
        future_dates = pd.date_range((last_date + pd.offsets.MonthBegin(1)).replace(day=1), periods=len(months), freq="MS")
    else:
        future_dates = pd.RangeIndex(1, len(months) + 1, name="Month")

    # Table
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "month_ahead": months,
        "spot_point": point,
        "spot_lower_95": lower,
        "spot_upper_95": upper
    })

    st.dataframe(
        forecast_df.assign(
            spot_point=lambda d: d["spot_point"].round(6),
            spot_lower_95=lambda d: d["spot_lower_95"].round(6),
            spot_upper_95=lambda d: d["spot_upper_95"].round(6),
        ),
        use_container_width=True
    )

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast table (CSV)",
        data=csv_bytes,
        file_name="fx_spot_normal_CI_forecast.csv",
        mime="text/csv"
    )

    # Plot the forecast with CI band
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))

    x_axis = np.arange(len(months))
    ax2.plot(x_axis, point, linewidth=2, label="Spot point forecast (Normal)")
    ax2.fill_between(x_axis, lower, upper, alpha=0.2, label="95% CI (Normal)")

    # ---- SAFE TICK HANDLING (fixes the earlier SyntaxError and ensures matching labels) ----
    step = max(1, len(x_axis) // 12)  # aim for ~12 ticks max
    tick_idx = np.arange(0, len(x_axis), step)

    if isinstance(future_dates, pd.DatetimeIndex):
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([d.strftime("%Y-%m") for d in future_dates[tick_idx]], rotation=45, ha="right")
        ax2.set_xlabel("Forecast month")
    else:
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([str(m) for m in months[tick_idx]], rotation=0, ha="center")
        ax2.set_xlabel("Months ahead")
    # ---------------------------------------------------------------------------------------

    ax2.set_ylabel("Spot")
    ax2.set_title("Spot forecast under Normal assumption (95% CI)")
    ax2.grid(True, linestyle=":", linewidth=0.8)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize="small")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    st.pyplot(fig2)

# -------------------------------
# Diagnostics (temporary; safe to delete/comment out)
# -------------------------------
with st.expander("Diagnostics (temporary; safe to delete)"):
    total_area = float(np.trapz(pdf_kde_full, grid))
    st.write(f"DEBUG — KDE total area over grid: **{total_area:.4f}**")
    st.write(f"DEBUG — grid range: **[{grid.min():.6f}, {grid.max():.6f}]**")
    st.write(f"DEBUG — μ = {mu:.6f}, σ = {sigma:.6f}, 95% bounds: "
             f"[{(mu - 1.96*sigma):.6f}, {(mu + 1.96*sigma):.6f}]")
    st.write(f"DEBUG — Tail (Normal): **{p_tail_norm:.4f}**")
    st.write(f"DEBUG — Tail (KDE, trapezoid fixed): **{p_tail_kde:.4f}**")
    st.write(f"DEBUG — Tail (Empirical): **{p_tail_emp:.4f}**")

    try:
        z_samp = kde_cv.sample(100_000, random_state=0)
        x_samp = mu_kde + sigma_kde * z_samp.ravel()
        p_tail_mc = float(np.mean(np.abs(x_samp - mu) > 1.96 * sigma))
        st.write(f"DEBUG — Tail (KDE Monte Carlo ~100k): **{p_tail_mc:.4f}**")
    except Exception as e:
        st.write(f"DEBUG — KDE Monte Carlo sampling failed: {e}")

    if st.checkbox("Run bandwidth sensitivity check (z-scale)", value=False):
        bws = [0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20]
        rows = []
        z = ((x - mu) / sigma).reshape(-1, 1)
        for bw in bws:
            kde_tmp = KernelDensity(kernel="gaussian", bandwidth=bw).fit(z)
            pdf_tmp = evaluate_kde_pdf_on_grid(kde_tmp, grid, mu, sigma)
            tail_tmp = tail_prob_from_pdf(grid, pdf_tmp, mu, sigma, 1.96)
            area_tmp = float(np.trapz(pdf_tmp, grid))
            rows.append({"bandwidth_z": bw, "kde_tail": tail_tmp, "total_area": area_tmp})
        st.dataframe(pd.DataFrame(rows))
