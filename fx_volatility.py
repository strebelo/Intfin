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

# --- FIXED TAIL INTEGRATION ---
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

uploaded = st.file_uploader("Upload CSV or Excel with a Date column and a Spot Rate column", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Awaiting file upload…")
    st.stop()

raw = parse_input_file(uploaded)
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
                             index=(list(raw.columns).index(price_guess) if price_guess in raw.columns else 1))

panel = compute_annual_log_changes(raw, date_col, price_col)
x = panel["annual_log_change"].dropna().values
mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
kde_cv, mu_kde, sigma_kde, best_bw = fit_kde_cv_gaussian(x, cv_folds=5)

# -------------------------------
# Forecast Section
# -------------------------------
st.subheader("Forecast: 95% Normal-confidence interval for the spot by month")

spot_col = panel.columns[0]
last_spot = float(panel[spot_col].iloc[-1])
min_spot = float(panel[spot_col].min())
max_spot = float(panel[spot_col].max())

spot_source = st.selectbox(
    "Spot source for forecast (S₀)",
    ["Custom input", "Last observed in data", "Historical MIN in data", "Historical MAX in data"],
    index=0
)

if spot_source == "Custom input":
    spot_now = st.number_input("Current spot (S₀)", min_value=0.0, value=round(last_spot, 6), format="%.6f")
elif spot_source == "Last observed in data":
    spot_now = last_spot
elif spot_source == "Historical MIN in data":
    spot_now = min_spot
else:
    spot_now = max_spot

horizon_m = st.number_input("Horizon (months)", min_value=1, max_value=240, value=12, step=1)
drift_source = st.selectbox(
    "Mean rate of change (drift) source",
    ["Historical mean (annual log Δ from data)", "Zero drift", "Custom annualized drift (% per year)"],
    index=0
)
custom_drift_pct = 0.0
if drift_source == "Custom annualized drift (% per year)":
    custom_drift_pct = st.number_input("Custom annualized drift (% per year, log-change)", value=0.0, step=0.1, format="%.4f")

if spot_now > 0:
    if drift_source == "Historical mean (annual log Δ from data)":
        mu_annual = mu
    elif drift_source == "Zero drift":
        mu_annual = 0.0
    else:
        mu_annual = float(custom_drift_pct) / 100.0

    mu_month = mu_annual / 12.0
    months = np.arange(1, int(horizon_m) + 1, dtype=int)
    mu_h = months * mu_month
    sigma_h = sigma * np.sqrt(months / 12.0)

    point = spot_now * np.exp(mu_h)
    lower = spot_now * np.exp(mu_h - 1.96 * sigma_h)
    upper = spot_now * np.exp(mu_h + 1.96 * sigma_h)

    last_date = panel.index.max() if isinstance(panel.index, pd.DatetimeIndex) else None
    if pd.notna(last_date):
        future_dates = pd.date_range((last_date + pd.offsets.MonthBegin(1)).replace(day=1), periods=len(months), freq="MS")
    else:
        future_dates = pd.RangeIndex(1, len(months) + 1, name="Month")

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "month_ahead": months,
        "spot_point": point,
        "spot_lower_95": lower,
        "spot_upper_95": upper
    })
    st.dataframe(forecast_df.round(6), use_container_width=True)

    # ---- Plot forecast with legend BELOW ----
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    x_axis = np.arange(len(months))
    ax2.plot(x_axis, point, linewidth=2, label="Spot point forecast (Normal)")
    ax2.fill_between(x_axis, lower, upper, alpha=0.2, label="95% CI (Normal)")

    step = max(1, len(x_axis) // 12)
    tick_idx = np.arange(0, len(x_axis), step)
    if isinstance(future_dates, pd.DatetimeIndex):
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([d.strftime("%Y-%m") for d in future_dates[tick_idx]], rotation=45, ha="right")
        ax2.set_xlabel("Forecast month")
    else:
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([str(m) for m in months[tick_idx]], rotation=0, ha="center")
        ax2.set_xlabel("Months ahead")

    ax2.set_ylabel("Spot")
    ax2.set_title("Spot forecast under Normal assumption (95% CI)")
    ax2.grid(True, linestyle=":", linewidth=0.8)

    # --- Legend BELOW the chart, neatly spaced ---
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        frameon=False,
        fontsize="small"
    )
    ax2.xaxis.labelpad = 10
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.40)  # extra space for x-labels and legend
    st.pyplot(fig2)
else:
    st.warning("Please enter a positive current spot value.")
