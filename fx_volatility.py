# fx_volatility.py
# Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails
#
# Run: streamlit run fx_volatility.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# Optional: KDE with cross-validated bandwidth
KDE_AVAILABLE = True
try:
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KernelDensity
except Exception:
    KDE_AVAILABLE = False

st.set_page_config(page_title="FX Annual Changes — Normal vs. Fat Tails", layout="wide")

st.title("Annual Log FX Changes — Normal vs. Fat Tails")

st.markdown(
    "Upload a **monthly FX spot rate series** (CSV/XLS/XLSX) with a date column and a spot column. "
    "The app computes annual log changes (log Sₜ − log Sₜ₋₁₂), tests for normality, and visualizes the distribution."
)

# -----------------------
# File upload & parsing
# -----------------------
file = st.file_uploader("Upload file", type=["csv", "xls", "xlsx"])

def read_table(f):
    if f.name.lower().endswith(".csv"):
        return pd.read_csv(f)
    else:
        return pd.read_excel(f)

def coerce_date_col(df):
    date_candidates = [c for c in df.columns if c.lower() in ["date", "month", "period"]]
    if date_candidates:
        dc = date_candidates[0]
    else:
        dc = df.columns[0]
    df = df.rename(columns={dc: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])

def find_spot_col(df):
    candidates = [c for c in df.columns if c.lower() in ["spot", "spot rate", "spot_rate", "price", "fx", "rate", "value"]]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if c != "Date" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

if file:
    df = read_table(file)
    df = coerce_date_col(df)
    spot_col = find_spot_col(df)
    if spot_col is None:
        st.error("Could not find a numeric spot column.")
        st.stop()

    df = df[["Date", spot_col]].dropna().sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date").asfreq("MS").interpolate()
    df = df.reset_index()

    st.success(f"Detected date column **Date** and spot column **{spot_col}**.")
    st.write(df.head())

    # ------------------------------
    # Compute annual log changes
    # ------------------------------
    df["log_spot"] = np.log(df[spot_col])
    df["ann_log_change"] = df["log_spot"].diff(12)
    series = df["ann_log_change"].dropna()

    if len(series) < 20:
        st.warning("Not enough 12-month observations (need at least 20).")
        st.stop()

    # ------------------------------
    # Summary statistics
    # ------------------------------
    st.subheader("Summary Statistics")
    mean_ = series.mean()
    std_ = series.std(ddof=1)
    skew_ = stats.skew(series, bias=False)
    kurt_ = stats.kurtosis(series, fisher=False, bias=False)  # raw kurtosis

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{mean_:.4f}")
    c2.metric("Std. Dev.", f"{std_:.4f}")
    c3.metric("Skewness", f"{skew_:.4f}")
    c4.metric("Kurtosis", f"{kurt_:.4f}")

    # ------------------------------
    # Normality tests
    # ------------------------------
    st.subheader("Normality Tests")
    sh_w, sh_p = stats.shapiro(series)
    st.write(f"**Shapiro–Wilk:** W = {sh_w:.4f}, p-value = {sh_p:.4f}")

    ad_res = stats.anderson(series, dist="norm")
    st.write(f"**Anderson–Darling:** A² = {ad_res.statistic:.4f}")
    with st.expander("Anderson–Darling critical values"):
        for cv, sig in zip(ad_res.critical_values, ad_res.significance_level):
            st.write(f"- {int(sig)}%: {cv:.3f}")

    st.caption("Reject normality if Shapiro–Wilk p < 0.05 or Anderson–Darling statistic > critical value.")

    # ------------------------------
    # Histogram & overlays
    # ------------------------------
    st.subheader("Distribution Visualization")

    show_norm = st.checkbox("Show Normal overlay", value=True)
    show_kde = st.checkbox("Show Kernel Density Estimation overlay", value=False)
    bins = st.number_input("Number of bins", min_value=8, max_value=80, value=24, step=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bins_edges, patches = ax.hist(series, bins=bins, density=True, alpha=0.6, edgecolor="black")

    x_grid = np.linspace(series.min(), series.max(), 400)

    # Normal overlay
    if show_norm:
        norm_pdf = stats.norm.pdf(x_grid, loc=mean_, scale=std_)
        ax.plot(x_grid, norm_pdf, "r--", lw=2, label="Normal (μ, σ)")

    # KDE overlay (Gaussian kernel + CV bandwidth)
    if show_kde:
        if KDE_AVAILABLE:
            X = series.values.reshape(-1, 1)
            bw_grid = np.logspace(-3, 0, 30)
            cv = KFold(n_splits=min(10, len(series)//5 if len(series) >= 50 else 5), shuffle=True, random_state=42)
            grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bw_grid}, cv=cv)
            grid.fit(X)
            best_bw = grid.best_params_["bandwidth"]
            kde = KernelDensity(kernel="gaussian", bandwidth=best_bw).fit(X)
            log_dens = kde.score_samples(x_grid.reshape(-1, 1))
            ax.plot(x_grid, np.exp(log_dens), "b-", lw=2, label=f"KDE (bw={best_bw:.4f})")
        else:
            st.warning("scikit-learn not installed; KDE unavailable.")

    ax.set_xlabel("Annual log change")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Annual Log FX Changes")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)

else:
    st.info("Upload a file to begin.")
