# fx_volatility.py
# Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails (interactive)
#
# Run: streamlit run fx_volatility.py

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from scipy import stats

# Optional KDE with CV bandwidth (sklearn)
KDE_AVAILABLE = True
try:
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KernelDensity
except Exception:
    KDE_AVAILABLE = False

st.set_page_config(page_title="FX Annual Changes — Normal vs. Fat Tails", layout="wide")

st.title("Annual Log FX Changes — Interactive Histogram")

st.markdown(
    "Upload a **monthly** FX spot series (CSV/XLS/XLSX) with a date column and a spot column. "
    "The app computes annual log changes (log Sₜ − log Sₜ₋₁₂), tests normality, and visualizes the distribution."
)

# -----------------------
# File upload & parsing
# -----------------------
file = st.file_uploader("Upload file", type=["csv", "xls", "xlsx"])

def read_table(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".csv"):
        data = pd.read_csv(f)
    else:
        data = pd.read_excel(f)
    return data

def coerce_date_col(df):
    # Heuristics: look for a column named 'date' (any case) or the first column that parses as dates
    date_candidates = [c for c in df.columns if c.lower() in ["date", "month", "period"]]
    if date_candidates:
        dc = date_candidates[0]
    else:
        dc = df.columns[0]  # fallback to first column
    out = df.copy()
    out[dc] = pd.to_datetime(out[dc], errors="coerce")
    out = out.dropna(subset=[dc]).rename(columns={dc: "Date"})
    return out

def find_spot_col(df):
    # Try common names; else take the first numeric column that's not Date
    candidates = [c for c in df.columns if c.lower().strip() in ["spot", "spot rate", "spot_rate", "price", "value", "fx", "rate"]]
    if candidates:
        return candidates[0]
    # find the first numeric non-date column
    for c in df.columns:
        if c == "Date":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

if file:
    raw = read_table(file)
    try:
        tbl = coerce_date_col(raw)
    except Exception:
        st.error("Could not identify/parse a Date column.")
        st.stop()

    spot_col = find_spot_col(tbl)
    if spot_col is None:
        st.error("Could not find a numeric spot column. Please include a column like 'Spot rate'.")
        st.stop()

    df = tbl[["Date", spot_col]].dropna().sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date").asfreq("MS").interpolate()  # enforce monthly start; fill gaps if any
    df = df.reset_index()

    st.success(f"Detected date column **Date** and spot column **{spot_col}**.")
    st.write(df.head())

    # -----------------------------------------
    # Compute annual log changes (12-month diff)
    # -----------------------------------------
    df["log_spot"] = np.log(df[spot_col])
    df["ann_log_change"] = df["log_spot"].diff(12)

    series = df["ann_log_change"].dropna()
    if len(series) < 20:
        st.warning("Not enough 12-month observations to proceed (need ≥ 20).")
        st.stop()

    st.subheader("Summary Statistics")
    mean_ = series.mean()
    std_ = series.std(ddof=1)
    skew_ = stats.skew(series, bias=False)
    kurtosis_raw = stats.kurtosis(series, fisher=False, bias=False)  # NOT excess

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean (annual log change)", f"{mean_:.4f}")
    c2.metric("Std. Dev.", f"{std_:.4f}")
    c3.metric("Skewness", f"{skew_:.4f}")
    c4.metric("Kurtosis (raw)", f"{kurtosis_raw:.4f}")

    # ----------------------
    # Normality tests (only SW & AD)
    # ----------------------
    st.subheader("Normality Tests")
    sh_w, sh_p = stats.shapiro(series)
    st.write(f"**Shapiro–Wilk:** W = {sh_w:.4f}, p-value = {sh_p:.4f}")

    ad_res = stats.anderson(series, dist="norm")
    st.write(f"**Anderson–Darling:** A² = {ad_res.statistic:.4f}")
    with st.expander("Anderson–Darling critical values"):
        for cv, sig in zip(ad_res.critical_values, ad_res.significance_level):
            st.write(f"- {int(sig)}%: {cv:.3f}  → reject if A² > {cv:.3f}")

    st.caption(
        "Interpretation: Shapiro–Wilk p < 0.05 ⇒ reject normality. "
        "Anderson–Darling statistic above a critical value ⇒ reject at that significance level."
    )

    # ---------------------------------------
    # Histogram + interactive click-to-read
    # ---------------------------------------
    st.subheader("Distribution (Interactive Histogram)")
    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_right:
        show_norm = st.checkbox("Show Normal overlay", value=True)
        show_kde = st.checkbox("Show Kernel Density Estimation overlay", value=False,
                               help="Gaussian kernel with cross-validated bandwidth (if scikit-learn is available).")
        bins = st.number_input("Bins", min_value=8, max_value=80, value=24, step=1)

    with col_left:
        # Build histogram as probability (fraction per bin)
        hist_counts, bin_edges = np.histogram(series.values, bins=bins, density=False)
        fractions = hist_counts / len(series)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_widths = np.diff(bin_edges)

        fig = go.Figure()

        fig.add_bar(
            x=bin_centers,
            y=fractions,
            width=bin_widths,
            name="Histogram (fraction)",
            hovertemplate="Bin center: %{x:.4f}<br>Fraction: %{y:.4f}<extra></extra>",
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.75,
        )

        # Prepare x-grid for overlays
        x_grid = np.linspace(bin_edges[0], bin_edges[-1], 400)
        bw_for_prob = np.mean(bin_widths)  # to convert density→probability-per-bin: multiply by bin width

        # Normal overlay (match histogram scale: prob mass per avg bin)
        if show_norm:
            from math import sqrt, pi, exp
            mu, sd = mean_, std_
            if sd > 0:
                norm_pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu) / sd) ** 2)
                norm_prob_per_bin = norm_pdf * bw_for_prob
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=norm_prob_per_bin,
                        mode="lines",
                        name="Normal (μ, σ) × bin width",
                        hovertemplate="x: %{x:.4f}<br>Prob/bin: %{y:.5f}<extra></extra>",
                    )
                )

        # KDE overlay (Gaussian kernel + CV bandwidth), plotted as prob per avg bin
        if show_kde:
            if KDE_AVAILABLE:
                X = series.values.reshape(-1, 1)
                # Bandwidth grid: log-spaced over a reasonable range
                bw_grid = np.logspace(-3, 0, 30)
                cv = KFold(n_splits=min(10, len(series)//5 if len(series) >= 50 else 5), shuffle=True, random_state=42)
                grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bw_grid}, cv=cv)
                grid.fit(X)
                best_bw = grid.best_params_["bandwidth"]

                kde = KernelDensity(kernel="gaussian", bandwidth=best_bw)
                kde.fit(X)
                log_dens = kde.score_samples(x_grid.reshape(-1, 1))
                dens = np.exp(log_dens)  # density integrates to 1
                kde_prob_per_bin = dens * bw_for_prob

                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=kde_prob_per_bin,
                        mode="lines",
                        name=f"KDE (Gaussian, bw={best_bw:.4f}) × bin width",
                        hovertemplate="x: %{x:.4f}<br>Prob/bin: %{y:.5f}<extra></extra>",
                    )
                )
            else:
                st.warning(
                    "KDE overlay requires scikit-learn. Install with `pip install scikit-learn` "
                    "or uncheck the KDE option."
                )

        fig.update_layout(
            xaxis_title="Annual log change",
            yaxis_title="Fraction of sample",
            bargap=0.02,
            margin=dict(
