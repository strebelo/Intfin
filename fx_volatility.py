# fx_volatility.py
# Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails (robust column detection & parsing)

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# Optional KDE with cross-validated bandwidth
KDE_AVAILABLE = True
try:
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KernelDensity
except Exception:
    KDE_AVAILABLE = False

st.set_page_config(page_title="FX Annual Changes — Normal vs. Fat Tails", layout="wide")

st.title("Annual Log FX Changes — Normal vs. Fat Tails")

st.markdown(
    "Upload a **monthly FX spot rate series** (CSV/XLS/XLSX). "
    "Then select which columns are Date and Spot. The app computes annual log changes (log Sₜ − log Sₜ₋₁₂), "
    "tests normality (Shapiro–Wilk, Anderson–Darling), and visualizes the distribution."
)

# -----------------------
# Helpers
# -----------------------
COMMON_SPOT_NAMES = {
    "spot", "spot rate", "spot_rate", "rate", "fx", "price", "value",
    "usd", "eur", "gbp", "jpy", "cad", "aud", "chf", "cny", "brl"
}
COMMON_DATE_NAMES = {"date", "month", "period", "time"}

def read_table(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        # Handle different CSV encodings & separators more gracefully
        try:
            return pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";")
    else:
        # Read first sheet by default
        return pd.read_excel(uploaded)

def guess_date_col(cols):
    # Prefer explicit date-ish names
    for c in cols:
        if c.lower().strip() in COMMON_DATE_NAMES:
            return c
    # Fallback: first column
    return cols[0] if cols else None

def guess_spot_col(df, exclude_col=None):
    # Prefer obvious names first
    for c in df.columns:
        if c == exclude_col: 
            continue
        if c.lower().strip() in COMMON_SPOT_NAMES:
            return c
    # Then prefer numeric-looking columns
    numeric_like = []
    for c in df.columns:
        if c == exclude_col:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_like.append(c)
        else:
            # If >80% of non-null values look like numbers (possibly with symbols), consider it
            nn = s.dropna().astype(str)
            looks_num = nn.str.contains(r"^-?[\s$€£]?(\d{1,3}([,.\s]\d{3})*|\d+)([.,]\d+)?\s*%?$")
            if len(nn) > 0 and looks_num.mean() > 0.8:
                numeric_like.append(c)
    return numeric_like[0] if numeric_like else None

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Attempt multiple passes to convert a messy numeric column (currency symbols, %,
    thousand separators, decimal commas) into float.
    Strategy:
      1) Strip currency/%/spaces, remove thousands separators, interpret '.' as decimal.
      2) If many commas and few dots -> treat comma as decimal, dot as thousands.
      3) Fallback to pandas to_numeric with errors='coerce'.
    """
    s0 = s.astype(str).str.strip()

    # Remove currency and percent signs
    s1 = s0.str.replace(r"[\s$€£¥%]", "", regex=True)

    # Heuristic: decide decimal separator
    # Count occurrences of '.' and ',' per value
    sample = s1.dropna().head(200)
    comma_ratio = sample.str.count(",").mean() if len(sample) else 0
    dot_ratio = sample.str.count(r"\.").mean() if len(sample) else 0

    # Case A: Use dot as decimal, remove commas as thousands
    a = pd.to_numeric(s1.str.replace(",", "", regex=False), errors="coerce")

    # Case B: Use comma as decimal — swap comma->dot, remove thousands dots
    b_tmp = s1.str.replace(".", "", regex=False)  # remove thousands dots
    b = pd.to_numeric(b_tmp.str.replace(",", ".", regex=False), errors="coerce")

    # Choose better parse: pick the version with more non-nulls
    if b.notna().sum() > a.notna().sum():
        parsed = b
    else:
        parsed = a

    # Final fallback
    parsed = pd.to_numeric(parsed, errors="coerce")

    return parsed

def ensure_monthly(df):
    """Force monthly start frequency and interpolate small gaps."""
    out = df.copy()
    out = out.set_index("Date").sort_index()
    try:
        out = out.asfreq("MS")
    except Exception:
        # If Date isn't aligned to month starts, round down to month start
        out.index = out.index.to_period("M").to_timestamp("MS")
        out = out.asfreq("MS")
    out = out.interpolate(limit_direction="both")
    return out.reset_index()

# -----------------------
# UI: File upload
# -----------------------
file = st.file_uploader("Upload CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])

if not file:
    st.info("Upload a file to begin.")
    st.stop()

df_raw = read_table(file)
if df_raw is None or df_raw.empty:
    st.error("Could not read the file or it is empty.")
    st.stop()

st.write("**Preview of uploaded data (first 8 rows):**")
st.dataframe(df_raw.head(8))

# Let user choose Date and Spot columns
st.subheader("Select Columns")
date_guess = guess_date_col(df_raw.columns.tolist())
spot_guess = guess_spot_col(df_raw, exclude_col=date_guess)

date_col = st.selectbox("Date column", options=df_raw.columns.tolist(), index=(df_raw.columns.tolist().index(date_guess) if date_guess in df_raw.columns else 0))
spot_col = st.selectbox("Spot column", options=[c for c in df_raw.columns if c != date_col],
                        index=([c for c in df_raw.columns if c != date_col].index(spot_guess) if spot_guess in [c for c in df_raw.columns if c != date_col] else 0))

# Parse dates
df = df_raw[[date_col, spot_col]].copy()
df.rename(columns={date_col: "Date", spot_col: "SpotRaw"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Clean spot to numeric
if pd.api.types.is_numeric_dtype(df["SpotRaw"]):
    df["Spot"] = df["SpotRaw"].astype(float)
else:
    df["Spot"] = clean_numeric_series(df["SpotRaw"])

# Show parsing diagnostics
miss = df["Spot"].isna().mean()
with st.expander("Column parsing diagnostics"):
    st.write(f"Selected **Date**: `{date_col}`  |  Selected **Spot**: `{spot_col}`")
    st.write(f"Proportion of unparseable Spot values: **{miss:.2%}** (NaNs after cleaning)")
    st.write("Sample cleaned values:")
    st.write(df[["Date", "SpotRaw", "Spot"]].head(10))

if df["Spot"].notna().sum() < 20:
    st.error("After cleaning, fewer than 20 numeric spot observations are available. Please check your Spot column selection/format.")
    st.stop()

# Enforce monthly frequency & fill small gaps
df2 = ensure_monthly(df[["Date", "Spot"]])

st.success("Data parsed successfully.")
st.write(df2.head())

# -----------------------------------------
# Compute annual log changes (12-month diff)
# -----------------------------------------
df2["log_spot"] = np.log(df2["Spot"])
df2["ann_log_change"] = df2["log_spot"].diff(12)
series = df2["ann_log_change"].dropna()

if len(series) < 20:
    st.warning("Not enough 12-month observations to proceed (need ≥ 20).")
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
c3.metric("Skewness (Normal=0)", f"{skew_:.4f}")
c4.metric("Kurtosis (raw; Normal=3)", f"{kurt_:.4f}")

# ------------------------------
# Normality tests (only SW & AD)
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
# Histogram & overlays (Matplotlib)
# ------------------------------
st.subheader("Distribution Visualization")

show_norm = st.checkbox("Show Normal overlay", value=True)
show_kde = st.checkbox("Show Kernel Density Estimation overlay", value=False,
                       help="Gaussian kernel with cross-validated bandwidth (if scikit-learn is installed).")
bins = st.number_input("Number of bins", min_value=8, max_value=80, value=24, step=1)

fig, ax = plt.subplots(figsize=(8, 5))
counts, bins_edges, _ = ax.hist(series, bins=bins, density=True, alpha=0.6, edgecolor="black")

x_grid = np.linspace(series.min(), series.max(), 400)

# Normal overlay
if show_norm and std_ > 0:
    ax.plot(x_grid, stats.norm.pdf(x_grid, loc=mean_, scale=std_), "r--", lw=2, label="Normal (μ, σ)")

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
        st.warning("scikit-learn not installed; KDE overlay unavailable.")

ax.set_xlabel("Annual log change")
ax.set_ylabel("Density")
ax.set_title("Distribution of Annual Log FX Changes")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

st.pyplot(fig)
