
# taylor_rule_app_updated.py
# Streamlit app: Taylor rule with smoothing, selectable inflation series,
# optional forecast upload, MSE, and zero-lower-bound guide.
#
# Usage:
#   streamlit run taylor_rule_app_updated.py

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Taylor Rule Explorer", layout="wide")

st.title("Taylor Rule Explorer")

st.markdown(
    """
This app models the federal funds rate using a Taylor rule with optional interest-rate smoothing.
You can choose the inflation series (Headline, Core CPI, or Core PCE), adjust parameters,
and optionally **upload a forecast file** for inflation and unemployment to extend the path.
"""
)

# ------------------------------
# Helpers
# ------------------------------

STANDARD_COLS = {
    "date": "date",
    # policy rate
    "fed_funds_actual": "fed_funds_actual",
    "actual fed funds rate": "fed_funds_actual",
    "actual fed funds": "fed_funds_actual",
    "actual": "fed_funds_actual",
    "actual rate": "fed_funds_actual",
    "actual fed funds rate": "fed_funds_actual",
    # unemployment
    "unemployment": "unemployment",
    # inflation measures
    "headline inflation": "headline_inflation",
    "headline_inflation": "headline_inflation",
    "cpi": "core_cpi",     # fallback if someone labels as CPI (not ideal but helps)
    "core cpi": "core_cpi",
    "core_cpi": "core_cpi",
    "core pce": "core_pce",
    "core_pce": "core_pce",
    # already-derived column (older files)
    "inflation_used": "inflation_used",
}

LABEL_MAP = {
    "headline_inflation": "Headline inflation",
    "core_cpi": "Core CPI",
    "core_pce": "Core PCE",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase and strip columns for loose matching
    new_cols = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in STANDARD_COLS:
            new_cols[c] = STANDARD_COLS[key]
    df = df.rename(columns=new_cols)
    return df

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df

def compute_taylor_path(df: pd.DataFrame, a: float, b: float,
                        r_star: float, pi_star: float,
                        rho: float, smoothing_lag: int,
                        use_actual_for_init: bool = True,
                        col_infl: str = "inflation_used",
                        col_u: str = "unemployment",
                        col_actual: str = "fed_funds_actual") -> pd.Series:
    """
    Compute i_t = rho * i_{t-L} + (1 - rho) * [ r* + pi* + a*(pi_t - pi*) + b*(u_t - u*) ]
    If there is no i_{t-L} available, falls back to the base term. If use_actual_for_init is True
    and actual is available, uses actual for initial lagged values when possible.
    """
    base = r_star + pi_star + a * (df[col_infl] - pi_star) + b * (df[col_u] - u_star)
    modeled = np.full(len(df), np.nan, dtype=float)

    # Use actuals for initialization if requested and present
    actual = df[col_actual] if col_actual in df.columns else pd.Series([np.nan]*len(df))

    for t in range(len(df)):
        if t - smoothing_lag >= 0:
            i_lag = modeled[t - smoothing_lag]
            # If we still don't have modeled lag (NaN) and actual available, try actual
            if np.isnan(i_lag) and use_actual_for_init and t - smoothing_lag >= 0 and not np.isnan(actual.iloc[t - smoothing_lag]):
                i_lag = actual.iloc[t - smoothing_lag]
        else:
            # Not enough history: try to backfill from actual (t-L)
            if use_actual_for_init and t - smoothing_lag >= 0 and not np.isnan(actual.iloc[t - smoothing_lag]):
                i_lag = actual.iloc[t - smoothing_lag]
            else:
                i_lag = np.nan

        if np.isnan(i_lag):
            # No lag available: revert to no-smoothing base for this step
            modeled[t] = base.iloc[t]
        else:
            modeled[t] = rho * i_lag + (1.0 - rho) * base.iloc[t]

    return pd.Series(modeled, index=df.index, name="fed_funds_modeled")

def merge_with_forecast(hist_df: pd.DataFrame, fcast_df: pd.DataFrame,
                        infl_choice_hist: str, infl_choice_fcast: str) -> pd.DataFrame:
    """
    Build a combined dataframe adding future rows from the forecast file.
    The forecast file should provide unemployment and at least one inflation series.
    """
    fcast_df = fcast_df.copy()
    # Normalize and ensure datetime
    fcast_df = normalize_columns(fcast_df)
    fcast_df = ensure_datetime(fcast_df)

    # Make sure we have required forecast inputs
    available_infl_f = [c for c in ["headline_inflation", "core_cpi", "core_pce"] if c in fcast_df.columns]
    if not available_infl_f:
        st.error("Forecast file must include at least one of: Headline inflation, Core CPI, or Core PCE.")
        st.stop()
    if "unemployment" not in fcast_df.columns:
        st.error("Forecast file must include a column named 'Unemployment'.")
        st.stop()

    # Pick forecast inflation series
    if infl_choice_fcast not in available_infl_f:
        infl_choice_fcast = available_infl_f[0]

    # Prepare forecast slice: keep only dates strictly greater than last historical date
    last_date = hist_df["date"].max()
    fcast_future = fcast_df[fcast_df["date"] > last_date].copy()
    if fcast_future.empty:
        return hist_df  # nothing to add

    # Build aligned columns for future period
    fcast_future = fcast_future[["date", infl_choice_fcast, "unemployment"]].rename(
        columns={infl_choice_fcast: "inflation_used"}
    )
    fcast_future["fed_funds_actual"] = np.nan  # unknown in the future

    # Append
    combined = pd.concat([
        hist_df[["date", "inflation_used", "unemployment", "fed_funds_actual"]],
        fcast_future[["date", "inflation_used", "unemployment", "fed_funds_actual"]]
    ], axis=0, ignore_index=True)

    return ensure_datetime(combined)

# ------------------------------
# Data upload
# ------------------------------
st.subheader("1) Upload historical data")

hist_file = st.file_uploader(
    "Historical CSV (must include Date, Actual Fed Funds Rate, Unemployment, and at least one of Headline inflation/Core CPI/Core PCE)",
    type=["csv"], key="hist"
)
if hist_file is None:
    st.info("Please upload a historical CSV to proceed.")
    st.stop()

df_raw = pd.read_csv(hist_file)
df = normalize_columns(df_raw)
df = ensure_datetime(df)

# Validate
missing_core = [c for c in ["date", "fed_funds_actual", "unemployment"] if c not in df.columns]
if missing_core:
    st.error(f"CSV missing required columns: {missing_core}")
    st.stop()

available_infl_cols = [c for c in ["headline_inflation", "core_cpi", "core_pce"] if c in df.columns]
if not available_infl_cols:
    st.error("CSV must include at least one of: Headline inflation, Core CPI, or Core PCE.")
    st.stop()

# ------------------------------
# Inflation choice
# ------------------------------
st.subheader("2) Choose inflation series")
col1, col2 = st.columns([2, 1])

with col1:
    infl_choice = st.selectbox(
        "Inflation series for the model",
        options=available_infl_cols,
        format_func=lambda x: LABEL_MAP.get(x, x),
    )

# Build inflation_used for historical sample
df["inflation_used"] = df[infl_choice]

with col2:
    st.metric("Series chosen", LABEL_MAP.get(infl_choice, infl_choice))

# ------------------------------
# Parameters
# ------------------------------
st.subheader("3) Model parameters")

with st.expander("Taylor rule parameters", expanded=True):
    st.latex(r"i_t \;=\; \rho \, i_{t-L} \;+\; (1-\rho)\,\Big(r^{\ast} + \pi^{\ast} + a(\pi_t-\pi^{\ast}) + b(u_t-u^{\ast})\Big)")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        r_star = st.number_input("r* (neutral real rate, %)", value=0.5, step=0.1, format="%.2f")
        pi_star = st.number_input("π* (inflation target, %)", value=2.0, step=0.1, format="%.2f")
    with colp2:
        a = st.slider("Inflation gap coefficient", min_value=0.0, max_value=3.0, value=1.0, step=0.05)
        b = st.slider("Unemployment gap coefficient", min_value=-3.0, max_value=0.0, value=-0.5, step=0.05)
    with colp3:
        rho = st.slider("Smoothing (ρ)", min_value=0.0, max_value=0.99, value=0.7, step=0.01)
        L = st.number_input("Smoothing lag (L, in months)", value=3, step=1, min_value=1)

u_star = st.number_input("Natural unemployment (u*, %)", value=4.0, step=0.1, format="%.2f")

# ------------------------------
# Optional forecast upload
# ------------------------------
st.subheader("4) (Optional) Upload forecast paths for inflation and unemployment")
st.caption("If provided, the app will extend the series and compute a **forecasted** fed funds path.")

fcast_file = st.file_uploader(
    "Forecast CSV (must include Date, Unemployment, and at least one of Headline inflation/Core CPI/Core PCE). Only dates after the last historical date are used.",
    type=["csv"], key="fcast"
)

infl_choice_fcast = None
if fcast_file is not None:
    df_fcast_raw = pd.read_csv(fcast_file)
    df_fcast_norm = normalize_columns(df_fcast_raw)
    available_infl_fcast = [c for c in ["headline_inflation", "core_cpi", "core_pce"] if c in df_fcast_norm.columns]
    if available_infl_fcast:
        infl_choice_fcast = st.selectbox(
            "Inflation series to use from the **forecast** file",
            options=available_infl_fcast,
            format_func=lambda x: LABEL_MAP.get(x, x),
            index=available_infl_fcast.index(infl_choice) if infl_choice in available_infl_fcast else 0
        )
    else:
        st.warning("No inflation series found in the forecast file; it must include Headline inflation, Core CPI, or Core PCE.")

# Build combined dataframe (historical + forecast future)
df_combined = df[["date", "inflation_used", "unemployment", "fed_funds_actual"]].copy()
if fcast_file is not None and infl_choice_fcast is not None:
    df_combined = merge_with_forecast(df_combined, pd.read_csv(fcast_file), infl_choice, infl_choice_fcast)

# ------------------------------
# Compute modeled paths
# ------------------------------
modeled = compute_taylor_path(
    df_combined, a=a, b=b, r_star=r_star, pi_star=pi_star,
    rho=rho, smoothing_lag=int(L),
    use_actual_for_init=True
)

df_combined["fed_funds_modeled"] = modeled

# Split back into hist vs forecast windows for plotting
last_hist_date = df["date"].max()
is_future = df_combined["date"] > last_hist_date
df_hist = df_combined[~is_future].copy()
df_future = df_combined[is_future].copy()

# ------------------------------
# MSE on overlapping historical period
# ------------------------------
mse = np.nan
if "fed_funds_actual" in df_hist.columns:
    mask = (~df_hist["fed_funds_actual"].isna()) & (~df_hist["fed_funds_modeled"].isna())
    if mask.any():
        diffsq = (df_hist.loc[mask, "fed_funds_modeled"] - df_hist.loc[mask, "fed_funds_actual"]) ** 2
        mse = diffsq.mean()

st.subheader("5) Results")

left, right = st.columns([2.2, 1.0])

with left:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot historical actual and modeled
    if "fed_funds_actual" in df_hist.columns:
        ax.plot(df_hist["date"], df_hist["fed_funds_actual"], label="Fed funds (actual)")
    ax.plot(df_hist["date"], df_hist["fed_funds_modeled"], label="Fed funds (modeled)")

    # Forecast modeled path as dashed orange
    if not df_future.empty:
        ax.plot(df_future["date"], df_future["fed_funds_modeled"], linestyle="--", label="Forecast (modeled)")

    # Zero lower bound guide
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_title("Fed Funds Rate: Actual vs. Taylor Rule (with smoothing)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    if not np.isnan(mse):
        st.info(f"Mean Squared Error (historical overlap): **{mse:.3f}**")

with right:
    st.markdown("**Data preview (first 12 rows)**")
    preview_cols = ["date", "fed_funds_actual", "inflation_used", "unemployment", "fed_funds_modeled"]
    show_cols = [c for c in preview_cols if c in df_combined.columns]
    st.dataframe(df_combined[show_cols].head(12), hide_index=True)

# ------------------------------
# Download modeled data
# ------------------------------
st.subheader("6) Download modeled series")
csv_bytes = df_combined.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV (historical + modeled + forecast)",
    data=csv_bytes,
    file_name="taylor_rule_modeled.csv",
    mime="text/csv",
)

st.caption("Tip: If your CSV headers differ, the app tries to normalize common labels (e.g., 'Actual Fed Funds Rate' → 'fed_funds_actual').")
