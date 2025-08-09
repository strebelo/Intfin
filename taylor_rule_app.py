
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Taylor Rule Explorer", layout="wide")

st.title("Taylor Rule Explorer")
st.write(
    "Upload a data file (CSV or Excel) that includes: dates, the **actual** Fed Funds Rate, "
    "three measures of inflation (headline, core CPI, core PCE), and the unemployment rate. "
    "Then select which columns correspond to each series, choose the inflation measure to use in the rule, "
    "set the parameters, and compare the rule-implied rate to the actual rate."
)

# --- File upload ---
uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

@st.cache_data
def load_data(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

if uploaded is not None:
    df_raw = load_data(uploaded)
    st.subheader("Preview of uploaded data")
    st.dataframe(df_raw.head(10))

    # --- Column mapping ---
    st.subheader("Map your columns")
    cols = list(df_raw.columns)

    date_col = st.selectbox("Date column", options=cols, index=0)
    fedfunds_col = st.selectbox("Actual Fed Funds Rate column", options=cols, index=min(1, len(cols)-1))
    headline_col = st.selectbox("Headline inflation column", options=cols, index=min(2, len(cols)-1))
    core_cpi_col = st.selectbox("Core CPI inflation column", options=cols, index=min(3, len(cols)-1))
    core_pce_col = st.selectbox("Core PCE inflation column", options=cols, index=min(4, len(cols)-1))
    unemp_col = st.selectbox("Unemployment rate column", options=cols, index=min(5, len(cols)-1))

    # --- Parse and clean ---
    df = df_raw.copy()
    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df = df[[date_col, fedfunds_col, headline_col, core_cpi_col, core_pce_col, unemp_col]].copy()
    df.columns = ["date", "fed_funds_actual", "pi_headline", "pi_core_cpi", "pi_core_pce", "u"]
    # Coerce numerics
    for c in ["fed_funds_actual", "pi_headline", "pi_core_cpi", "pi_core_pce", "u"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing date
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    st.write(f"Data spans **{df['date'].min().date()}** to **{df['date'].max().date()}** with {len(df)} rows.")

    # --- Inflation measure choice ---
    st.subheader("Choose inflation measure for πₜ")
    pi_choice = st.radio(
        "Inflation series to use in the Taylor rule",
        options=[("Headline inflation", "pi_headline"), ("Core CPI", "pi_core_cpi"), ("Core PCE", "pi_core_pce")],
        format_func=lambda x: x[0],
        horizontal=True,
    )
    pi_col = dict(pi_choice)["pi_headline" if isinstance(pi_choice, tuple) else pi_choice]

    # The radio returns tuple; handle robustly:
    if isinstance(pi_choice, tuple):
        pi_col = pi_choice[1]
    else:
        pi_col = pi_choice

    # --- Parameters ---
    st.subheader("Parameters")
    with st.expander("Set Taylor rule parameters", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            r_star = st.number_input("r* (neutral real rate, %)", value=0.5, step=0.1, format="%.2f")
            pi_star = st.number_input("π* (inflation target, %)", value=2.0, step=0.1, format="%.2f")
        with colB:
            u_star = st.number_input("u* (natural unemployment rate, %)", value=4.0, step=0.1, format="%.2f")
            a = st.number_input("a (inflation gap coefficient)", value=1.5, step=0.1, format="%.2f")
        with colC:
            b = st.number_input("b (unemployment gap coefficient)", value=1.0, step=0.1, format="%.2f")
            rho = st.slider("ρ (smoothing on FFₜ₋₃)", min_value=0.0, max_value=0.99, value=0.8, step=0.01)

        st.caption("Rule used: **FFₜ = ρ·FFₜ₋₃ + (1−ρ)·[ r* + π* + a·(πₜ − π*) + b·(uₜ − u*) ]**")

    # --- Compute model ---
    LAG = 3  # periods

    # Build the time series needed
    pi_t = df[pi_col].to_numpy(dtype=float)
    u_t = df["u"].to_numpy(dtype=float)
    ff_actual = df["fed_funds_actual"].to_numpy(dtype=float)

    ff_model = np.full_like(ff_actual, fill_value=np.nan, dtype=float)

    # We'll start computing once we have t-LAG available.
    # Seed the first LAG values with actual FF (common in empirical work to initialize the recursion).
    for t in range(min(LAG, len(ff_model))):
        ff_model[t] = np.nan  # leave as NaN so errors don't use these early periods

    for t in range(LAG, len(ff_model)):
        baseline = r_star + pi_star + a * (pi_t[t] - pi_star) + b * (u_t[t] - u_star)
        ff_model[t] = rho * ff_model[t - LAG] + (1 - rho) * baseline if not np.isnan(ff_model[t - LAG]) else (1 - rho) * baseline

    # --- Metrics ---
    # Compute on overlapping, valid observations
    valid = (~np.isnan(ff_model)) & (~np.isnan(ff_actual))
    n_valid = int(valid.sum())

    if n_valid >= 1:
        mse = float(np.nanmean((ff_actual[valid] - ff_model[valid]) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.nanmean(np.abs(ff_actual[valid] - ff_model[valid])))
    else:
        mse = rmse = mae = np.nan

    st.subheader("Fit metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations (used)", f"{n_valid}")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}" if np.isfinite(mse) else "n/a")
    col3.metric("Root MSE (RMSE)", f"{rmse:.3f}" if np.isfinite(rmse) else "n/a")
    col4.metric("Mean Absolute Error (MAE)", f"{mae:.3f}" if np.isfinite(mae) else "n/a")

    # --- Plot ---
    st.subheader("Actual vs Taylor Rule (modeled) Fed Funds Rate")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["fed_funds_actual"], label="Actual Fed Funds Rate")
    ax.plot(df["date"], ff_model, label="Taylor Rule (modeled)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.set_title("Fed Funds Rate: Actual vs Taylor Rule")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # --- Download modeled series ---
    out = df[["date"]].copy()
    out["fed_funds_actual"] = df["fed_funds_actual"]
    out["fed_funds_modeled"] = ff_model
    out["inflation_used"] = df[pi_col]
    out["unemployment"] = df["u"]

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download modeled series (CSV)",
        data=csv_bytes,
        file_name="taylor_rule_modeled_series.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV/XLSX to get started. Include columns for date, actual Fed Funds, headline/core CPI/core PCE inflation, and unemployment.")
