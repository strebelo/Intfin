# forecast with RER.py
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def compute_weights(n_obs: int, lambda_decay: float) -> np.ndarray:
    """
    Exponential-decay weights for an expanding window sample of length n_obs.
    Last observation has weight 1, first has lambda_decay^(n_obs-1).
    """
    idx = np.arange(n_obs)
    age = (n_obs - 1) - idx
    w = lambda_decay ** age
    return w


def estimate_ar1(series: np.ndarray, weights: np.ndarray | None = None):
    """
    Estimate AR(1): y_t = alpha + rho * y_{t-1} + eps_t
    using OLS or weighted least squares.
    series: 1D array y_0, ..., y_T
    """
    y = np.asarray(series)
    if len(y) < 3:
        return np.nan, np.nan  # not enough data

    y_dep = y[1:]        # y_1 ... y_T
    x_reg = y[:-1]       # y_0 ... y_{T-1}
    X = np.column_stack([np.ones_like(x_reg), x_reg])

    if weights is not None:
        w = np.asarray(weights)[1:]  # align with y_dep
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y_dep * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(X, y_dep, rcond=None)

    alpha, rho = beta
    return alpha, rho


def forecast_ar1(alpha: float, rho: float, last_y: float, h: int) -> float:
    """
    h-step-ahead forecast from AR(1): y_{t+1} = alpha + rho*y_t.
    """
    y_fore = last_y
    for _ in range(h):
        y_fore = alpha + rho * y_fore
    return y_fore


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.title("Real Exchange Rate AR(1) Forecasting vs Random Walk")

st.markdown(
    """
This app:

1. Builds AR(1) models for  
   \\(x_t = \\log(RER_t)\\) and \\(p_t = \\log(P_t^*/P_t)\\) from your data.
2. Uses those to forecast the spot rate  
   \\(\\hat S_{t+h} = \\exp(\\hat p_{t+h} - \\hat x_{t+h})\\).
3. Compares these forecasts to a **random walk** out-of-sample at horizons  
   1, 6, 12, 24, 60, 120 months.
4. Produces **monthly forecasts for the next T years** from the last observation.

**Excel input format (first 5 columns):**

1. Date  
2. Spot exchange rate \\(S_t\\) (USD / foreign currency)  
3. Real exchange rate \\(RER_t\\)  
4. U.S. CPI (\\(P_t\\))  
5. Foreign CPI (\\(P_t^*\\))
"""
)

uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)

    if df_raw.shape[1] < 5:
        st.error("The file must have at least 5 columns (date, spot, RER, US CPI, foreign CPI).")
        st.stop()

    date_col = df_raw.columns[0]
    spot_col = df_raw.columns[1]
    rer_col = df_raw.columns[2]
    cpi_us_col = df_raw.columns[3]
    cpi_for_col = df_raw.columns[4]

    st.write("Using the following columns:")
    st.write(f"- Date: **{date_col}**")
    st.write(f"- Spot: **{spot_col}** (USD/foreign)")
    st.write(f"- RER: **{rer_col}**")
    st.write(f"- CPI US: **{cpi_us_col}**")
    st.write(f"- CPI Foreign: **{cpi_for_col}**")

    df = df_raw.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)

    # Build x_t and p_t
    df["x"] = np.log(df[rer_col])
    df["p"] = np.log(df[cpi_for_col] / df[cpi_us_col])

    # Clean up any invalid entries
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[spot_col, "x", "p"])

    st.subheader("Preview of processed data")
    st.dataframe(df[[date_col, spot_col, rer_col, cpi_us_col, cpi_for_col, "x", "p"]].head())

    # User choices for estimation and forecasts
    st.sidebar.header("Estimation Options")

    weighting_option = st.sidebar.radio(
        "AR(1) estimation weighting",
        ["Equal weights (OLS)", "Exponential decay (recent data gets more weight)"],
    )

    half_life_years = None
    lambda_decay = None
    if weighting_option == "Exponential decay (recent data gets more weight)":
        half_life_years = st.sidebar.slider(
            "Half-life of weights (years)",
            min_value=0.5,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="After this many years, the weight of an observation is half of the weight of the most recent observation."
        )

    min_window_years = st.sidebar.number_input(
        "Minimum estimation window (years)",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="AR(1) estimation begins only after this many years of data."
    )

    T_years_ahead = st.sidebar.number_input(
        "Years ahead T for monthly forecasts from the last observation",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
    )

    horizons = [1, 6, 12, 24, 60, 120]  # months

    run_button = st.button("Run estimation, evaluation, and produce forecasts")

    if run_button:
        # Basic arrays
        dates = df[date_col].values
        S = df[spot_col].values.astype(float)
        x = df["x"].values.astype(float)
        p = df["p"].values.astype(float)
        N = len(df)

        if N < 20:
            st.error("Not enough observations. Please provide a longer series.")
            st.stop()

        min_window_months = int(round(min_window_years * 12))
        if min_window_months < 3:
            min_window_months = 3  # at least a bit of data

        # Decay parameter if needed
        if weighting_option == "Exponential decay (recent data gets more weight)":
            half_life_months = max(int(round(half_life_years * 12)), 1)
            lambda_decay = 0.5 ** (1.0 / half_life_months)

        max_horizon = max(horizons)

        if N <= min_window_months + 1:
            st.error("Not enough data after applying the minimum estimation window.")
            st.stop()

        # --------------------------------------------------------
        # Out-of-sample evaluation: AR-based vs Random Walk
        # --------------------------------------------------------
        errs_ar = {h: [] for h in horizons}
        errs_rw = {h: [] for h in horizons}

        # t is the forecast origin index
        # We'll estimate AR(1) using data from index 0..t (expanding window)
        # and forecast at t+h for various h.
        for t in range(min_window_months, N - 1):
            # Available sample for estimation: 0..t
            x_sample = x[:t + 1]
            p_sample = p[:t + 1]

            if weighting_option == "Equal weights (OLS)":
                w_sample = None
            else:
                w_sample = compute_weights(len(x_sample), lambda_decay)

            alpha_x, rho_x = estimate_ar1(x_sample, w_sample)
            alpha_p, rho_p = estimate_ar1(p_sample, w_sample)

            # Skip if estimation failed (unlikely)
            if np.isnan(alpha_x) or np.isnan(rho_x) or np.isnan(alpha_p) or np.isnan(rho_p):
                continue

            S_t = S[t]
            x_t = x[t]
            p_t = p[t]

            for h in horizons:
                target_idx = t + h
                if target_idx >= N:
                    continue

                # AR-based forecast for x_{t+h} and p_{t+h}
                x_hat = forecast_ar1(alpha_x, rho_x, x_t, h)
                p_hat = forecast_ar1(alpha_p, rho_p, p_t, h)
                S_hat_ar = np.exp(p_hat - x_hat)

                S_true = S[target_idx]

                # Random walk forecast: S_hat = S_t
                S_hat_rw = S_t

                errs_ar[h].append(S_true - S_hat_ar)
                errs_rw[h].append(S_true - S_hat_rw)

        # Compute RMSEs
        rows = []
        for h in horizons:
            e_ar = np.array(errs_ar[h])
            e_rw = np.array(errs_rw[h])
            if len(e_ar) == 0:
                continue

            rmse_ar = np.sqrt(np.mean(e_ar ** 2))
            rmse_rw = np.sqrt(np.mean(e_rw ** 2))
            ratio = rmse_ar / rmse_rw if rmse_rw > 0 else np.nan
            rows.append(
                {
                    "Horizon (months)": h,
                    "Number of forecasts": len(e_ar),
                    "RMSE AR-based": rmse_ar,
                    "RMSE Random walk": rmse_rw,
                    "AR beats RW? (RMSE)": rmse_ar < rmse_rw,
                    "RMSE ratio (AR/RW)": ratio,
                }
            )

        if len(rows) == 0:
            st.warning("Not enough data to compute out-of-sample forecasts for the chosen horizons.")
        else:
            st.subheader("Out-of-sample RMSE: AR-based forecasts vs Random Walk")
            df_rmse = pd.DataFrame(rows)
            st.dataframe(df_rmse.style.format(
                {
                    "RMSE AR-based": "{:.6f}",
                    "RMSE Random walk": "{:.6f}",
                    "RMSE ratio (AR/RW)": "{:.3f}",
                }
            ))

        # --------------------------------------------------------
        # Forecasts for the next T years (monthly) from last obs
        # --------------------------------------------------------
        st.subheader("Monthly forecasts for the next T years from the last observation")

        # Estimate AR(1) on the full sample (under chosen weighting scheme)
        if weighting_option == "Equal weights (OLS)":
            w_full = None
        else:
            w_full = compute_weights(N, lambda_decay)

        alpha_x_full, rho_x_full = estimate_ar1(x, w_full)
        alpha_p_full, rho_p_full = estimate_ar1(p, w_full)

        if np.isnan(alpha_x_full) or np.isnan(rho_x_full) or np.isnan(alpha_p_full) or np.isnan(rho_p_full):
            st.error("AR(1) estimation on the full sample failed.")
            st.stop()

        H_future = int(round(T_years_ahead * 12))

        last_date = df[date_col].iloc[-1]
        last_x = x[-1]
        last_p = p[-1]

        future_dates = []
        future_x = []
        future_p = []
        future_S = []

        x_curr = last_x
        p_curr = last_p

        for h in range(1, H_future + 1):
            x_curr = alpha_x_full + rho_x_full * x_curr
            p_curr = alpha_p_full + rho_p_full * p_curr
            S_curr = np.exp(p_curr - x_curr)

            future_dates.append(last_date + pd.DateOffset(months=h))
            future_x.append(x_curr)
            future_p.append(p_curr)
            future_S.append(S_curr)

        df_forecast = pd.DataFrame(
            {
                "Date": future_dates,
                "x_forecast (log RER)": future_x,
                "p_forecast (log P*/P)": future_p,
                "Spot_forecast (S_hat_t)": future_S,
            }
        )

        st.write(f"Forecasts for the next **{T_years_ahead}** years ({H_future} months):")
        st.dataframe(df_forecast.head(24))  # show first 2 years as preview

        csv_bytes = df_forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full forecast as CSV",
            data=csv_bytes,
            file_name="spot_forecasts_next_T_years.csv",
            mime="text/csv",
        )

        st.markdown(
            """
**Notes:**

- AR(1) is estimated recursively with an expanding window for the out-of-sample comparison.  
- Random walk forecast is \\( \\hat S_{t+h}^{RW} = S_t \\).  
- Forecasts for the next T years use AR(1) estimated on the full sample (with the chosen weighting scheme).  
- Exponential decay uses a half-life: after the chosen half-life length, an observation has half the weight of the most recent one.
"""
        )
