# forecast with RER.py
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------
# User settings
# ---------------------------

EXCEL_FILE = "your_data.xlsx"  # path to your Excel file

DATE_COL = 0      # column index of dates in the Excel file
SPOT_COL = 1      # column index of nominal spot exchange rate
RER_COL = 2       # column index of real exchange rate
CPI_DOM_COL = 3   # column index of domestic CPI
CPI_FOR_COL = 4   # column index of foreign CPI

MIN_WINDOW_MONTHS = 120   # minimum sample length before we start forecasting (e.g., 10 years)
CPI_WINDOW_MONTHS = 60    # last 5 years for CPI ratio AR(1)
MAX_H = 60                # maximum horizon (months) = 5 years
HORIZONS = [1, 3, 6, 12, 24, 36, 60]  # horizons to evaluate

# ---------------------------
# Helpers
# ---------------------------

def ar1_forecast_multi_step(y_hist, h):
    """
    Estimate AR(1) with constant: y_t = alpha + rho y_{t-1} + eps,
    using y_hist (1D array of length T). Then generate an h-step-ahead
    forecast starting from the last observation y_hist[-1], using only
    information in y_hist.

    Returns:
        y_hat (float): h-step-ahead forecast.
        alpha_hat, rho_hat (floats): estimated parameters.
    """
    y_hist = np.asarray(y_hist)
    if len(y_hist) < 3:
        # Too little data: fallback to random walk
        return y_hist[-1], 0.0, 1.0

    # Prepare regression: y_t vs [1, y_{t-1}]
    y = y_hist[1:]          # y_1 .. y_{T-1}
    x_lag = y_hist[:-1]     # y_0 .. y_{T-2}
    X = sm.add_constant(x_lag)
    model = sm.OLS(y, X).fit()
    alpha_hat, rho_hat = model.params

    # Multi-step forecast using direct iteration
    y_fore = y_hist[-1]
    for _ in range(h):
        y_fore = alpha_hat + rho_hat * y_fore

    return y_fore, alpha_hat, rho_hat


def compute_structural_forecast(df_log, origin_idx, h, cpi_window_months=60):
    """
    Compute s_{t+h|t}^{struct} = q_{t+h|t} + c_{t+h|t}, where:
    - q_t = log RER
    - c_t = log(CPI_dom / CPI_for)
    At origin index origin_idx, using only data up to that point (no look-ahead bias).

    For q_t: AR(1) with constant using full history up to origin_idx.
    For c_t: AR(1) with constant using last `cpi_window_months` data up to origin_idx.

    df_log has columns:
      's'  : log spot
      'q'  : log RER
      'c'  : log CPI ratio

    Returns:
        s_hat_struct (float): structural nominal forecast in logs.
    """
    # --- RER AR(1) on full history up to origin_idx ---
    q_hist = df_log["q"].iloc[:origin_idx + 1].values  # 0..origin_idx
    q_hat_h, _, _ = ar1_forecast_multi_step(q_hist, h)

    # --- CPI ratio AR(1) on last cpi_window_months up to origin_idx ---
    c_hist_full = df_log["c"].iloc[:origin_idx + 1].values
    if len(c_hist_full) > cpi_window_months:
        c_hist = c_hist_full[-cpi_window_months:]
    else:
        c_hist = c_hist_full

    c_hat_h, _, _ = ar1_forecast_multi_step(c_hist, h)

    # Structural nominal forecast
    s_hat_struct = q_hat_h + c_hat_h
    return s_hat_struct


# ---------------------------
# Main script
# ---------------------------

def main():
    # --- Load data ---
    df_raw = pd.read_excel(EXCEL_FILE)

    # Keep only required columns and rename
    df = df_raw.iloc[:, [DATE_COL, SPOT_COL, RER_COL, CPI_DOM_COL, CPI_FOR_COL]].copy()
    df.columns = ["date", "spot", "rer", "cpi_dom", "cpi_for"]

    # Parse dates, sort
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    # Drop rows with missing or non-positive values that break logs
    df = df.dropna()
    df = df[(df["spot"] > 0) & (df["rer"] > 0) & (df["cpi_dom"] > 0) & (df["cpi_for"] > 0)]

    n = len(df)
    print(f"Number of monthly observations after cleaning: {n}")

    # --- Build log variables ---
    df_log = pd.DataFrame(index=df.index)
    df_log["s"] = np.log(df["spot"].astype(float))                      # log spot
    df_log["q"] = np.log(df["rer"].astype(float))                       # log RER
    df_log["c"] = np.log(df["cpi_dom"].astype(float) /
                         df["cpi_for"].astype(float))                   # log CPI ratio

    # Sanity check
    print("\nLast few observations (logs):")
    print(df_log.tail())

    # --- Out-of-sample evaluation ---
    errors_struct = {h: [] for h in HORIZONS}
    errors_rw = {h: [] for h in HORIZONS}
    origins = []

    # We need enough history before starting recursive forecasting
    # Also keep at least MAX_H observations after origin for full horizon evaluation
    for origin_idx in range(MIN_WINDOW_MONTHS, n - 1):
        origins.append(origin_idx)
        s_t = df_log["s"].iloc[origin_idx]

        for h in HORIZONS:
            if origin_idx + h >= n:
                continue

            s_actual = df_log["s"].iloc[origin_idx + h]

            # Structural forecast: uses only data up to origin_idx
            s_hat_struct = compute_structural_forecast(df_log, origin_idx,
                                                       h, cpi_window_months=CPI_WINDOW_MONTHS)

            # Random walk forecast
            s_hat_rw = s_t

            errors_struct[h].append(s_actual - s_hat_struct)
            errors_rw[h].append(s_actual - s_hat_rw)

    if not origins:
        print("\nNot enough data to run out-of-sample evaluation. "
              "Increase sample or reduce MIN_WINDOW_MONTHS / MAX_H.")
        return

    # --- RMSFE summary ---
    results = []
    for h in HORIZONS:
        es = np.array(errors_struct[h], dtype=float)
        er = np.array(errors_rw[h], dtype=float)
        if len(es) == 0 or len(er) == 0:
            continue

        rmsfe_struct = np.sqrt(np.mean(es ** 2))
        rmsfe_rw = np.sqrt(np.mean(er ** 2))
        rel = rmsfe_struct / rmsfe_rw if rmsfe_rw > 0 else np.nan
        improvement = (1 - rel) * 100 if np.isfinite(rel) else np.nan

        results.append(
            {
                "Horizon (months)": h,
                "Obs used": len(es),
                "RMSFE - Structural": rmsfe_struct,
                "RMSFE - Random walk": rmsfe_rw,
                "Ratio (Struct/RW)": rel,
                "Improvement (%)": improvement,
            }
        )

    res_df = pd.DataFrame(results).set_index("Horizon (months)")
    print("\nOut-of-sample RMSFE comparison (logs):")
    print(res_df.to_string(float_format=lambda x: f"{x: .6f}"))

    # --- Example: forecast path from a chosen origin ---
    # Pick the last valid origin that allows a full 5-year forecast
    last_origin_idx = n - MAX_H - 1
    if last_origin_idx < MIN_WINDOW_MONTHS:
        print("\nNot enough room at the end of the sample for a full 5-year forecast path.")
        return

    origin_date = df_log.index[last_origin_idx]
    print(f"\nExample 5-year forecast path from origin: {origin_date.date()}")

    s0 = df_log["s"].iloc[last_origin_idx]
    horizons_seq = np.arange(0, MAX_H + 1)
    struct_path = []
    rw_path = []

    for h in horizons_seq:
        if h == 0:
            struct_path.append(s0)
            rw_path.append(s0)
        else:
            s_hat_struct = compute_structural_forecast(
                df_log, last_origin_idx, int(h), cpi_window_months=CPI_WINDOW_MONTHS
            )
            struct_path.append(s_hat_struct)
            rw_path.append(s0)

    # Convert back to levels for interpretability
    struct_path_levels = np.exp(struct_path)
    rw_path_levels = np.exp(rw_path)
    dates_forecast = pd.date_range(start=origin_date, periods=MAX_H + 1, freq="MS")

    fc_df = pd.DataFrame(
        {
            "spot_actual": df["spot"],
            "spot_struct_forecast": pd.Series(struct_path_levels, index=dates_forecast),
            "spot_rw_forecast": pd.Series(rw_path_levels, index=dates_forecast),
        }
    )

    print("\nForecast path (first few rows, levels):")
    print(fc_df[['spot_struct_forecast', 'spot_rw_forecast']].dropna().head(10))

    # If you want, save to Excel or CSV for plotting elsewhere
    fc_df.to_csv("fx_forecast_example_path.csv")
    print("\nSaved example forecast path to fx_forecast_example_path.csv")


if __name__ == "__main__":
    main()
