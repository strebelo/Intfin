# fx_rer_arma_forecast.py
# Streamlit app: ARMA on log RER with recursive OOS nominal FX forecasts (multi-horizon)
#
# Run: streamlit run fx_rer_arma_forecast.py
# Requirements: streamlit, numpy, pandas, matplotlib, statsmodels, scipy

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(page_title="ARMA on RER → Nominal FX OOS Forecasts (Multi-horizon)", layout="wide")

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

    date_col = inv_map.get("date", None)
    rer_col = inv_map.get("rer", None) or inv_map.get("realexchangerate", None)
    spot_col = inv_map.get("spot", None) or inv_map.get("nominalspot", None) or inv_map.get("exchangerate", None)
    infl_col = inv_map.get("inflationdiff", None) or inv_map.get("inflationdifferential", None) \
               or inv_map.get("pidomminusfor", None) or inv_map.get("inflation", None)

    # Fallbacks
    if date_col is None:
        for col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if dt.notna().mean() > 0.8:
                date_col = col
                break
    if rer_col is None:
        for col in df.columns:
            if col != date_col and pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.9:
                rer_col = col
                break
    return date_col, rer_col, spot_col, infl_col

def ensure_clean_timeseries(df, date_col, rer_col, spot_col, infl_col):
    df = df[[date_col, rer_col, spot_col, infl_col]].copy()
    df[date_col] = coerce_datetime(df[date_col])
    for c in [rer_col, spot_col, infl_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[date_col, rer_col, spot_col, infl_col]).sort_values(date_col)
    df = df.set_index(date_col)
    df = df[~df.index.duplicated(keep="last")]
    return df

def fit_arma_log_rer(y_log, p, q):
    model = SARIMAX(y_log, order=(p, 0, q), trend='c',
                    enforce_stationarity=True, enforce_invertibility=True)
    res = model.fit(disp=False)
    return res

def ljung_box_summary(residuals, lags=12):
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    stat = float(lb['lb_stat'].iloc[0])
    pval = float(lb['lb_pvalue'].iloc[0])
    return stat, pval

def recursive_oos_one_step(y_rer, s_nom, infl_diff, p, q, train_frac, infl_is_percent=False):
    """
    Expanding-window OOS 1-step forecasts (as before).
    Returns oos DataFrame and metrics.
    """
    df = pd.DataFrame({"RER": y_rer, "S": s_nom, "INF_DIFF": infl_diff}).dropna().copy()
    df["logRER"] = np.log(df["RER"])
    df["logS"]   = np.log(df["S"])
    if infl_is_percent:
        df["INF_DIFF"] = df["INF_DIFF"] / 100.0

    n = len(df)
    if n < 30:
        raise ValueError("Not enough data after alignment; need at least ~30 observations.")

    train_n = max(20, int(np.floor(train_frac * n)))
    if train_n >= n - 1:
        train_n = n - 2

    oos_records = []

    for t in range(train_n, n - 1):
        est = df.iloc[:t+1]
        y_log = est["logRER"]

        try:
            res = fit_arma_log_rer(y_log, p, q)
        except Exception:
            model = SARIMAX(y_log, order=(p,0,q), trend='c',
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)

        q_t = df["logRER"].iloc[t]
        s_t = df["logS"].iloc[t]
        infl_next = df["INF_DIFF"].iloc[t+1]

        q_hat_next = float(res.get_forecast(steps=1).predicted_mean.iloc[0])
        dq_hat = q_hat_next - q_t
        s_hat_next = s_t + dq_hat + infl_next
        s_next_actual = df["logS"].iloc[t+1]

        oos_records.append({
            "date_forecast_for": df.index[t+1],
            "train_end": df.index[t],
            "q_log_hat_next": q_hat_next,
            "dq_log_hat": dq_hat,
            "infl_diff_next": infl_next,
            "s_log_hat_next": s_hat_next,
            "s_log_actual_next": s_next_actual,
            "s_log_error_ARMA": s_next_actual - s_hat_next,
            "s_log_hat_rw": s_t,
            "s_log_error_RW": s_next_actual - s_t
        })

    oos = pd.DataFrame(oos_records).set_index("date_forecast_for")
    arma_mse = float(np.mean(oos["s_log_error_ARMA"]**2))
    rw_mse   = float(np.mean(oos["s_log_error_RW"]**2))
    metrics = {
        "OOS log-MSE (Nominal, ARMA-implied)": arma_mse,
        "OOS log-MSE (Nominal, Random Walk)": rw_mse,
        "ARMA better (lower MSE)?": arma_mse < rw_mse
    }
    return oos, metrics, train_n, df

def recursive_oos_multi_h(df_aligned, p, q, train_frac, horizons):
    """
    Multi-horizon expanding-window OOS forecasts on nominal FX in logs.
    df_aligned must contain columns: logRER, logS, INF_DIFF. Index must be time.
    Returns:
      summary_df (per-horizon MSEs),
      per_origin_df (long-table of forecasts & errors for each origin/horizon).
    """
    df = df_aligned.copy()
    n = len(df)
    train_n = max(20, int(np.floor(train_frac * n)))
    train_n = min(train_n, n - 2)

    # Collect per-origin/horizon records
    recs = []

    for t in range(train_n, n - 1):
        # Fit up to t
        y_log = df["logRER"].iloc[:t+1]
        try:
            res = fit_arma_log_rer(y_log, p, q)
        except Exception:
            model = SARIMAX(y_log, order=(p,0,q), trend='c',
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)

        q_t = df["logRER"].iloc[t]
        s_t = df["logS"].iloc[t]

        for h in horizons:
            if t + h >= n:
                continue  # not enough future data to score

            # Forecast logRER to t+h
            q_hat_path = res.get_forecast(steps=h).predicted_mean
            q_hat_th = float(q_hat_path.iloc[-1])  # q_{t+h|t}
            dq_hat_th = q_hat_th - q_t

            # Cumulate inflation differential from t+1...t+h (assumed given)
            infl_cum = float(df["INF_DIFF"].iloc[t+1:t+1+h].sum())

            # Nominal log forecast and actual
            s_hat_th = s_t + dq_hat_th + infl_cum
            s_act_th = float(df["logS"].iloc[t+h])

            # Random walk(log) forecast for horizon h is just s_t
            s_rw_th = s_t

            recs.append({
                "origin": df.index[t],
                "target": df.index[t+h],
                "horizon": h,
                "s_log_hat_ARMA": s_hat_th,
                "s_log_hat_RW": s_rw_th,
                "s_log_actual": s_act_th,
                "err_ARMA": s_act_th - s_hat_th,
                "err_RW": s_act_th - s_rw_th
            })

    long_df = pd.DataFrame(recs)
    if long_df.empty:
        raise ValueError("No valid OOS pairs for the chosen train split and horizons (data too short).")

    # MSE per horizon
    mse = long_df.groupby("horizon")[["err_ARMA", "err_RW"]].apply(lambda g: pd.Series({
        "logMSE_ARMA": np.mean(g["err_ARMA"]**2),
        "logMSE_RW":   np.mean(g["err_RW"]**2),
        "N_evals":     len(g)
    })).reset_index()

    mse["ARMA_better"] = mse["logMSE_ARMA"] < mse["logMSE_RW"]
    return mse, long_df

# -------------------------------
# UI
# -------------------------------

st.title("ARMA on log RER → Nominal FX Recursive OOS Forecasts (incl. Multi-horizon)")

st.markdown(
    """
Upload **monthly** (or regular-frequency) data with columns:
- **Date**, **RER** (level), **Nominal Spot** (level), **Inflation Differential** (dom − for).

We estimate **ARMA(p, q)** on **log RER**, produce **recursive expanding-window OOS** forecasts,
map them into **log nominal FX** using  
\\(\\Delta s_{t\\to t+h} \\approx \\Delta q_{t\\to t+h} + \\sum_{k=1}^h (\\pi_d-\\pi_f)_{t+k}\\),
and compare against a **random walk in logs** by **log-MSE**.
"""
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
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

date_guess, rer_guess, spot_guess, infl_guess = guess_columns(raw)

st.subheader("Select columns")
c1, c2, c3, c4 = st.columns(4)
with c1:
    date_col = st.selectbox("Date column", options=list(raw.columns),
                            index=(list(raw.columns).index(date_guess) if date_guess in raw.columns else 0))
with c2:
    rer_col = st.selectbox("RER (level) column", options=list(raw.columns),
                           index=(list(raw.columns).index(rer_guess) if rer_guess in raw.columns else 1))
with c3:
    spot_col = st.selectbox("Nominal Spot (level) column", options=list(raw.columns),
                            index=(list(raw.columns).index(spot_guess) if spot_guess in raw.columns else 2))
with c4:
    infl_col = st.selectbox("Inflation differential column", options=list(raw.columns),
                            index=(list(raw.columns).index(infl_guess) if infl_guess in raw.columns else 3))

try:
    panel = ensure_clean_timeseries(raw, date_col, rer_col, spot_col, infl_col)
except Exception as e:
    st.error(f"Problem parsing and cleaning the data: {e}")
    st.stop()

# Prepare aligned/log panel for re-use
panel_aligned = panel.copy()
panel_aligned["logRER"] = np.log(panel_aligned[rer_col])
panel_aligned["logS"] = np.log(panel_aligned[spot_col])

st.subheader("Model configuration")
mcol1, mcol2, mcol3 = st.columns([1,1,2])
with mcol1:
    p = st.number_input("AR order p", min_value=0, max_value=5, value=1, step=1)
with mcol2:
    q = st.number_input("MA order q", min_value=0, max_value=5, value=1, step=1)
with mcol3:
    train_frac = st.slider("Training fraction for initial fit (expanding window thereafter)", 0.4, 0.9, 0.66, 0.01)

infl_is_percent = st.checkbox("Inflation differential is in **percent** (e.g., 2 = 2%)", value=False)
if infl_is_percent:
    panel_aligned["INF_DIFF"] = panel_aligned[infl_col] / 100.0
else:
    panel_aligned["INF_DIFF"] = panel_aligned[infl_col]

# Initial diagnostics on the first training window
try:
    n = len(panel_aligned)
    init_n = max(20, int(np.floor(train_frac * n)))
    if init_n >= n:
        init_n = n - 1
    res0 = fit_arma_log_rer(panel_aligned["logRER"].iloc[:init_n], p, q)
except Exception as e:
    st.error(f"Initial ARMA fit failed: {e}")
    st.stop()

st.subheader("Diagnostics (initial training window)")
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("AIC", f"{res0.aic:.2f}")
with d2:
    st.metric("BIC", f"{res0.bic:.2f}")
with d3:
    resid_mean = float(np.mean(res0.resid))
    st.metric("Residual mean", f"{resid_mean:.4g}")
with d4:
    resid_std = float(np.std(res0.resid, ddof=1))
    st.metric("Residual std", f"{resid_std:.4g}")

try:
    lb_stat, lb_p = ljung_box_summary(res0.resid, lags=12)
    jb_stat, jb_p = stats.jarque_bera(res0.resid)
    st.write(f"**Ljung–Box (lag 12)**: stat = {lb_stat:.3f}, p = {lb_p:.3g}")
    st.write(f"**Jarque–Bera normality**: stat = {jb_stat:.3f}, p = {jb_p:.3g}")
except Exception:
    st.write("Diagnostics could not be computed (insufficient data).")

with st.expander("Residual plots (initial window)"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(res0.resid, lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Residuals")
    ax.grid(True, linestyle=":", linewidth=0.8)
    st.pyplot(fig)

    fig_acf, ax_acf = plt.subplots(figsize=(7, 2.8))
    plot_acf(res0.resid, lags=24, ax=ax_acf)
    ax_acf.set_title("ACF of residuals")
    st.pyplot(fig_acf)

    fig_pacf, ax_pacf = plt.subplots(figsize=(7, 2.8))
    plot_pacf(res0.resid, lags=24, ax=ax_pacf, method="ywm")
    ax_pacf.set_title("PACF of residuals")
    st.pyplot(fig_pacf)

# -------------------------------
# Recursive OOS (1-step) as before
# -------------------------------
st.subheader("Recursive OOS: one-step forecasts & evaluation")
try:
    oos1, metrics1, train_n, df_aligned = recursive_oos_one_step(
        y_rer=panel[rer_col],
        s_nom=panel[spot_col],
        infl_diff=panel[infl_col],
        p=int(p), q=int(q),
        train_frac=float(train_frac),
        infl_is_percent=infl_is_percent
    )
except Exception as e:
    st.error(f"Recursive 1-step OOS failed: {e}")
    st.stop()

cA, cB, cC = st.columns(3)
with cA:
    st.metric("OOS log-MSE (Nominal, ARMA-implied)", f"{metrics1['OOS log-MSE (Nominal, ARMA-implied)']:.6f}")
with cB:
    st.metric("OOS log-MSE (Nominal, RW)", f"{metrics1['OOS log-MSE (Nominal, Random Walk)']:.6f}")
with cC:
    st.metric("ARMA better (1-step)?", "Yes" if metrics1["ARMA better (lower MSE)?"] else "No")

st.caption(f"Initial training observations: **{train_n}** of {len(df_aligned)}. Then expanding-window 1-step OOS.")

st.dataframe(
    oos1.assign(
        s_log_hat_next=lambda d: d["s_log_hat_next"].round(6),
        s_log_actual_next=lambda d: d["s_log_actual_next"].round(6),
        s_log_error_ARMA=lambda d: d["s_log_error_ARMA"].round(6),
        s_log_error_RW=lambda d: d["s_log_error_RW"].round(6),
        dq_log_hat=lambda d: d["dq_log_hat"].round(6),
        infl_diff_next=lambda d: d["infl_diff_next"].round(6),
    ),
    use_container_width=True
)

# -------------------------------
# NEW: Multi-horizon OOS (1,6,12,24,36,60,120 months)
# -------------------------------
st.subheader("Recursive OOS: multi-horizon forecasts vs Random Walk")

default_horizons = [1, 6, 12, 24, 36, 60, 120]
st.write("Fixed horizons (months):", default_horizons)

try:
    mse_by_h, long_table = recursive_oos_multi_h(
        df_aligned=df_aligned,  # from the 1-step function (already logs & INF_DIFF aligned)
        p=int(p), q=int(q),
        train_frac=float(train_frac),
        horizons=default_horizons
    )
except Exception as e:
    st.error(f"Multi-horizon OOS failed: {e}")
    st.stop()

# Summary table by horizon
st.markdown("**Log-MSE by horizon (lower is better)**")
st.dataframe(
    mse_by_h.rename(columns={
        "horizon": "H (months)",
        "logMSE_ARMA": "log-MSE (ARMA→Nominal)",
        "logMSE_RW": "log-MSE (RW in logs)",
        "N_evals": "#OOS evals",
        "ARMA_better": "ARMA better?"
    }).assign(**{
        "log-MSE (ARMA→Nominal)": lambda d: d["log-MSE (ARMA→Nominal)"].round(6),
        "log-MSE (RW in logs)": lambda d: d["log-MSE (RW in logs)"].round(6)
    }),
    use_container_width=True
)

# Plot: MSE by horizon
with st.expander("Plot: OOS log-MSE by horizon"):
    figh, axh = plt.subplots(figsize=(8, 3.8))
    axh.plot(mse_by_h["horizon"], mse_by_h["logMSE_ARMA"], marker="o", label="ARMA→Nominal")
    axh.plot(mse_by_h["horizon"], mse_by_h["logMSE_RW"], marker="o", linestyle="--", label="RW (logs)")
    axh.set_xscale("log")
    axh.set_xticks(default_horizons)
    axh.get_xaxis().set_major_formatter(plt.FixedFormatter([str(h) for h in default_horizons]))
    axh.set_xlabel("Horizon (months, log scale)")
    axh.set_ylabel("log-MSE")
    axh.set_title("OOS log-MSE by forecast horizon")
    axh.grid(True, linestyle=":", linewidth=0.8)
    axh.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize="small")
    axh.tick_params(axis="x", pad=8)
    figh.tight_layout()
    figh.subplots_adjust(bottom=0.28)
    st.pyplot(figh)

# Long table download (all origins × horizons)
st.download_button(
    "Download multi-horizon OOS (long table, CSV)",
    data=long_table.to_csv(index=False).encode("utf-8"),
    file_name="oos_multi_h_long.csv",
    mime="text/csv"
)

# Small preview of long table
with st.expander("Preview: long table (origins × horizons)"):
    st.dataframe(
        long_table.assign(
            s_log_hat_ARMA=lambda d: d["s_log_hat_ARMA"].round(6),
            s_log_hat_RW=lambda d: d["s_log_hat_RW"].round(6),
            s_log_actual=lambda d: d["s_log_actual"].round(6),
            err_ARMA=lambda d: d["err_ARMA"].round(6),
            err_RW=lambda d: d["err_RW"].round(6),
        ).head(200),
        use_container_width=True
    )

st.success("Multi-horizon OOS comparison complete — try different (p, q) and train splits to test robustness.")
