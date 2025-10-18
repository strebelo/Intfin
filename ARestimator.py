# ARestimator.py
# AR(p) model picker for log(x) — Streamlit App
# -------------------------------------------------
# How to run locally:
# 1) Install deps:    pip install streamlit pandas numpy statsmodels plotly openpyxl
# 2) Launch:          streamlit run app.py
#
# Excel format expected:
# - Column 1: dates or datetimes (sorted ascending; no duplicates)
# - Column 2: x (strictly positive; the app models ln(x))
#
# The app:
# - Loads & validates data
# - Creates y = ln(x)
# - Fits AR(p) for p=1..Pmax with statsmodels AutoReg
# - Suggests p by IC (BIC by default)
# - Shows fit summary and residual diagnostics
# - Offers simple out-of-sample forecasts

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.stattools import durbin_watson
import streamlit as st

st.set_page_config(page_title="AR(p) on ln(x)", layout="wide")

st.title("Estimate AR(p) on ln(x) & suggest p")
st.caption("Upload an Excel file (dates in col 1, x in col 2). The app models **ln(x)**.")

with st.sidebar:
    st.header("Settings")
    p_max = st.number_input("Max lag Pmax", min_value=1, max_value=72, value=12, step=1, help="Upper bound for search over p=1..Pmax.")
    ic_choice = st.selectbox("Information criterion", ["BIC", "AIC", "HQIC"], index=0, help="p with the smallest IC is suggested.")
    include_const = st.checkbox("Include constant (trend='c')", value=True, help="Unchecked sets trend='n' (no constant).")
    robust_se = st.checkbox("Use heteroskedasticity-robust SE (HC0)", value=False)
    forecast_h = st.number_input("Forecast horizon (steps ahead)", min_value=0, max_value=120, value=0, step=1)
    st.markdown("---")
    st.markdown("**Tip:** Sort your dates ascending and ensure x>0 (required for ln).")

uploaded = st.file_uploader("Upload Excel (.xlsx, .xls). Dates in first column; x in second.", type=["xlsx","xls"])

def _read_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, header=0)
    if df.shape[1] < 2:
        raise ValueError("Need at least two columns: [date, x].")
    # Keep first 2 columns only; rename
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "x"]
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].index[:5].tolist()
        raise ValueError(f"Some dates could not be parsed (rows: {bad}).")
    # Sort and drop dupes
    df = df.sort_values("date").drop_duplicates("date")
    # Coerce x numeric
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    # Drop missing
    df = df.dropna()
    return df.reset_index(drop=True)

def _fit_autoreg(y: pd.Series, p: int, trend: str, cov_type: str):
    model = AutoReg(y, lags=p, old_names=False, trend=trend)
    res = model.fit()
    # Optionally re-compute robust covariance
    if cov_type == "HC0":
        res = res.get_robustcov_results(cov_type="HC0")
    return res

def _collect_ic_table(y: pd.Series, pmax: int, trend: str, cov_type: str) -> pd.DataFrame:
    rows = []
    for p in range(1, pmax+1):
        try:
            res = _fit_autoreg(y, p, trend, cov_type)
            rows.append({
                "p": p,
                "AIC": res.aic,
                "BIC": res.bic,
                "HQIC": res.hqic,
                "nobs": int(res.nobs)
            })
        except Exception as e:
            rows.append({"p": p, "AIC": np.nan, "BIC": np.nan, "HQIC": np.nan, "nobs": np.nan})
    tab = pd.DataFrame(rows).set_index("p")
    return tab

def _diagnostics(residuals: pd.Series, lags: int = 12):
    # Ljung-Box for serial correlation
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    # Jarque-Bera normality
    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    # Durbin-Watson
    dw = durbin_watson(residuals)
    return lb, (jb_stat, jb_p, skew, kurt), dw

def _plot_series(df: pd.DataFrame):
    fig = px.line(df, x="date", y="x", title="x over time")
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

def _plot_log_series(df_log: pd.DataFrame):
    fig = px.line(df_log, x="date", y="ln_x", title="ln(x) over time")
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

def _plot_residuals(date_index, resid: pd.Series):
    fig = px.line(x=date_index, y=resid, title="Residuals (fitted model)")
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=10), xaxis_title="date", yaxis_title="residual")
    st.plotly_chart(fig, use_container_width=True)

def _plot_acf_pacf(series: pd.Series, nlags: int = 24, title_prefix: str = ""):
    acf_vals = sm.tsa.stattools.acf(series, nlags=nlags, fft=True)
    pacf_vals = sm.tsa.stattools.pacf(series, nlags=nlags, method="ywm")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig1.update_layout(title=f"{title_prefix}ACF", xaxis_title="lag", yaxis_title="acf", height=300, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals))
    fig2.update_layout(title=f"{title_prefix}PACF", xaxis_title="lag", yaxis_title="pacf", height=300, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig2, use_container_width=True)

def _forecast(res, steps: int):
    if steps <= 0:
        return None
    # For AutoReg, out-of-sample forecasts:
    start = len(res.model.endog)
    end = len(res.model.endog) + steps - 1
    fc = res.predict(start=start, end=end, dynamic=False)
    return pd.Series(fc)

if uploaded:
    try:
        df = _read_excel(uploaded)
    except Exception as e:
        st.error(f"File problem: {e}")
        st.stop()

    # Validate positivity for log
    if (df["x"] <= 0).any():
        st.error("x must be strictly positive to take logs. Please clean your file.")
        st.stop()

    # Show raw x
    st.subheader("Raw data")
    st.dataframe(df.head(10), use_container_width=True)
    _plot_series(df)

    # ln(x)
    df["ln_x"] = np.log(df["x"])
    st.subheader("Transformed series: ln(x)")
    st.dataframe(df[["date","ln_x"]].head(10), use_container_width=True)
    _plot_log_series(df)

    # Quick ACF/PACF of ln(x)
    st.markdown("**Correlograms for ln(x) (useful for ballpark p):**")
    _plot_acf_pacf(df["ln_x"], nlags=min(36, max(10, p_max+6)), title_prefix="ln(x) ")

    # Fit candidates and IC table
    trend = "c" if include_const else "n"
    cov_type = "HC0" if robust_se else "nonrobust"
    st.subheader("Model selection")
    with st.spinner("Fitting AR(p) candidates..."):
        ic_table = _collect_ic_table(df["ln_x"], p_max, trend, cov_type)
    st.dataframe(ic_table.style.format(precision=3), use_container_width=True)

    # Suggest p
    best_p = ic_table[ic_choice].astype(float).idxmin()
    st.success(f"Suggested p by {ic_choice}: **p = {best_p}**")

    # Allow override
    chosen_p = st.number_input("Choose p to fit", min_value=1, max_value=p_max, value=int(best_p), step=1)

    # Fit final
    with st.spinner("Estimating final model..."):
        res = _fit_autoreg(df["ln_x"], int(chosen_p), trend, cov_type)

    st.subheader("Final model summary")
    st.code(res.summary().as_text())

    # Residual diagnostics
    resid = pd.Series(res.resid, index=df["date"].iloc[-len(res.resid):])
    st.markdown("**Residual diagnostics**")
    lb, (jb_stat, jb_p, skew, kurt), dw = _diagnostics(resid, lags=min(12, max(4, int(chosen_p)+2)))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Durbin–Watson", f"{dw:0.3f}")
        st.caption("≈2 suggests little serial correlation.")
    with c2:
        st.metric("Jarque–Bera p‑value", f"{jb_p:0.3g}")
        st.caption("Low p suggests non‑normal residuals.")
    with c3:
        st.metric("Ljung–Box p‑value (lag m)", f"{float(lb['lb_pvalue'].iloc[0]):0.3g}")
        st.caption("Low p suggests residual autocorrelation.")

    _plot_residuals(resid.index, resid)

    st.markdown("**Correlograms of residuals**")
    _plot_acf_pacf(resid, nlags=min(36, max(10, int(chosen_p)+6)), title_prefix="Residual ")

    # Simple forecasts
    if forecast_h > 0:
        st.subheader("Out-of-sample forecasts (for ln(x))")
        fc = _forecast(res, int(forecast_h))
        if fc is not None:
            # Build a date index beyond last date, assuming a regular frequency if inferable
            last_date = df["date"].iloc[-1]
            try:
                # Try to infer freq
                inferred = pd.infer_freq(df["date"])
                future_idx = pd.date_range(last_date, periods=int(forecast_h)+1, freq=inferred)[1:]
            except Exception:
                # Fallback: use integer steps
                future_idx = [f"t+{i}" for i in range(1, int(forecast_h)+1)]
            fc_df = pd.DataFrame({"date": future_idx, "forecast_ln_x": fc.values})
            st.dataframe(fc_df, use_container_width=True)
            figf = px.line(fc_df, x="date", y="forecast_ln_x", title="Forecasts for ln(x)")
            figf.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(figf, use_container_width=True)
            st.caption("Note: Confidence intervals are not shown for AutoReg forecasts in this simple app.")

    # Download IC table
    csv = ic_table.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download information-criteria table (CSV)", data=csv, file_name="ar_ic_table.csv", mime="text/csv")

else:
    st.info("Upload your Excel file to begin.")

