# fx_forecast_class.py
# One-step-ahead FX forecasting app:
#   • Y = S_{t+1}
#   • Regressors at time t: [S_t, S_{t-1}, …, X_t]
#   • Robust date normalization + numeric coercion
#   • In-sample chart and R²
#   • OOS block (toggle via ENABLE_OUT_OF_Sample) using log-% MSE: 100*ln(Forecast/Actual)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from typing import Optional

# ========================== Instructor switch ===============================
ENABLE_OUT_OF_SAMPLE = YES  # Set False to hide the OOS section

# Optional: sidebar code to toggle (uncomment to use)
# code = st.sidebar.text_input("Instructor code", type="password", placeholder="••••")
# ENABLE_OUT_OF_SAMPLE = ENABLE_OUT_OF_SAMPLE or (code == "reveal-oos")

# ======================== Optional statsmodels ==============================
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

def add_constant_df(X: pd.DataFrame) -> pd.DataFrame:
    """Add intercept column consistently for both numpy and statsmodels paths."""
    if HAS_SM:
        return sm.add_constant(X, has_constant="add")
    Xc = X.copy()
    if "const" not in Xc.columns:
        Xc.insert(0, "const", 1.0)
    return Xc

def np_ols_fit(y: pd.Series, X: pd.DataFrame):
    """Simple OLS via numpy for environments without statsmodels."""
    Xc = add_constant_df(X)
    beta, *_ = np.linalg.lstsq(Xc.values, y.values, rcond=None)
    class Result:
        params = pd.Series(beta, index=Xc.columns)
        def predict(self, Xnew):
            Xn = add_constant_df(pd.DataFrame(Xnew, copy=True))
            return np.dot(Xn.values, self.params.values)
    return Result()

# ======================== Date normalization ================================
def normalize_date_index(df: pd.DataFrame, prefer_col: Optional[str] = None) -> pd.DataFrame:
    """
    Make df.index a clean DatetimeIndex.
    - Chooses a date column (case-insensitive 'date' or 'prefer_col'), else tries the index.
    - Handles strings 'YYYY-MM' / 'YYYY-MM-DD', integers like 197401/19740101, Excel serial days, PeriodIndex.
    - Sorts by date and drops rows with NaT dates.
    """
    df = df.copy()

    # Already datetime/period index?
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="start")
        return df.sort_index()

    # Choose candidate date column
    date_cols_exact = [c for c in df.columns if str(c).lower() == "date"]
    if prefer_col and prefer_col in df.columns:
        cand = prefer_col
    elif date_cols_exact:
        cand = date_cols_exact[0]
    else:
        cand = df.columns[0] if re.fullmatch(r"(?i).*date.*", str(df.columns[0])) else None

    if cand is None:
        # Try to coerce the existing index
        try:
            dt = pd.to_datetime(df.index, errors="coerce")
            if pd.isna(dt).all():
                st.warning("Could not identify a date column; consider passing prefer_col='your_date_col'.")
                return df
            df.index = dt
            df = df.loc[~pd.isna(df.index)]
            return df.sort_index()
        except Exception:
            st.warning("Could not identify a date column; consider passing prefer_col='your_date_col'.")
            return df

    s = df[cand]
    dt = pd.to_datetime(s, errors="coerce")

    # If many NaT, try specific numeric/string formats
    if dt.isna().mean() > 0.2:
        # Numeric possibilities (Excel serial / YYYYMM / YYYYMMDD)
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().mean() > 0.8:
            arr = s_num.astype("Int64")
            if arr.notna().all():
                amin, amax = int(arr.min()), int(arr.max())
                # Excel serial days (rough range)
                if 10000 <= amin <= 80000 and 10000 <= amax <= 80000:
                    dt_try = pd.to_datetime(arr.astype("float"), unit="D", origin="1899-12-30", errors="coerce")
                    if dt_try.notna().sum() >= 0.8 * len(arr):
                        dt = dt_try
                # YYYYMM
                if dt.isna().mean() > 0.2:
                    s6 = arr.astype(str).str.zfill(6)
                    dt_try = pd.to_datetime(s6, format="%Y%m", errors="coerce")
                    if dt_try.notna().sum() >= 0.8 * len(arr):
                        dt = dt_try
                # YYYYMMDD
                if dt.isna().mean() > 0.2:
                    s8 = arr.astype(str).str.zfill(8)
                    dt_try = pd.to_datetime(s8, format="%Y%m%d", errors="coerce")
                    if dt_try.notna().sum() >= 0.8 * len(arr):
                        dt = dt_try

        # Strings like YYYY-MM (no day)
        if dt.isna().mean() > 0.2:
            s_str = s.astype(str)
            if s_str.str.match(r"^\d{4}-\d{2}$").mean() > 0.5:
                dt_try = pd.to_datetime(s_str, format="%Y-%m", errors="coerce")
                if dt_try.notna().sum() >= 0.8 * len(s_str):
                    dt = dt_try

    # Drop NaT and set index
    ok = ~pd.isna(dt)
    df = df.loc[ok].copy()
    df.index = pd.DatetimeIndex(dt[ok])
    df = df.drop(columns=[cand], errors="ignore")
    return df.sort_index()

# ============================== UI ==========================================
st.title("Forecasting spot exchange rates (one-step ahead)")
# st.caption("Y = S_{t+1}; regressors at time t. OOS block uses log-% MSE and can be toggled.")

uploaded = st.file_uploader(
    "Upload data (CSV or Excel). Must include a 'date' column or a date-like index/column.",
    type=["csv", "xlsx", "xls"]
)

if uploaded is None:
    st.info("Please upload a CSV/Excel file to begin.")
    st.stop()

# Read file
try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(uploaded)
    else:
        raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Normalize dates once
df = normalize_date_index(raw_df, prefer_col="date")
if df.empty:
    st.error("Dataframe is empty after date normalization.")
    st.stop()

# ========================= Coerce + choose columns ===========================
# Try to coerce non-numeric columns into numeric (remove commas/thin spaces)
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        continue
    s = (
        df[c].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u2009", "", regex=False)  # thin space
        .str.strip()
    )
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() >= 0.80:
        df[c] = s_num

candidate_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not candidate_cols:
    st.error("No numeric columns found after loading/coercing the data.")
    st.stop()

# Show a formatted date preview without touching the working df/index
st.write(
    df.copy()
      .assign(date_fmt=df.index.strftime("%Y-%m"))
      .set_index("date_fmt")
)

# ========================= Controls (spot default) ===========================
def _score_as_spot(name: str) -> int:
    """Higher score => more likely to be a spot FX series (deprioritize policy rates)."""
    n = name.lower().strip()
    if any(bad in n for bad in ["bank rate", "boe", "policy", "base rate", "yield", "cpi", "ppi"]):
        return -5
    score = 0
    if n in {"spot", "s", "spot_fx", "spot rate", "exchange rate", "exrate", "fx_spot", "s_fx"}:
        score += 8
    if "spot" in n:  score += 5
    if "fx" in n:    score += 3
    if "exch" in n:  score += 3
    if "rate" in n:  score += 1
    if re.search(r"\b[A-Z]{3}[/\- ]?[A-Z]{3}\b", name): score += 6
    if re.fullmatch(r"[A-Z]{6}", name): score += 5
    for kw in ["usd", "eur", "gbp", "jpy", "chf", "aud", "cad", "nzd", "brl", "mxn", "cny", "hkd", "sgd"]:
        if kw in n:
            score += 2
    return score

# Choose default spot column by score; fall back to first numeric
scored = sorted(((c, _score_as_spot(str(c))) for c in candidate_cols), key=lambda x: x[1], reverse=True)
spot_col = scored[0][0] if scored else candidate_cols[0]
default_index = candidate_cols.index(spot_col)

target = st.selectbox(
    "Dependent variable (exchange rate)",
    candidate_cols,
    index=default_index,
    help="Defaults to the most likely spot exchange-rate series."
)

# Exogenous choices exclude the target; default empty (contemporaneous X_t)
exog_choices = [c for c in candidate_cols if c != target]
exogs = st.multiselect(
    "Independent variables at time t (optional, contemporaneous)",
    exog_choices,
    default=[]
)

# AR lags on the dependent variable
max_lags = st.slider("Number of AR lags on the exchange rate", 0, 12, 0)

# ========================= Build regression data =============================
# Build lagged target regressors S_{t-1}, S_{t-2}, ...
lag_cols = {}
for L in range(1, max_lags + 1):
    lag_name = f"{target}_lag{L}"
    lag_cols[lag_name] = df[target].shift(L)

X_parts = [pd.DataFrame(lag_cols, index=df.index)]
if exogs:
    X_parts.append(df[exogs])  # contemporaneous X_t

X = pd.concat(X_parts, axis=1)

# ---- ONE-STEP-AHEAD CHANGE: set Y = S_{t+1} at index t ----
Y = df[target].shift(-1).rename("Y_next")  # S_{t+1}

# Align and drop rows with NaN from lags/contemporaneous
XY = pd.concat([Y, X], axis=1).dropna()
if XY.empty:
    st.error("After constructing Y=S_{t+1} and regressors at t, no rows remain. Increase data length or reduce lags.")
    st.stop()

Y = XY["Y_next"]
X = XY.drop(columns=["Y_next"])

# ========================= In-sample quick fit (optional) ====================
with st.expander("Model performance statistics", expanded=False):
    if HAS_SM:
        fit = sm.OLS(Y, add_constant_df(X)).fit()
        st.write(fit.summary())
    else:
        st.info("statsmodels not found; showing numpy OLS coefficients only.")
        fit = np_ols_fit(Y, X)
        st.write("Coefficients:", fit.params)

# ========================= In-sample performance (chart + R²) ================
st.subheader("Model Performance")

# Fit using the full estimation sample
if HAS_SM:
    fit_ins = sm.OLS(Y, add_constant_df(X)).fit()
    yhat = pd.Series(fit_ins.predict(add_constant_df(X)), index=Y.index, dtype=float)
    r2_insample = fit_ins.rsquared   # R² for S_{t+1} fit
else:
    fit_ins = np_ols_fit(Y, X)
    yhat = pd.Series(fit_ins.predict(X), index=Y.index, dtype=float)
    ss_res = np.sum((Y - yhat) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2_insample = 1 - ss_res / ss_tot

st.metric("Model R² (Sₜ₊₁ on info at t)", f"{r2_insample:.4f}")

# Align for plotting
wide_is = pd.DataFrame({
    "Actual Sₜ₊₁": pd.to_numeric(Y, errors="coerce"),
    "Model Ŝₜ₊₁":  pd.to_numeric(yhat, errors="coerce"),
}).dropna()

if wide_is.empty or not isinstance(wide_is.index, pd.DatetimeIndex):
    wide_is = wide_is.copy()
    wide_is.index = pd.to_datetime(wide_is.index, errors="coerce")
    wide_is = wide_is.dropna()

if wide_is.empty or wide_is.index.nunique() <= 1:
    st.warning("Not enough valid in-sample points to plot.")
else:
    long_df = wide_is.copy()
    long_df["date"] = long_df.index
    long_df = long_df.melt(id_vars="date", var_name="Series", value_name="Value")

    vmin, vmax = long_df["Value"].min(), long_df["Value"].max()
    tiny = (vmax - vmin) * 0.01 or 0.01
    ydom = [vmin - tiny, vmax + tiny]

    color_scale = alt.Scale(
        domain=["Actual Sₜ₊₁", "Model Ŝₜ₊₁"],
        range=["black", "#1f77b4"]
    )

    chart_is = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date (indexed by t)"),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=ydom), title="Next-period exchange rate Sₜ₊₁"),
            color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="Series")),
            size=alt.condition("datum.Series == 'Actual Sₜ₊₁'", alt.value(3), alt.value(1.5)),
            tooltip=["date:T", "Series:N", "Value:Q"]
        )
        .interactive()
    )
    st.altair_chart(chart_is, use_container_width=True)

# =================== Out-of-sample vs Random Walk (optional) =================
if ENABLE_OUT_OF_SAMPLE:
    st.subheader("Out-of-Sample Forecast Performance (log-% errors) — One-step ahead")

    ratio = st.slider("Training fraction", 0.5, 0.95, 0.8, help="Share of observations used for training.")
    split = int(len(Y) * ratio)
    if split < max(10, max_lags + 2) or split >= len(Y) - 1:
        st.warning("Not enough observations in train/test after split. Adjust the training fraction.")
    else:
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

        # Fit OOS (predict S_{t+1})
        if HAS_SM:
            fit_oos = sm.OLS(Y_tr, add_constant_df(X_tr)).fit()
            pred = pd.Series(fit_oos.predict(add_constant_df(X_te)), index=X_te.index, dtype=float)
        else:
            fit_oos = np_ols_fit(Y_tr, X_tr)
            pred = pd.Series(fit_oos.predict(X_te), index=X_te.index, dtype=float)

        # Random walk one-step-ahead: forecast S_{t+1} with S_t aligned on the same index
        rw = df[target].reindex(Y_te.index)  # S_t aligned to predict Y_te = S_{t+1}

        # Align to common index
        common_idx = Y_te.index.intersection(pred.index).intersection(rw.index)
        Y_te_c = pd.to_numeric(Y_te.reindex(common_idx), errors="coerce")
        pred_c = pd.to_numeric(pred.reindex(common_idx), errors="coerce")
        rw_c   = pd.to_numeric(rw.reindex(common_idx),   errors="coerce")

        wide = pd.DataFrame({"Actual Sₜ₊₁": Y_te_c, "Model Ŝₜ₊₁": pred_c, "RW (Sₜ)": rw_c}).dropna()

        if wide.empty:
            st.warning("No overlapping non-NaN points in OOS test after alignment.")
        else:
            # ---- Log-% errors: 100 * ln(Forecast / Actual) ----
            eps = 1e-12
            err_model = 100 * np.log((wide["Model Ŝₜ₊₁"] + eps) / (wide["Actual Sₜ₊₁"] + eps))
            err_rw    = 100 * np.log((wide["RW (Sₜ)"]       + eps) / (wide["Actual Sₜ₊₁"] + eps))

            mse_model = float(np.mean(err_model ** 2))
            mse_rw    = float(np.mean(err_rw ** 2))

            c1, c2 = st.columns(2)
            with c1: st.metric("Log-% MSE — Model (Ŝₜ₊₁)", f"{mse_model:.6g}")
            with c2: st.metric("Log-% MSE — Random Walk (Sₜ)", f"{mse_rw:.6g}")

            if np.isfinite(mse_model) and np.isfinite(mse_rw):
                if mse_model < mse_rw:
                    imp = (mse_rw - mse_model) / mse_rw * 100 if mse_rw > 0 else np.nan
                    st.success(f"✅ Model beats RW (Log-% MSE ↓ {imp:.2f}%).")
                elif mse_model > mse_rw:
                    det = (mse_model - mse_rw) / mse_rw * 100 if mse_rw > 0 else np.nan
                    st.warning(f"⚠️ Random walk better (Model Log-% MSE ↑ {det:.2f}% vs RW).")
                else:
                    st.info("⚖️ Tie: same Log-% MSE.")
            else:
                st.info("Results not comparable due to non-finite values.")

            # ---- OOS plot ----
            plot_df = wide.copy()
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                plot_df.index = pd.to_datetime(plot_df.index, errors="coerce")
            good = ~pd.isna(plot_df.index)
            plot_df = plot_df.loc[good]

            if plot_df.index.nunique() <= 1:
                st.warning("Only one unique date in OOS window—cannot plot a time series.")
            else:
                long_df = plot_df.copy()
                long_df["date"] = long_df.index
                long_df = long_df.melt(id_vars="date", var_name="Series", value_name="Value")

                vmin, vmax = long_df["Value"].min(), long_df["Value"].max()
                tiny = (vmax - vmin) * 0.01 or 0.01
                ydom = [vmin - tiny, vmax + tiny]

                color_scale = alt.Scale(
                    domain=["Actual Sₜ₊₁", "Model Ŝₜ₊₁", "RW (Sₜ)"],
                    range=["black", "#1f77b4", "#ff7f0e"]
                )

                chart = (
                    alt.Chart(long_df)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Date (indexed by t)"),
                        y=alt.Y("Value:Q", scale=alt.Scale(domain=ydom), title="Next-period exchange rate Sₜ₊₁"),
                        color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="Series")),
                        size=alt.condition("datum.Series == 'Actual Sₜ₊₁'", alt.value(3), alt.value(1.5)),
                        tooltip=["date:T", "Series:N", "Value:Q"]
                    )
                    .interactive()
                )

                st.altair_chart(chart, use_container_width=True)
