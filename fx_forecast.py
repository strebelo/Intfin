# fx_forecast.py
# Robust FX forecasting app with optional out-of-sample section (hidden by default).
# - Normalizes date index on load (handles strings, PeriodIndex, Excel serials, YYYYMM, etc.)
# - Students can choose target, AR lags, and contemporaneous exogenous variables
# - Out-of-sample block computes MSE using log-% errors: 100 * ln(Forecast / Actual)
# - Tight Altair plot with Actual in black, on top

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# ========================== Instructor switch ===============================
# Set to True when you want to reveal the Out-of-Sample block
ENABLE_OUT_OF_SAMPLE = False

# Optional: uncomment to use a sidebar code to toggle the OOS section
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
    else:
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
def normalize_date_index(df: pd.DataFrame, prefer_col: str | None = None) -> pd.DataFrame:
    """
    Make df.index a clean DatetimeIndex.
    - Picks a date column (case-insensitive 'date' if present, or prefer_col)
    - Handles strings 'YYYY-MM' / 'YYYY-MM-DD', integers like 197401/19740101, Excel serial days, PeriodIndex
    - Sorts by date and drops rows with NaT dates (reports how many)
    """
    df = df.copy()

    # Already datetime/period index?
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="start")
        return df.sort_index()

    # Choose candidate date column
    date_cols_exact = [c for c in df.columns if c.lower() == "date"]
    if prefer_col and prefer_col in df.columns:
        cand = prefer_col
    elif date_cols_exact:
        cand = date_cols_exact[0]
    else:
        # Heuristic: if the first column name contains "date"
        cand = df.columns[0] if re.fullmatch(r"(?i).*date.*", str(df.columns[0])) else None

    # If no candidate column, try coercing the index itself
    if cand is None:
        try:
            dt = pd.to_datetime(df.index, errors="coerce")
            n_bad = int(pd.isna(dt).sum())
            if n_bad < len(df):
                df.index = dt
                df = df.loc[~pd.isna(df.index)]
                return df.sort_index()
        except Exception:
            pass
        st.warning("Could not identify a date column; consider passing prefer_col='your_date_col'.")
        return df

    s = df[cand]

    # Try broad parse first
    dt = pd.to_datetime(s, errors="coerce")
    share_nat = float(pd.isna(dt).mean())

    # If many NaT, try specific patterns and numeric cases
    if share_nat > 0.2:
        s_num = pd.to_numeric(s, errors="ignore")

        # Excel serial days
        s_num2 = pd.to_numeric(s, errors="coerce")
        if s_num2.notna().mean() > 0.8:
            arr = s_num2.astype("Int64")
            if arr.notna().all():
                amin, amax = int(arr.min()), int(arr.max())
                # Typical Excel serial range (rough)
                if 10000 <= amin <= 80000 and 10000 <= amax <= 80000:
                    dt_try = pd.to_datetime(arr.astype("float"), unit="D", origin="1899-12-30", errors="coerce")
                    if dt_try.notna().sum() >= 0.8 * len(arr):
                        dt = dt_try

            # YYYYMM (e.g., 197401)
            if dt.isna().mean() > 0.2:
                s6 = arr.astype(str).str.zfill(6)
                dt_try = pd.to_datetime(s6, format="%Y%m", errors="coerce")
                if dt_try.notna().sum() >= 0.8 * len(arr):
                    dt = dt_try

            # YYYYMMDD (e.g., 19740101)
            if dt.isna().mean() > 0.2:
                s8 = arr.astype(str).str.zfill(8)
                dt_try = pd.to_datetime(s8, format="%Y%m%d", errors="coerce")
                if dt_try.notna().sum() >= 0.8 * len(arr):
                    dt = dt_try

        # Strings YYYY-MM only
        if dt.isna().mean() > 0.2:
            s_str = s.astype(str)
            if s_str.str.match(r"^\d{4}-\d{2}$").mean() > 0.5:
                dt_try = pd.to_datetime(s_str, format="%Y-%m", errors="coerce")
                if dt_try.notna().sum() >= 0.8 * len(s_str):
                    dt = dt_try

    n_nat = int(pd.isna(dt).sum())
    if n_nat > 0:
        st.info(f"Date normalization: dropping {n_nat} rows with unparseable dates from '{cand}'.")
    df = df.loc[~pd.isna(dt)].copy()
    df.index = pd.DatetimeIndex(dt[~pd.isna(dt)])
    # keep the original date col if you want; otherwise remove:
    df = df.drop(columns=[cand], errors="ignore")
    return df.sort_index()

# ============================== UI ==========================================
st.title("FX Forecast (AR lags + optional contemporaneous exogenous)")
st.caption("Dates are normalized automatically. OOS block uses log-% MSE and is hidden by default.")

uploaded = st.file_uploader("Upload data (CSV or Excel). Must include a 'date' column or a date-like index/column.", type=["csv", "xlsx", "xls"])

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

# Normalize dates
df = normalize_date_index(raw_df, prefer_col="date")

if df.empty:
    st.error("Dataframe is empty after date normalization.")
    st.stop()

# Keep only numeric columns for modeling
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found after loading the data.")
    st.stop()

st.write("Preview:", df.head())

# ========================= Controls =========================================
target = st.selectbox("Dependent variable (exchange rate)", numeric_cols, index=min(1, len(numeric_cols)-1))
exog_choices = [c for c in numeric_cols if c != target]
exogs = st.multiselect("Independent variables (optional, contemporaneous)", exog_choices, default=[])

max_lags = st.slider("Number of AR lags on the exchange rate", 1, 12, 1)

# ========================= Build regression data =============================
# Build lagged target regressors
lag_cols = {}
for L in range(1, max_lags + 1):
    lag_name = f"{target}_lag{L}"
    lag_cols[lag_name] = df[target].shift(L)

X_parts = [pd.DataFrame(lag_cols, index=df.index)]
if exogs:
    X_parts.append(df[exogs])

X = pd.concat(X_parts, axis=1)

# Align Y and X, drop rows with NaN from lags/contemporaneous
Y = df[target].copy()
XY = pd.concat([Y.rename("Y"), X], axis=1).dropna()
if XY.empty:
    st.error("After constructing lags and aligning data, there are no rows to estimate. Increase data length or reduce lags.")
    st.stop()

Y = XY["Y"]
X = XY.drop(columns=["Y"])

# ========================= In-sample quick fit (optional) ====================
with st.expander("In-sample fit (quick look)", expanded=False):
    # Fit OLS
    if HAS_SM:
        fit = sm.OLS(Y, add_constant_df(X)).fit()
        st.write(fit.summary())
    else:
        st.info("statsmodels not found; showing numpy OLS coefficients only.")
        fit = np_ols_fit(Y, X)
        st.write("Coefficients:", fit.params)

# =================== Out-of-sample vs Random Walk (hidden) ===================
if ENABLE_OUT_OF_SAMPLE:
    st.subheader("Out-of-Sample Forecast Test (log-% errors)")

    ratio = st.slider("Training fraction", 0.5, 0.95, 0.8, help="Share of observations used for training.")
    split = int(len(Y) * ratio)
    if split < max(10, max_lags + 1) or split >= len(Y) - 1:
        st.warning("Not enough observations in train/test after split. Adjust the training fraction.")
    else:
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

        # Fit OOS
        if HAS_SM:
            fit_oos = sm.OLS(Y_tr, add_constant_df(X_tr)).fit()
            pred = pd.Series(fit_oos.predict(add_constant_df(X_te)), index=X_te.index, dtype=float)
        else:
            fit_oos = np_ols_fit(Y_tr, X_tr)
            pred = pd.Series(fit_oos.predict(X_te), index=X_te.index, dtype=float)

        # Random walk forecast = lag1 of Y
        rw = Y.shift(1).iloc[split:].astype(float)

        # Align to common index
        common_idx = Y_te.index.intersection(pred.index).intersection(rw.index)
        Y_te_c = pd.to_numeric(Y_te.reindex(common_idx), errors="coerce")
        pred_c = pd.to_numeric(pred.reindex(common_idx), errors="coerce")
        rw_c   = pd.to_numeric(rw.reindex(common_idx),   errors="coerce")

        wide = pd.DataFrame({"Actual": Y_te_c, "Model": pred_c, "Random walk": rw_c}).dropna()

        if wide.empty:
            st.warning("No overlapping non-NaN points in OOS test after alignment.")
        else:
            # ---- Log-% errors: 100 * ln(Forecast / Actual) ----
            eps = 1e-12
            err_model = 100 * np.log((wide["Model"] + eps) / (wide["Actual"] + eps))
            err_rw    = 100 * np.log((wide["Random walk"] + eps) / (wide["Actual"] + eps))

            mse_model = float(np.mean(err_model ** 2))
            mse_rw    = float(np.mean(err_rw ** 2))

            c1, c2 = st.columns(2)
            with c1: st.metric("Log-% MSE — Model", f"{mse_model:.6g}")
            with c2: st.metric("Log-% MSE — Random Walk", f"{mse_rw:.6g}")

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

            # ------------- Tight Altair plot: Actual (black) on top -------------
            plot_df = wide.copy()
            # Index should already be DatetimeIndex thanks to normalize_date_index
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                # Fallback (should be rare)
                plot_df.index = pd.to_datetime(plot_df.index, errors="coerce")

            # Guardrails
            good = ~pd.isna(plot_df.index)
            plot_df = plot_df.loc[good]
            if plot_df.index.nunique() <= 1:
                st.warning("Only one unique date in OOS window—cannot plot a time series.")
            else:
                long_df = plot_df.copy()
                long_df["date"] = long_df.index
                long_df = long_df.melt(id_vars="date", var_name="Series", value_name="Value")

                # Tight y-range
                vmin, vmax = long_df["Value"].min(), long_df["Value"].max()
                tiny = (vmax - vmin) * 0.01 or 0.01
                ydom = [vmin - tiny, vmax + tiny]

                others = long_df[long_df["Series"] != "Actual"]
                actual = long_df[long_df["Series"] == "Actual"]

                base_x = alt.X("date:T", title="Date")
                base_y = alt.Y("Value:Q", scale=alt.Scale(domain=ydom), title="Exchange rate")
                base_tt = ["date:T", "Series:N", "Value:Q"]

                layer_others = (
                    alt.Chart(others)
                    .mark_line()
                    .encode(
                        x=base_x,
                        y=base_y,
                        color=alt.Color(
                            "Series:N",
                            scale=alt.Scale(domain=["Model", "Random walk"], range=["#1f77b4", "#ff7f0e"]),
                            legend=alt.Legend(title="Series"),
                        ),
                        tooltip=base_tt,
                    )
                )

                layer_actual = (
                    alt.Chart(actual)
                    .mark_line(color="black")
                    .encode(x=base_x, y=base_y, size=alt.value(3), tooltip=base_tt)
                )

                st.altair_chart((layer_others + layer_actual).interactive(), use_container_width=True)
else:
    # OOS section is completely hidden from students
    pass
