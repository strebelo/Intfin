# fx_forecast.py
# Robust FX forecasting app with optional second-row "codes".
# Student controls AR lags on the exchange rate; exogenous variables enter contemporaneously (optional).
# The app now immediately reveals whether the random walk performs better (no button).

import streamlit as st

# ---- Explicit import probes so redacted errors don't hide the root cause ----
missing = []

try:
    import pandas as pd
except Exception:
    missing.append("pandas"); pd = None

try:
    import numpy as np
except Exception:
    missing.append("numpy"); np = None

# Optional: statsmodels
HAS_SM = True
try:
    import statsmodels.api as sm
except Exception:
    HAS_SM = False

# Optional: matplotlib (we'll prefer st.line_chart to avoid this dependency)
HAS_MPL = True
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    HAS_MPL = False

# Optional Excel readers
HAS_OPENPYXL = True
try:
    import openpyxl  # noqa: F401
except Exception:
    HAS_OPENPYXL = False

HAS_XLRD = True
try:
    import xlrd  # noqa: F401
except Exception:
    HAS_XLRD = False

st.set_page_config(page_title="FX Forecast", layout="wide")
st.title("Exchange-Rate Forecasting App")

# If core libs are missing, stop with a clear message
core_missing = [m for m in missing if m in ("pandas", "numpy")]
if core_missing:
    st.error(
        "Missing required packages: " + ", ".join(core_missing) +
        "\n\nAdd them to requirements.txt and redeploy."
    )
    st.stop()

# -------- Utilities --------
def add_constant_df(Xdf):
    Xdf = pd.DataFrame(Xdf).copy()
    if "const" not in Xdf.columns:
        Xdf.insert(0, "const", 1.0)
    return Xdf

class SimpleOLSResult:
    def __init__(self, params, columns):
        self.params = pd.Series(params, index=columns)

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.params.index)
        return X.values @ self.params.values

    def text_summary(self):
        return (
            "Simple OLS (NumPy fallback)\n"
            f"Parameters ({len(self.params)}):\n" +
            "\n".join(f"  {k}: {v:.6g}" for k, v in self.params.items())
        )

def np_ols_fit(y, X_df):
    X = X_df.values
    y = y.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return SimpleOLSResult(beta, X_df.columns)

def coerce_numeric_columns(df, date_col="date"):
    """Coerce all non-date columns to numeric, leaving date alone."""
    for c in df.columns:
        if date_col is not None and c == date_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def detect_probable_codes_row(df, date_col="date"):
    """Heuristic: if first data row has non-numeric entries in many non-date cols, treat as codes."""
    if df.empty:
        return False
    row0 = df.iloc[0]
    non_date_cols = [c for c in df.columns if c != date_col]
    if not non_date_cols:
        return False
    nonnum = 0
    for c in non_date_cols:
        v = row0[c]
        if pd.isna(v):
            continue
        try:
            float(str(v))
        except Exception:
            nonnum += 1
    return nonnum >= max(1, len(non_date_cols) // 3)

def make_target_lags(series, L, name):
    """Create lag columns for the target only."""
    if L <= 0:
        return pd.DataFrame(index=series.index)
    cols = {}
    for l in range(1, L + 1):
        cols[f"{name}_lag{l}"] = series.shift(l)
    return pd.DataFrame(cols, index=series.index)

# -------- Instructions --------
st.write(
    """
    **File format (monthly):**
    1) **Row 1**: variable names (e.g., `date`, `spot`, `cpi_us`, `cpi_uk`, ...)\n
    2) **Row 2** *(optional)*: variable **codes** (e.g., FRED tickers or your labels)\n
    3) **Row 3+**: monthly observations\n
    The first column can be `date` (YYYY-MM or YYYY-MM-DD).
    """
)

# -------- Upload --------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    if not HAS_SM:
        st.info("Optional: `statsmodels` not found — using NumPy OLS fallback.")
    if not HAS_MPL:
        st.info("Optional: `matplotlib` not found — using Streamlit charts.")
    st.stop()

# ---- Robust file read ----
try:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, header=0)
    elif name.endswith(".xlsx"):
        if not HAS_OPENPYXL:
            st.error("`openpyxl` not installed. Add it to requirements.txt or upload CSV.")
            st.stop()
        df = pd.read_excel(uploaded, engine="openpyxl", header=0)
    else:  # .xls
        if not HAS_XLRD:
            st.error("`xlrd` not installed for .xls files. Add it or upload CSV/.xlsx.")
            st.stop()
        df = pd.read_excel(uploaded, engine="xlrd", header=0)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

# ---- Handle optional codes row on line 2 ----
st.subheader("Data Preview & Options")

heuristic_codes = detect_probable_codes_row(df, date_col="date" if "date" in df.columns else None)
use_codes_row = st.checkbox("Row 2 contains variable codes (keep for reference, exclude from data)",
                            value=heuristic_codes)

codes_map = {}
if use_codes_row and len(df) >= 1:
    codes_series = df.iloc[0]
    for c in df.columns:
        if str(c).lower() != "date":
            codes_map[str(c)] = str(codes_series[c])
    df = df.iloc[1:].reset_index(drop=True)

# Coerce numeric (after removing codes row)
date_col = "date" if "date" in df.columns else None
df = coerce_numeric_columns(df, date_col=date_col)

st.write("Preview (first 6 rows):")
st.dataframe(df.head(6))

if codes_map:
    with st.sidebar.expander("Variable codes (from row 2)"):
        st.write(pd.DataFrame({"variable": list(codes_map.keys()),
                               "code": [codes_map[k] for k in codes_map]}))

# Datetime index if available
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].notna().any():
            df = df.set_index("date").sort_index()
        else:
            st.warning("Could not parse 'date' — continuing without a datetime index.")
            df = df.drop(columns=["date"])
    except Exception:
        st.warning("Could not parse 'date' — continuing without a datetime index.")
        df = df.drop(columns=["date"])

# ---- Variable selection ----
all_cols = list(df.columns)
if not all_cols:
    st.error("Your file has no usable columns after parsing.")
    st.stop()

target = st.selectbox("Dependent variable (exchange rate)", all_cols)
exog_choices = [c for c in all_cols if c != target]

exogs = st.multiselect(
    "Independent variables (contemporaneous only — optional)",
    exog_choices,
    default=[]
)

# ---- AR lags on the exchange rate only ----
ar_lags = st.slider("Number of AR lags on the exchange rate", 0, 12, 1)

# If no exogs are chosen and ar_lags == 0, there's nothing to estimate except a constant.
if (not exogs) and ar_lags == 0:
    st.warning("Select at least 1 AR lag when no independent variables are chosen.")
    st.stop()

Y = df[target].astype(float)
X_exog = df[exogs].astype(float) if exogs else pd.DataFrame(index=df.index)
X_ar = make_target_lags(Y, ar_lags, name=target)
X = pd.concat([X_exog, X_ar], axis=1)

# Align and drop NaNs introduced by lags
data = pd.concat([Y.rename(target), X], axis=1).dropna()
if data.empty:
    st.error("After adding AR lags and dropping NaNs, no rows remain. Try fewer lags or check data.")
    st.stop()

Y = data[target].astype(float)
X = data.drop(columns=[target]).astype(float)

# Add constant
if HAS_SM:
    X = sm.add_constant(X, has_constant='add')
else:
    X = add_constant_df(X)

# ---- Estimate ----
st.subheader("Regression Results")
if HAS_SM:
    fit = sm.OLS(Y, X).fit()
    try:
        st.text(fit.summary().as_text())
    except Exception as e:
        st.warning(f"Full statsmodels summary unavailable ({e}). Showing fallback parameters.")
        params_df = pd.DataFrame({"param": fit.params.index, "estimate": fit.params.values})
        st.dataframe(params_df)
else:
    fit = np_ols_fit(Y, X)
    st.text(fit.text_summary())

# In-sample R^2
yhat_in = pd.Series(fit.predict(X), index=Y.index)
ss_res = float(np.sum((Y - yhat_in) ** 2))
ss_tot = float(np.sum((Y - Y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
st.metric("In-sample R²", f"{r2:.4f}")

# ---- OOS vs Random Walk ----
st.subheader("Out-of-Sample Forecast Test")
ratio = st.slider("Training fraction", 0.5, 0.95, 0.8)
split = int(len(Y) * ratio)

X_tr, X_te = X.iloc[:split], X.iloc[split:]
Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

if HAS_SM:
    fit_oos = sm.OLS(Y_tr, X_tr).fit()
else:
    fit_oos = np_ols_fit(Y_tr, X_tr)

pred = pd.Series(fit_oos.predict(X_te), index=X_te.index)
rw = Y.shift(1).iloc[split:]

# Align to the same index
pred, Y_te = pred.align(Y_te, join="inner")
rw, Y_te = rw.align(Y_te, join="inner")

mse_model = float(np.mean((pred - Y_te) ** 2))
mse_rw = float(np.mean((rw - Y_te) ** 2))

c1, c2 = st.columns(2)
with c1: st.metric("OOS MSE — Model", f"{mse_model:.6g}")
with c2: st.metric("OOS MSE — Random Walk", f"{mse_rw:.6g}")

# ---- Immediate verdict (no button) ----
if np.isfinite(mse_model) and np.isfinite(mse_rw):
    if mse_model < mse_rw:
        improvement = (mse_rw - mse_model) / mse_rw * 100.0 if mse_rw > 0 else np.nan
        st.success(f"✅ Model beats the random walk"
                   + (f" (MSE ↓ {improvement:.2f}%)." if np.isfinite(improvement) else "."))
    elif mse_model > mse_rw:
        deterioration = (mse_model - mse_rw) / mse_rw * 100.0 if mse_rw > 0 else np.nan
        st.warning(f"⚠️ Random walk performs better"
                   + (f" (Model MSE ↑ {deterioration:.2f}% vs RW)." if np.isfinite(deterioration) else "."))
    else:
        st.info("⚖️ Tie: same MSE.")
else:
    st.info("Results not comparable due to insufficient or non-finite values.")

# ---- Chart ----
chart_df = pd.DataFrame({"Actual": Y_te, "Model": pred, "Random walk": rw}).dropna()
st.line_chart(chart_df)

# ---- Footer: show missing non-core deps so you can fix requirements.txt ----
notes = []
if not HAS_SM: notes.append("statsmodels")
if not HAS_MPL: notes.append("matplotlib (optional)")
if not HAS_OPENPYXL: notes.append("openpyxl (needed for .xlsx)")
if not HAS_XLRD: notes.append("xlrd (needed for legacy .xls)")

if notes:
    st.caption("Optional/missing packages: " + ", ".join(notes))
