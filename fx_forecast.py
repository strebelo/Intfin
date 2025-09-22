# fx_forecast.py
# Robust FX forecasting app that runs even if some packages are missing.
# It detects missing imports and falls back to pure NumPy/Streamlit where possible.

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

# Minimal NumPy OLS fallback if statsmodels is not available
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

st.write(
    """
    Upload a file with **monthly** data (CSV or Excel). Include columns like:
    - `date` (optional)
    - Spot exchange rate (dependent variable)
    - Inflation, GDP growth, trade balances (explanatory variables)
    
    Choose regressors and lags; we estimate OLS and compare out-of-sample
    forecasts to a random-walk benchmark.
    """
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    # Show which optional deps are missing (to fix requirements) but don’t block
    if not HAS_SM:
        st.info("Optional: `statsmodels` not found — using NumPy OLS fallback.")
    if not HAS_MPL:
        st.info("Optional: `matplotlib` not found — using Streamlit charts.")
    st.stop()

# ---- Robust file read ----
try:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith(".xlsx"):
        if not HAS_OPENPYXL:
            st.error("`openpyxl` not installed. Add it to requirements.txt or upload CSV.")
            st.stop()
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:  # .xls
        if not HAS_XLRD:
            st.error("`xlrd` not installed for .xls files. Add it or upload CSV/.xlsx.")
            st.stop()
        df = pd.read_excel(uploaded, engine="xlrd")
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.write("Preview:", df.head())

# Datetime index if available
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    except Exception:
        st.warning("Could not parse 'date' — continuing without a datetime index.")

# ---- Variable selection ----
all_cols = list(df.columns)
if not all_cols:
    st.error("Your file has no columns.")
    st.stop()

target = st.selectbox("Dependent variable (exchange rate)", all_cols)
exog_choices = [c for c in all_cols if c != target]
exogs = st.multiselect("Independent variables (choose ≥1)", exog_choices,
                       default=exog_choices[:1] if exog_choices else [])

if not exogs:
    st.warning("Select at least one independent variable.")
    st.stop()

max_lag = st.slider("Number of lags per chosen variable", 0, 12, 1)

def make_lags(df_in, cols, L):
    lagged = {}
    for v in cols:
        for l in range(1, L + 1):
            lagged[f"{v}_lag{l}"] = df_in[v].shift(l)
    return pd.DataFrame(lagged, index=df_in.index)

X = df[exogs].copy()
if max_lag > 0:
    X = pd.concat([X, make_lags(df, exogs, max_lag)], axis=1)

Y = df[target].copy()
data = pd.concat([Y, X], axis=1).dropna()
if data.empty:
    st.error("After adding lags and dropping NaNs, no rows remain. Try fewer lags or check data.")
    st.stop()

Y = data[target]
X = data.drop(columns=[target])

# Add constant
if HAS_SM:
    X = sm.add_constant(X)
else:
    X = add_constant_df(X)

# ---- Estimate ----
if HAS_SM:
    fit = sm.OLS(Y, X).fit()
    st.subheader("Regression Results")
    st.text(fit.summary().as_text())
else:
    fit = np_ols_fit(Y, X)
    st.subheader("Regression Results (fallback)")
    st.text(fit.text_summary())

# In-sample R^2
yhat_in = fit.predict(X)
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

pred = fit_oos.predict(X_te)
rw = Y.shift(1).iloc[split:]

pred, Y_te = pred.align(Y_te, join="inner")
rw, Y_te = rw.align(Y_te, join="inner")

mse_model = float(np.mean((pred - Y_te) ** 2))
mse_rw = float(np.mean((rw - Y_te) ** 2))

c1, c2 = st.columns(2)
with c1: st.metric("OOS MSE — Model", f"{mse_model:.6g}")
with c2: st.metric("OOS MSE — Random Walk", f"{mse_rw:.6g}")

st.line_chart(
    pd.DataFrame({"Actual": Y_te, "Model": pred, "Random walk": rw})
    .dropna()
)

# ---- Secret button ----
if st.button("Reveal secret forecast comparison"):
    if mse_model < mse_rw:
        st.success("✅ Model beats the random walk.")
    elif mse_model > mse_rw:
        st.warning("⚠️ Random walk performs better.")
    else:
        st.info("⚖️ Tie: same MSE.")

# ---- Footer: show missing non-core deps so you can fix requirements.txt ----
notes = []
if not HAS_SM: notes.append("statsmodels")
if not HAS_MPL: notes.append("matplotlib (optional)")
if not HAS_OPENPYXL: notes.append("openpyxl (needed for .xlsx)")
if not HAS_XLRD: notes.append("xlrd (needed for legacy .xls)")

if notes:
    st.caption("Optional/missing packages: " + ", ".join(notes))
