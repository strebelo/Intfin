# fx_forecast.py
# Streamlit app to forecast exchange rates with user-chosen regressors and lags
# Works with or without statsmodels installed (falls back to NumPy OLS if needed)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Optional statsmodels import with graceful fallback ----------
HAS_SM = True
try:
    import statsmodels.api as sm  # preferred
except Exception:
    HAS_SM = False

    # Minimal replacements for add_constant and OLS fit/predict via NumPy
    def add_constant(X):
        X = pd.DataFrame(X).copy()
        if "const" not in X.columns:
            X.insert(0, "const", 1.0)
        return X

    class SimpleOLSResult:
        def __init__(self, params, columns):
            self.params = pd.Series(params, index=columns)

        def predict(self, X):
            X = pd.DataFrame(X, columns=self.params.index)
            return X.values @ self.params.values

        # Small text summary so the UI shows *something* without statsmodels
        def text_summary(self):
            return (
                "Simple OLS (NumPy fallback)\n"
                f"Parameters ({len(self.params)}):\n"
                + "\n".join(f"  {k}: {v:.6g}" for k, v in self.params.items())
            )

    def np_ols_fit(y, X_df):
        # X_df must already include a const column
        X = X_df.values
        y = y.values
        # Solve (X'X)beta = X'y
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return SimpleOLSResult(beta, X_df.columns)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="FX Forecast", layout="wide")
st.title("Exchange-Rate Forecasting App")

st.write(
    """
    Upload a file with **monthly** data (CSV or Excel). Include columns like:
    - `date` (optional but recommended)
    - Spot exchange rate (e.g., `usd_gbp`)
    - Inflation (e.g., `us_infl`, `uk_infl`)
    - **Quarterly** GDP growth (monthly rows can be forward-filled or left with NaNs)
    - Trade deficits, etc.

    Choose regressors and lags; we estimate an OLS regression and compare
    its out-of-sample forecast against a random-walk benchmark.
    """
)

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.stop()

# Robust reader (CSV preferred; Excel requires openpyxl/xlrd)
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.write("Preview:", df.head())

# Ensure a datetime index if `date` exists
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    except Exception:
        st.warning("Could not parse 'date' column; continuing without a datetime index.")

# ---------- Variable selection ----------
all_vars = list(df.columns)
if len(all_vars) == 0:
    st.error("No columns found in the uploaded data.")
    st.stop()

target_var = st.selectbox("Dependent variable (exchange rate)", all_vars)
exog_candidates = [v for v in all_vars if v != target_var]
exog_vars = st.multiselect("Independent variables (choose ≥1)", exog_candidates, default=exog_candidates[:1])

if not exog_vars:
    st.warning("Select at least one independent variable.")
    st.stop()

max_lag = st.slider("Number of lags to include for each chosen variable", 0, 12, 1)

# ---------- Build lagged design matrix ----------
def make_lags(dataframe, cols, lags):
    lagged = {}
    for v in cols:
        for L in range(1, lags + 1):
            lagged[f"{v}_lag{L}"] = dataframe[v].shift(L)
    return pd.DataFrame(lagged, index=dataframe.index)

X = df[exog_vars].copy()
if max_lag > 0:
    X = pd.concat([X, make_lags(df, exog_vars, max_lag)], axis=1)

Y = df[target_var].copy()
data = pd.concat([Y, X], axis=1).dropna()
if data.empty:
    st.error("After adding lags and dropping NaNs, no rows remain. Try fewer lags or check your data.")
    st.stop()

Y = data[target_var]
X = data.drop(columns=[target_var])

# Add constant
if HAS_SM:
    X = sm.add_constant(X)
else:
    X = add_constant(X)

# ---------- Estimate model ----------
if HAS_SM:
    model = sm.OLS(Y, X).fit()
else:
    model = np_ols_fit(Y, X)

st.subheader("Regression Results")
if HAS_SM:
    # statsmodels' summary is a text table; render as preformatted text
    st.text(model.summary().as_text())
else:
    st.text(model.text_summary())

# In-sample R^2 (compute ourselves for consistency across both paths)
y_hat_in = model.predict(X)
ss_res = np.sum((Y - y_hat_in) ** 2)
ss_tot = np.sum((Y - Y.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
st.write(f"**In-sample R²:** {r2:.4f}")

# ---------- Out-of-sample forecast vs random walk ----------
st.subheader("Out-of-Sample Forecast Test")
split_ratio = st.slider("Fraction of data used for training", 0.5, 0.95, 0.8)
split_idx = int(len(Y) * split_ratio)

train_X, test_X = X.iloc[:split_idx], X.iloc[split_idx:]
train_Y, test_Y = Y.iloc[:split_idx], Y.iloc[split_idx:]

if HAS_SM:
    oos_model = sm.OLS(train_Y, train_X).fit()
else:
    oos_model = np_ols_fit(train_Y, train_X)

pred = oos_model.predict(test_X)

# Random-walk benchmark: y_{t-1}
rw_forecast = Y.shift(1).iloc[split_idx:]

# Align (just in case)
pred, test_Y_aligned = pred.align(test_Y, join="inner")
rw_forecast, test_Y_aligned = rw_forecast.align(test_Y_aligned, join="inner")

mse_model = float(np.mean((pred - test_Y_aligned) ** 2))
mse_rw = float(np.mean((rw_forecast - test_Y_aligned) ** 2))

col1, col2 = st.columns(2)
with col1:
    st.metric("OOS MSE — Model", f"{mse_model:.6g}")
with col2:
    st.metric("OOS MSE — Random Walk", f"{mse_rw:.6g}")

fig, ax = plt.subplots()
ax.plot(test_Y_aligned.index, test_Y_aligned.values, label="Actual")
ax.plot(pred.index, pred.values, label="Model forecast")
ax.plot(rw_forecast.index, rw_forecast.values, label="Random walk", linestyle="--")
ax.set_title("Out-of-Sample Forecast Comparison")
ax.legend()
st.pyplot(fig)

# ---------- Secret button to reveal performance ----------
if st.button("Reveal secret forecast comparison"):
    if mse_model < mse_rw:
        st.success("✅ Model beats the random walk.")
    elif mse_model > mse_rw:
        st.warning("⚠️ Random walk performs better.")
    else:
        st.info("⚖️ Tie: same MSE.")
