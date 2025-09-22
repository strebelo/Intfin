# fx_forecast.py
# Streamlit app to forecast exchange rates with user-chosen regressors and lags

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm   # <-- correct import

st.set_page_config(page_title="FX Forecast", layout="wide")
st.title("Exchange-Rate Forecasting App")

st.write(
    """
    Upload an Excel file with **monthly** data on:
    * Spot exchange rate (e.g., USD/GBP)
    * Inflation for the two countries
    * **Quarterly** GDP growth for the two countries
    * Trade deficit for the two countries

    You can then choose which variables and how many lags to include.
    The app estimates an OLS regression and compares its out-of-sample
    forecast to a random walk.
    """
)

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)
st.write("Preview of uploaded data:", df.head())

# Ensure a datetime index if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

# ---------- User choices ----------
all_vars = list(df.columns)
target_var = st.selectbox("Dependent variable (exchange rate)", all_vars)
exog_vars = st.multiselect("Independent variables", [v for v in all_vars if v != target_var])

if not exog_vars:
    st.warning("Select at least one independent variable.")
    st.stop()

max_lag = st.slider("Number of lags to include for each variable", 0, 12, 1)

# ---------- Build design matrix ----------
def make_lags(data, lags):
    lagged = {}
    for v in data.columns:
        for L in range(1, lags + 1):
            lagged[f"{v}_lag{L}"] = data[v].shift(L)
    return pd.DataFrame(lagged, index=data.index)

X = df[exog_vars]
if max_lag > 0:
    X = pd.concat([X, make_lags(df[exog_vars], max_lag)], axis=1)

# Align with dependent variable and drop missing rows
Y = df[target_var]
data = pd.concat([Y, X], axis=1).dropna()
Y = data[target_var]
X = data.drop(columns=[target_var])
X = sm.add_constant(X)  # adds intercept

# ---------- Estimate model ----------
model = sm.OLS(Y, X).fit()
st.subheader("Regression Results")
st.write(model.summary())

# ---------- Out-of-sample forecast vs random walk ----------
st.subheader("Out-of-Sample Forecast Test")
split_ratio = st.slider("Fraction of data for training", 0.5, 0.95, 0.8)
split_idx = int(len(Y) * split_ratio)

train_X, test_X = X.iloc[:split_idx], X.iloc[split_idx:]
train_Y, test_Y = Y.iloc[:split_idx], Y.iloc[split_idx:]

oos_model = sm.OLS(train_Y, train_X).fit()
pred = oos_model.predict(test_X)

# Random walk benchmark: previous actual as forecast
rw_forecast = Y.shift(1).iloc[split_idx:]

mse_model = np.mean((pred - test_Y) ** 2)
mse_rw = np.mean((rw_forecast - test_Y) ** 2)

st.write(f"**Out-of-sample MSE (model):** {mse_model:.4f}")
st.write(f"**Out-of-sample MSE (random walk):** {mse_rw:.4f}")

fig, ax = plt.subplots()
ax.plot(test_Y.index, test_Y, label="Actual")
ax.plot(test_Y.index, pred, label="Model forecast")
ax.plot(test_Y.index, rw_forecast, label="Random walk", linestyle="--")
ax.legend()
ax.set_title("Out-of-Sample Forecast Comparison")
st.pyplot(fig)

# ---------- Secret button to reveal performance ----------
if st.button("Reveal secret forecast comparison"):
    st.success("Model beats random walk!" if mse_model < mse_rw else "Random walk performs better.")
