# fx_forecast.py
# Streamlit app for exchange-rate forecasting with in-sample fit and (hidden) out-of-sample test vs random walk.

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS


st.set_page_config(page_title="FX Forecasting Lab", layout="wide")
st.title("FX Forecasting Lab: Competing with the Random Walk")

st.markdown(
    """
Upload an **Excel** file with a monthly Date column and these series (names can be mapped below):
- Spot exchange rate (e.g., USD/GBP)
- Inflation (home & foreign), monthly
- Real GDP growth (home & foreign), **quarterly**
- Trade deficit (home & foreign), **quarterly** (level or growth — we’ll treat as given)

> The model forecasts **next month’s exchange-rate change** (Δs<sub>t+1</sub>) from lagged Δs and macro **differentials** (home − foreign).  
> The random-walk benchmark implies Δs<sub>t+1</sub>=0.
"""
)

# ----- Config -----
INSTRUCTOR_PASSCODE = "Taylor1953"  # <-- change if you like

# ----- Upload -----
file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
if file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# Load first sheet by default
try:
    df_raw = pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

# Basic cleaning
df_raw_columns = list(df_raw.columns)
st.subheader("Map your columns")
with st.expander("Map columns (required)"):
    date_col = st.selectbox("Date column (monthly frequency)", df_raw_columns, index=0)
    s_col    = st.selectbox("Spot exchange rate (level, e.g., USD/GBP)", df_raw_columns)
    inf_h    = st.selectbox("Inflation (home, monthly)", df_raw_columns)
    inf_f    = st.selectbox("Inflation (foreign, monthly)", df_raw_columns)
    gdp_h    = st.selectbox("Real GDP growth (home, quarterly)", df_raw_columns)
    gdp_f    = st.selectbox("Real GDP growth (foreign, quarterly)", df_raw_columns)
    tb_h     = st.selectbox("Trade deficit (home, quarterly)", df_raw_columns)
    tb_f     = st.selectbox("Trade deficit (foreign, quarterly)", df_raw_columns)

# Parse and set Date index
df = df_raw.copy()
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception as e:
    st.error(f"Date parsing failed: {e}")
    st.stop()

df = df.set_index(date_col).sort_index()

# Sanity: ensure monthly index (we’ll allow gaps but warn)
if df.index.inferred_type not in ["datetime64", "datetime64tz"]:
    st.error("Date column must be datetime-like.")
    st.stop()

# Build core series
data = pd.DataFrame(index=df.index)
data["s_level"] = df[s_col].astype(float)
data["log_s"]   = np.log(data["s_level"])
data["dlog_s"]  = data["log_s"].diff() * 100  # percent monthly change

# Monthly inflation differential
data["inf_h"] = df[inf_h].astype(float)
data["inf_f"] = df[inf_f].astype(float)
data["inf_diff"] = data["inf_h"] - data["inf_f"]  # home - foreign

# Quarterly → Monthly: forward-fill within each quarter
def upsample_quarterly_to_monthly(series_like):
    # If index is monthly but values are quarterly, ffill within months
    ser = series_like.copy()
    # If file already monthly for these cols, we keep as-is; otherwise ffill gaps
    return ser.asfreq("MS", method=None).ffill() if ser.index.inferred_freq != "MS" else ser

# GDP growth differential (already growth rates per quarter; we ffill monthly)
gdp_h_q = df[gdp_h].astype(float)
gdp_f_q = df[gdp_f].astype(float)
gdp_h_m = upsample_quarterly_to_monthly(gdp_h_q).reindex(data.index).ffill()
gdp_f_m = upsample_quarterly_to_monthly(gdp_f_q).reindex(data.index).ffill()
data["gdp_diff"] = gdp_h_m - gdp_f_m

# Trade deficit differential (level or growth as supplied; ffill monthly)
tb_h_q = df[tb_h].astype(float)
tb_f_q = df[tb_f].astype(float)
tb_h_m = upsample_quarterly_to_monthly(tb_h_q).reindex(data.index).ffill()
tb_f_m = upsample_quarterly_to_monthly(tb_f_q).reindex(data.index).ffill()
data["tb_diff"] = tb_h_m - tb_f_m

st.subheader("Variable choices")
st.caption("We forecast next month’s Δs. Choose which predictors to include and how many lags.")

colA, colB, colC = st.columns(3)
with colA:
    use_ds = st.checkbox("Include Δs lags (momentum/mean reversion)", value=True)
    max_lag_ds = st.slider("Lags of Δs", 0, 12, 3)
with colB:
    use_inf = st.checkbox("Include inflation differential", value=True)
    max_lag_inf = st.slider("Lags of inf_diff", 0, 12, 3)
with colC:
    use_gdp = st.checkbox("Include GDP growth differential", value=True)
    max_lag_gdp = st.slider("Lags of gdp_diff", 0, 12, 3)

colD, colE = st.columns(2)
with colD:
    use_tb = st.checkbox("Include trade-deficit differential", value=False)
    max_lag_tb = st.slider("Lags of tb_diff", 0, 12, 2)
with colE:
    add_rate = st.checkbox("(Optional) add short-rate differential (if present)", value=False)
    rate_h_col = st.selectbox("Short rate (home)", ["<none>"] + df_raw_columns, index=0)
    rate_f_col = st.selectbox("Short rate (foreign)", ["<none>"] + df_raw_columns, index=0)
    max_lag_rate = st.slider("Lags of rate_diff", 0, 12, 3)

# Optional rate differential
if add_rate and rate_h_col != "<none>" and rate_f_col != "<none>":
    try:
        rate_h = df[rate_h_col].astype(float)
        rate_f = df[rate_f_col].astype(float)
        data["rate_diff"] = rate_h.reindex(data.index).ffill() - rate_f.reindex(data.index).ffill()
    except Exception as e:
        st.warning(f"Could not build rate_diff: {e}")

# Build regression design: y_t = Δs_{t+1}, X_t = lags at time t
y = data["dlog_s"].shift(-1).rename("dlog_s_lead1")

X_list = []
def add_lags(series, K, prefix):
    cols = {}
    for k in range(1, K + 1):
        cols[f"{prefix}_lag{k}"] = series.shift(k)
    return pd.DataFrame(cols)

if use_ds and max_lag_ds > 0:
    X_list.append(add_lags(data["dlog_s"], max_lag_ds, "ds"))

if use_inf and max_lag_inf > 0:
    X_list.append(add_lags(data["inf_diff"], max_lag_inf, "inf"))

if use_gdp and max_lag_gdp > 0:
    X_list.append(add_lags(data["gdp_diff"], max_lag_gdp, "gdp"))

if use_tb and max_lag_tb > 0:
    X_list.append(add_lags(data["tb_diff"], max_lag_tb, "tb"))

if "rate_diff" in data.columns and max_lag_rate > 0:
    X_list.append(add_lags(data["rate_diff"], max_lag_rate, "rate"))

if len(X_list) == 0:
    st.warning("Select at least one predictor or lag.")
    st.stop()

X = pd.concat(X_list, axis=1)
Z = pd.concat([y, X], axis=1).dropna()

if Z.empty or Z.shape[0] < 24:
    st.warning("Not enough non-missing rows after lags. Try fewer lags or check your data.")
    st.stop()

# Train/Test split controls
st.subheader("Sample split")
col1, col2 = st.columns(2)
with col1:
    use_ratio = st.checkbox("Use a train ratio (instead of a date)", value=True)
with col2:
    if use_ratio:
        train_ratio = st.slider("Training share", 0.5, 0.95, 0.7, 0.01)
        split_idx = int(len(Z) * train_ratio)
        split_date = Z.index[split_idx]
    else:
        min_date, max_date = Z.index.min(), Z.index.max()
        split_date = st.date_input("Train ends on/before:", value=pd.to_datetime(min_date).date(),
                                   min_value=pd.to_datetime(min_date).date(),
                                   max_value=pd.to_datetime(max_date).date())
        split_date = pd.to_datetime(split_date)

train = Z[Z.index <= split_date] if not use_ratio else Z.iloc[:split_idx]
test  = Z[Z.index >  split_date] if not use_ratio else Z.iloc[split_idx:]

if len(train) < 24:
    st.warning("Training sample is too short; increase training period.")
    st.stop()

# Fit OLS on training
X_train = add_constant(train.drop(columns=["dlog_s_lead1"]))
y_train = train["dlog_s_lead1"]
model = OLS(y_train, X_train).fit()

st.subheader("In-sample results (training period)")
st.write(f"Training obs: **{len(train)}**, Test obs: **{len(test)}**")
st.write(model.summary2().tables[1])  # coefficients table

r2 = model.rsquared
r2_adj = model.rsquared_adj
st.metric("R² (train)", f"{r2:.3f}", delta=f"Adj {r2_adj:.3f}")

# Plot actual vs fitted (train)
train_fitted = pd.Series(model.fittedvalues, index=train.index, name="Fitted Δs_{t+1}")
fig1, ax1 = plt.subplots(figsize=(8, 4))
train["dlog_s_lead1"].plot(ax=ax1, label="Actual Δs_{t+1}")
train_fitted.plot(ax=ax1, label="Fitted Δs_{t+1}")
ax1.set_title("Training: Actual vs Fitted next-month Δs")
ax1.set_ylabel("Percent")
ax1.legend()
st.pyplot(fig1)

# ----- Instructor OOS block -----
st.subheader("Instructor")
with st.expander("Enter passcode to reveal out-of-sample test"):
    code = st.text_input("Passcode", type="password")
    if code == INSTRUCTOR_PASSCODE and len(test) >= 12:
        st.success("Instructor mode enabled. Running real-time out-of-sample evaluation.")
        # Real-time (expanding window) forecasts of dlog_s_lead1
        preds = []
        idxs  = []

        full = Z.copy()
        # Find end position of train in full index
        if use_ratio:
            train_end_pos = split_idx
        else:
            train_end_pos = full.index.get_loc(full[full.index <= split_date].index[-1]) + 0

        for tpos in range(train_end_pos, len(full)):
            # Use data up through tpos (inclusive) as "available at t"
            avail = full.iloc[:tpos+1]
            # Need at least as much as initial training
            if len(avail) < len(train):
                continue
            X_avail = add_constant(avail.drop(columns=["dlog_s_lead1"]))
            y_avail = avail["dlog_s_lead1"]
            # Fit on available data up to tpos-1 so we forecast obs at tpos (which is y at index tpos)
            # But because y is Δs_{t+1} aligned to features at t, predicting y at tpos uses X at tpos
            # We fit on all rows strictly before tpos to avoid peeking
            X_fit = X_avail.iloc[:-1]
            y_fit = y_avail.iloc[:-1]
            if len(y_fit) < 24:
                continue
            m = OLS(y_fit, X_fit).fit()
            yhat = float(m.predict(X_avail.iloc[[-1]]))
            preds.append(yhat)
            idxs.append(avail.index[-1])

        if len(preds) == 0:
            st.warning("Not enough data to run the expanding-window evaluation.")
        else:
            oos_pred = pd.Series(preds, index=idxs, name="Model Δŝ")
            # Align with actual OOS targets
            y_oos = Z.loc[oos_pred.index, "dlog_s_lead1"]
            e_model = (y_oos - oos_pred)
            e_rw    = (y_oos - 0.0)  # RW predicts zero change

            msfe_model = float(np.mean(e_model**2))
            msfe_rw    = float(np.mean(e_rw**2))
            r2_os = 1.0 - msfe_model / msfe_rw if msfe_rw > 0 else np.nan

            st.metric("Out-of-sample R² (vs RW)", f"{r2_os:.3f}",
                      delta=f"MSFE(model)={msfe_model:.4f} | MSFE(RW)={msfe_rw:.4f}")

            # Simple DM-style t-test (no HAC for simplicity)
            d = (e_rw**2 - e_model**2).dropna()
            dm_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
            st.caption(f"Simple DM-style t-stat on loss diff (no HAC): {dm_t:.2f}  (rule of thumb: |t|>1.96 ≈ 5%)")

            # Plot cumulative level forecast vs actual levels (start from last train obs)
            # Build level path by cumulating predicted Δs (percent log changes)
            start_date = train.index[-1]
            s0 = data.loc[start_date, "log_s"]
            cum_model = (oos_pred/100.0).cumsum() + s0
            cum_rw    = s0 + 0.0*(oos_pred/100.0).cumsum()
            actual    = data["log_s"].loc[cum_model.index]

            fig2, ax2 = plt.subplots(figsize=(9,4))
            np.exp(actual).plot(ax=ax2, label="Actual spot level")
            np.exp(cum_model).plot(ax=ax2, label="Model-implied level path")
            np.exp(cum_rw).plot(ax=ax2, label="RW-implied level path")
            ax2.set_title("Out-of-sample: Levels implied by Δs forecasts")
            ax2.set_ylabel("Spot (level)")
            ax2.legend()
            st.pyplot(fig2)
    elif code and code != INSTRUCTOR_PASSCODE:
        st.error("Incorrect passcode.")
    elif len(test) < 12:
        st.info("Passcode ok, but test sample is too short for evaluation (need ≥ 12 observations).")

st.markdown(
    """
**Notes for students**
- We forecast next month’s exchange-rate *change* using lagged predictors.  
- The random walk is a tough benchmark: it says today’s best guess of next month’s level is today’s level (so expected change is zero).  
- Green OOS \(R^2\) means your model beats the random walk on average squared error; negative means it loses.
"""
)
