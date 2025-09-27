import streamlit as st

ENABLE_OUT_OF_SAMPLE = True   # change to True when you want the program to perform out of sample tests

# ------------------ Safe imports ---------------------------------------------
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas"); pd = None
try:
    import numpy as np
except Exception:
    missing.append("numpy"); np = None

HAS_SM = True
try:
    import statsmodels.api as sm
except Exception:
    HAS_SM = False

HAS_OPENPYXL = True
try:
    import openpyxl  # noqa
except Exception:
    HAS_OPENPYXL = False

HAS_XLRD = True
try:
    import xlrd  # noqa
except Exception:
    HAS_XLRD = False

st.set_page_config(page_title="FX Forecast", layout="wide")
st.title("Exchange-Rate Forecasting App")

if any(m in ("pandas", "numpy") for m in missing):
    st.error("Missing required packages: " + ", ".join(missing))
    st.stop()

# ------------------ Utilities -------------------------------------------------
def add_constant_df(Xdf):
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
        return "\n".join(f"{k}: {v:.6g}" for k, v in self.params.items())

def np_ols_fit(y, X_df):
    X = X_df.values
    y = y.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return SimpleOLSResult(beta, X_df.columns)

def coerce_numeric_columns(df, exclude=None):
    exclude = set([] if exclude is None else exclude)
    for c in df.columns:
        if c in exclude:
            continue
        df[c] = df[c].replace({None: np.nan})
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def detect_probable_codes_row(df, date_col="date"):
    if df.empty:
        return False
    non_date_cols = [c for c in df.columns if c != date_col]
    row0 = df.iloc[0]
    nonnum = sum(
        1 for c in non_date_cols
        if pd.notna(row0[c]) and not str(row0[c]).replace(".", "", 1).isdigit()
    )
    return nonnum >= max(1, len(non_date_cols) // 3)

def make_target_lags(series, L, name):
    if L <= 0:
        return pd.DataFrame(index=series.index)
    return pd.concat({f"{name}_lag{l}": series.shift(l) for l in range(1, L + 1)}, axis=1)

# ------------------ Instructions ---------------------------------------------
st.write("""
**File format (monthly):**
1) **Row 1**: variable names (e.g., `date`, `spot exchange rate`, `inflation U.S.`, …)  
2) **Row 2** *(optional)*: variable codes  
3) **Row 3+**: monthly observations  
The first column can be `date` (YYYY-MM or YYYY-MM-DD).
""")

# ------------------ Upload ---------------------------------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.stop()

try:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, header=0)
    elif name.endswith(".xlsx"):
        if not HAS_OPENPYXL:
            st.error("`openpyxl` not installed."); st.stop()
        df = pd.read_excel(uploaded, engine="openpyxl", header=0)
    else:  # .xls
        if not HAS_XLRD:
            st.error("`xlrd` not installed."); st.stop()
        df = pd.read_excel(uploaded, engine="xlrd", header=0)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# ------------------ Optional codes row ---------------------------------------
heuristic_codes = detect_probable_codes_row(df, date_col="date" if "date" in df.columns else None)
use_codes_row = st.checkbox("Row 2 contains variable codes", value=heuristic_codes)

codes_map = {}
if use_codes_row and len(df) >= 1:
    codes_series = df.iloc[0]
    for c in df.columns:
        if str(c).lower() != "date":
            codes_map[str(c)] = str(codes_series[c])
    df = df.iloc[1:].reset_index(drop=True)

# ------------------ Date handling: single clean 'date' column -----------------
date_candidates = [c for c in df.columns if str(c).strip().lower() in {"date", "month", "period"}]
date_col = date_candidates[0] if date_candidates else None

if date_col:
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if parsed.notna().any():
        # replace with formatted string YYYY-MM
        df[date_col] = parsed.dt.strftime("%Y-%m")
    else:
        st.warning("Could not parse dates; keeping original text.")
    # ensure the column is exactly named 'date'
    if date_col != "date":
        df.rename(columns={date_col: "date"}, inplace=True)
else:
    st.info("No explicit date column found (expected one of: date, month, period).")

# ------------------ Numeric coercion -----------------------------------------
df = coerce_numeric_columns(df, exclude={"date"})

# ------------------ Preview ---------------------------------------------------
st.write("Preview (first 6 rows):")
st.dataframe(df.head(6).fillna(""))

if codes_map:
    with st.sidebar.expander("Variable codes (from row 2)"):
        st.write(pd.DataFrame({"variable": list(codes_map.keys()),
                               "code": [codes_map[k] for k in codes_map]}))

# ------------------ Variable selection ---------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found after parsing.")
    st.stop()

target = st.selectbox("Dependent variable (exchange rate)",
                      numeric_cols,
                      index=0)
exog_choices = [c for c in numeric_cols if c != target]
exogs = st.multiselect("Independent variables (optional)", exog_choices, default=[])

# ------------------ AR lags --------------------------------------------------
ar_lags = st.slider("Number of AR lags on the exchange rate", 0, 12, 0)
if (not exogs) and ar_lags == 0:
    st.warning("Select at least 1 AR lag when no independent variables are chosen.")
    st.stop()

Y = df[target].astype(float)
X_exog = df[exogs].astype(float) if exogs else pd.DataFrame(index=df.index)
X_ar = make_target_lags(Y, ar_lags, name=target)
X = pd.concat([X_exog, X_ar], axis=1)

data = pd.concat([Y.rename(target), X], axis=1).replace({None: np.nan}).dropna()
if data.empty:
    st.error("No rows remain after adding AR lags and dropping NaNs.")
    st.stop()

Y = data[target].astype(float)
X = data.drop(columns=[target]).astype(float)

if HAS_SM:
    X = sm.add_constant(X, has_constant='add')
else:
    X = add_constant_df(X)

# ------------------ Regression -----------------------------------------------
st.subheader("Regression Results")
if HAS_SM:
    fit = sm.OLS(Y, X).fit()
    st.text(fit.summary().as_text())
else:
    fit = np_ols_fit(Y, X)
    st.text(fit.text_summary())

yhat_in = pd.Series(fit.predict(X), index=Y.index, dtype=float)
ss_res = float(np.sum((Y - yhat_in) ** 2))
ss_tot = float(np.sum((Y - Y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
st.metric("In-sample R²", f"{r2:.4f}")

# ======================= Out-of-sample vs Random Walk =======================
if ENABLE_OUT_OF_SAMPLE:
    st.subheader("Out-of-Sample Forecast Test")

    ratio = st.slider("Training fraction", 0.5, 0.95, 0.8)
    split = int(len(Y) * ratio)

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

    fit_oos = sm.OLS(Y_tr, X_tr).fit() if HAS_SM else np_ols_fit(Y_tr, X_tr)
    pred = pd.Series(fit_oos.predict(X_te), index=X_te.index, dtype=float)
    rw = Y.shift(1).iloc[split:].astype(float)

    pred, Y_te = pred.align(Y_te, join="inner")
    rw, Y_te  = rw.align(Y_te,  join="inner")

    mse_model = float(np.mean((pred - Y_te) ** 2))
    mse_rw    = float(np.mean((rw   - Y_te) ** 2))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("OOS MSE — Model", f"{mse_model:.6g}")
    with c2:
        st.metric("OOS MSE — Random Walk", f"{mse_rw:.6g}")

    if np.isfinite(mse_model) and np.isfinite(mse_rw):
        if mse_model < mse_rw:
            imp = (mse_rw - mse_model) / mse_rw * 100 if mse_rw > 0 else np.nan
            st.success(f"✅ Model beats the random walk (MSE ↓ {imp:.2f}%).")
        elif mse_model > mse_rw:
            det = (mse_model - mse_rw) / mse_rw * 100 if mse_rw > 0 else np.nan
            st.warning(f"⚠️ Random walk performs better (Model MSE ↑ {det:.2f}% vs RW).")
        else:
            st.info("⚖️ Tie: same MSE.")
    else:
        st.info("Results not comparable due to insufficient or non-finite values.")

import altair as alt
import numpy as np

# --- Align the three series as before ---
common_idx = Y_te.index.intersection(pred.index).intersection(rw.index)
Y_te_c = pd.to_numeric(Y_te.reindex(common_idx), errors="coerce")
pred_c = pd.to_numeric(pred.reindex(common_idx), errors="coerce")
rw_c   = pd.to_numeric(rw.reindex(common_idx),  errors="coerce")

chart_df = pd.DataFrame({
    "Actual":      Y_te_c,
    "Model":       pred_c,
    "Random walk": rw_c
}).dropna()
chart_df.index = chart_df.index.astype(str)

# --- Compute tight vertical range ---
vmin = chart_df.min().min()
vmax = chart_df.max().max()
tiny = (vmax - vmin) * 0.01 or 0.01  # 1 % of range or 0.01 if flat
domain = [vmin - tiny, vmax + tiny]

# --- Prepare long form for Altair ---
plot_df = chart_df.reset_index(names="date").melt(
    "date", var_name="Series", value_name="Value"
)

# --- Altair chart with fixed y-domain ---
line = (
    alt.Chart(plot_df)
       .mark_line()
       .encode(
           x=alt.X("date:T", title="Date"),
           y=alt.Y("Value:Q", scale=alt.Scale(domain=domain), title="Exchange rate"),
           color="Series:N",
           tooltip=["date:T","Series:N","Value:Q"]
       )
       .properties(width="container")
       .interactive()
)

st.altair_chart(line, use_container_width=True)
