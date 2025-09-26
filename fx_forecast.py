# ... [imports and helpers unchanged above] ...

# ---- Variable selection ----
all_cols = list(df.columns)
if not all_cols:
    st.error("Your file has no usable columns after parsing.")
    st.stop()

target = st.selectbox("Dependent variable (exchange rate)", all_cols)
exog_choices = [c for c in all_cols if c != target]

# Note the "(optional)" label now
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

# exogs are truly optional now
X_exog = df[exogs].astype(float) if exogs else pd.DataFrame(index=df.index)

def make_target_lags(series, L, name):
    if L <= 0:
        return pd.DataFrame(index=series.index)
    cols = {f"{name}_lag{l}": series.shift(l) for l in range(1, L + 1)}
    return pd.DataFrame(cols, index=series.index)

X_ar = make_target_lags(Y, ar_lags, name=target)
X = pd.concat([X_exog, X_ar], axis=1)

# Align and drop NaNs introduced by lags
data = pd.concat([Y.rename(target), X], axis=1).dropna()
if data.empty:
    st.error("After adding AR lags and dropping NaNs, no rows remain. Try fewer lags or check data.")
    st.stop()

Y = data[target].astype(float)
X = data.drop(columns=[target]).astype(float)

# Add constant and estimate (unchanged)
if HAS_SM:
    X = sm.add_constant(X, has_constant='add')
else:
    X = add_constant_df(X)

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

# ... [R², OOS test, chart, footer unchanged below] ...
