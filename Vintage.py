import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score

# -------------------------------------------------
# Page setup
# -------------------------------------------------

st.set_page_config(page_title="Port Vintage Logit Model", layout="wide")
st.title("Port Vintage Logistic Regression")

st.write("""
Upload an Excel file with monthly data. The file should contain these columns:

- `year`
- `month`
- `tmax`
- `tmin`
- `rain`
- `vintage`

where `vintage = 1` for classic vintage years and `0` otherwise.
""")

# -------------------------------------------------
# Upload file
# -------------------------------------------------

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload your Excel file to begin.")
    st.stop()

# -------------------------------------------------
# Read data
# -------------------------------------------------

df = pd.read_excel(uploaded_file)
df.columns = [str(c).strip().lower() for c in df.columns]

required_cols = ["year", "month", "tmax", "tmin", "rain", "vintage"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"The uploaded file is missing these required columns: {missing_cols}")
    st.stop()

for c in required_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Replace common missing-value codes
df.loc[df["rain"] <= -99, "rain"] = np.nan
df.loc[df["tmax"] <= -99, "tmax"] = np.nan
df.loc[df["tmin"] <= -99, "tmin"] = np.nan

# -------------------------------------------------
# Monthly climate variables
# -------------------------------------------------

df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
df["gdd_month"] = np.maximum(df["tmean"] - 10.0, 0.0)

# -------------------------------------------------
# Build yearly dataset
# -------------------------------------------------

rows = []
years = sorted(df["year"].dropna().astype(int).unique())

for y in years:
    d = df[df["year"] == y].copy()
    prev = df[df["year"] == y - 1].copy()

    vintage_vals = d["vintage"].dropna().unique()
    if len(vintage_vals) == 0:
        continue

    rows.append({
        "year": int(y),
        "vintage": int(vintage_vals[0]),
        "gdd_apr_sep": d.loc[d["month"].between(4, 9), "gdd_month"].sum(skipna=True),
        "rain_sep": d.loc[d["month"] == 9, "rain"].sum(skipna=True),
        "rain_apr_jun": d.loc[d["month"].between(4, 6), "rain"].sum(skipna=True),
        "temp_jul_aug": d.loc[d["month"].between(7, 8), "tmean"].mean(skipna=True),
        "rain_oct_feb": (
            prev.loc[prev["month"].between(10, 12), "rain"].sum(skipna=True)
            + d.loc[d["month"].between(1, 2), "rain"].sum(skipna=True)
        ),
        "temp_jul": d.loc[d["month"] == 7, "tmean"].mean(skipna=True),
        "temp_aug": d.loc[d["month"] == 8, "tmean"].mean(skipna=True),
        "rain_apr": d.loc[d["month"] == 4, "rain"].sum(skipna=True),
        "rain_may": d.loc[d["month"] == 5, "rain"].sum(skipna=True),
        "rain_jun": d.loc[d["month"] == 6, "rain"].sum(skipna=True),
    })

year_df = pd.DataFrame(rows)

if year_df.empty:
    st.error("No usable yearly data could be constructed from the uploaded file.")
    st.stop()

year_df["gdd_sq"] = year_df["gdd_apr_sep"] ** 2
year_df["aug_x_sep_rain"] = year_df["temp_aug"] * year_df["rain_sep"]

# -------------------------------------------------
# Model choices
# -------------------------------------------------

model_options = {
    "Best default model": [
        "gdd_apr_sep",
        "gdd_sq",
        "rain_sep",
        "rain_apr_jun",
        "temp_jul_aug",
        "rain_oct_feb"
    ],
    "Simple harvest model": [
        "rain_sep"
    ],
    "Monthly spring rain model": [
        "rain_apr",
        "rain_may",
        "rain_jun",
        "rain_sep"
    ],
    "July-August split model": [
        "rain_apr_jun",
        "temp_jul",
        "temp_aug",
        "rain_sep",
        "rain_oct_feb"
    ],
    "Interaction model": [
        "rain_apr_jun",
        "temp_jul",
        "temp_aug",
        "rain_sep",
        "aug_x_sep_rain",
        "rain_oct_feb"
    ]
}

model_name = st.selectbox(
    "Choose logistic regression specification",
    list(model_options.keys()),
    index=0
)

feature_cols = model_options[model_name]

# -------------------------------------------------
# Estimation sample
# -------------------------------------------------

model_df = year_df[["year", "vintage"] + feature_cols].dropna().copy()

if model_df.empty:
    st.error("After dropping missing values, no observations remain for this model.")
    st.stop()

X = model_df[feature_cols].copy()
y = model_df["vintage"].astype(int).copy()

# -------------------------------------------------
# Estimate logistic regression
# -------------------------------------------------

X_const = sm.add_constant(X, has_constant="add")

try:
    result = sm.Logit(y, X_const).fit(disp=False)
except Exception as e:
    st.error(f"Logit estimation failed: {e}")
    st.stop()

# -------------------------------------------------
# Predicted probabilities and classifications
# -------------------------------------------------

model_df["predicted_probability"] = result.predict(X_const)
threshold = 0.5
model_df["predicted_class"] = (model_df["predicted_probability"] >= threshold).astype(int)

# -------------------------------------------------
# Diagnostics
# -------------------------------------------------

auc = roc_auc_score(y, model_df["predicted_probability"]) if len(np.unique(y)) > 1 else np.nan
brier = brier_score_loss(y, model_df["predicted_probability"])
precision = precision_score(y, model_df["predicted_class"], zero_division=0)
recall = recall_score(y, model_df["predicted_class"], zero_division=0)

coef_table = pd.DataFrame({
    "Variable": result.params.index,
    "Coefficient": result.params.values,
    "Std. Error": result.bse.values,
    "z-stat": result.tvalues.values,
    "p-value": result.pvalues.values
})

diag_df = pd.DataFrame({
    "Metric": [
        "Pseudo R-squared",
        "Log-Likelihood",
        "LR test p-value",
        "ROC AUC",
        "Brier Score",
        "Precision",
        "Recall",
        "Observations"
    ],
    "Value": [
        result.prsquared,
        result.llf,
        result.llr_pvalue,
        auc,
        brier,
        precision,
        recall,
        len(model_df)
    ]
})

# -------------------------------------------------
# Display results
# -------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Estimated coefficients")
    st.dataframe(coef_table, use_container_width=True)

with col2:
    st.subheader("Model diagnostics")
    st.dataframe(diag_df, use_container_width=True)

# -------------------------------------------------
# Diagnostics explanation
# -------------------------------------------------

st.subheader("How to read the diagnostics")

st.markdown("""
- **Coefficient**: positive means the variable raises the probability of a classic vintage; negative means it lowers it.
- **Std. Error**: measures uncertainty in the coefficient estimate.
- **z-stat**: coefficient divided by its standard error.
- **p-value**: smaller values suggest stronger statistical evidence that the variable matters.

- **Pseudo R-squared**: a rough measure of fit for logistic regression. Higher is better, but it is not the same as OLS \(R^2\).
- **Log-Likelihood**: overall fit statistic; less negative is better.
- **LR test p-value**: tests whether the model as a whole improves on a constant-only model.
- **ROC AUC**: measures how well the model ranks vintage years above non-vintage years.
    - 0.50 = random
    - 0.60 = weak
    - 0.70 = acceptable
    - 0.80 = strong
- **Brier Score**: measures how accurate the predicted probabilities are. Lower is better.
- **Precision**: of the years predicted to be vintage, how many actually were.
- **Recall**: of the true vintage years, how many the model correctly identified.
""")

# -------------------------------------------------
# Predicted probabilities table
# -------------------------------------------------

st.subheader("Predicted probability of vintage by year")

display_cols = ["year", "vintage", "predicted_probability", "predicted_class"] + feature_cols
st.dataframe(
    model_df[display_cols].sort_values("year").reset_index(drop=True),
    use_container_width=True
)

# -------------------------------------------------
# Plot predicted probabilities
# -------------------------------------------------

st.subheader("Graph: predicted probability and actual vintage years")

fig, ax = plt.subplots(figsize=(10, 5))

# Blue line: predicted probability
ax.plot(
    model_df["year"],
    model_df["predicted_probability"],
    linewidth=2,
    label="Predicted probability"
)

# Red points: actual vintage years
vintage_points = model_df[model_df["vintage"] == 1]

ax.scatter(
    vintage_points["year"],
    vintage_points["predicted_probability"],
    s=80,
    label="Declared vintage"
)

# Threshold line
ax.axhline(0.5, linestyle="--", alpha=0.7)

ax.set_xlabel("Year")
ax.set_ylabel("Probability of Vintage")
ax.set_title("Predicted Probability of Classic Port Vintages")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Full regression summary
# -------------------------------------------------

st.subheader("Full regression summary")
st.text(result.summary())
