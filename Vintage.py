import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

st.title("Port Vintage Declaration Prediction")

st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

st.write("Raw data preview")
st.dataframe(df.head())

# Expected columns
required_cols = ["year", "month", "tmax", "tmin", "rain", "vintage"]

if not all(col in df.columns for col in required_cols):
    st.error("Spreadsheet must contain: year, month, tmax, tmin, rain, vintage")
    st.stop()

df["tmean"] = (df["tmax"] + df["tmin"]) / 2

base_temp = st.sidebar.slider("GDD Base Temperature", 5, 15, 10)

df["gdd"] = np.maximum(df["tmean"] - base_temp, 0)

# ----------------------------------
# Construct annual variables
# ----------------------------------

years = sorted(df.year.unique())
rows = []

for y in years:
    sub = df[df.year == y]
    prev = df[df.year == y - 1]

    row = {}
    row["year"] = y

    row["GDD_Apr_Sep"] = sub[sub.month.between(4, 9)]["gdd"].sum()
    row["Rain_Sep"] = sub[sub.month == 9]["rain"].sum()

    row["Temp_Jul_Aug"] = sub[sub.month.isin([7, 8])]["tmean"].mean()
    row["Temp_Jul"] = sub[sub.month == 7]["tmean"].mean()

    row["TempJul_x_RainSep"] = row["Temp_Jul"] * row["Rain_Sep"]
    row["TempJulAug_x_RainSep"] = row["Temp_Jul_Aug"] * row["Rain_Sep"]

    row["Rain_Apr_May"] = sub[sub.month.isin([4, 5])]["rain"].sum()
    row["Rain_Jun_Aug"] = sub[sub.month.isin([6, 7, 8])]["rain"].sum()
    row["Rain_Sep_Oct"] = sub[sub.month.isin([9, 10])]["rain"].sum()

    row["DTR_Aug_Sep"] = (
        sub[sub.month.isin([8, 9])]["tmax"] -
        sub[sub.month.isin([8, 9])]["tmin"]
    ).mean()

    row["Temp_Apr_Jun"] = sub[sub.month.isin([4, 5, 6])]["tmean"].mean()

    # Rain Oct-Feb
    rain_oct_dec = prev[prev.month.isin([10, 11, 12])]["rain"].sum()
    rain_jan_feb = sub[sub.month.isin([1, 2])]["rain"].sum()
    row["Rain_Oct_Feb"] = rain_oct_dec + rain_jan_feb

    # Average temperature from previous October to current August
    temp_oct_dec = prev[prev.month.isin([10, 11, 12])]["tmean"]
    temp_jan_aug = sub[sub.month.isin([1, 2, 3, 4, 5, 6, 7, 8])]["tmean"]
    temp_oct_aug = pd.concat([temp_oct_dec, temp_jan_aug])

    avg_temp_oct_aug = temp_oct_aug.mean()

    # Rainfall from previous October to current August
    rain_oct_dec = prev[prev.month.isin([10, 11, 12])]["rain"].sum()
    rain_jan_aug = sub[sub.month.isin([1, 2, 3, 4, 5, 6, 7, 8])]["rain"].sum()
    rain_oct_aug = rain_oct_dec + rain_jan_aug

    # Aridity index = 100 * average temperature Oct-Aug / rainfall Oct-Aug
    if rain_oct_aug > 0:
        row["Aridity_Index"] = 100 * avg_temp_oct_aug / rain_oct_aug
    else:
        row["Aridity_Index"] = np.nan

    # Tmax in June
    row["Tmax_June"] = sub[sub.month == 6]["tmax"].mean()

    # Tmax in June and July
    row["Tmax_June_July"] = sub[sub.month.isin([6, 7])]["tmax"].mean()

    # Aridity x September rain
    row["Aridity_x_RainSep"] = row["Aridity_Index"] * row["Rain_Sep"]

    # Rain in September squared
    row["RainSep_sq"] = row["Rain_Sep"] ** 2

    # Aridity x September rain squared
    row["Aridity_x_RainSep_sq"] = row["Aridity_Index"] * row["RainSep_sq"]

    # Max temperature in June
    row["Tmax_June"] = sub[sub.month == 6]["tmax"].mean()

    # Max temperature in July
    row["Tmax_July"] = sub[sub.month == 7]["tmax"].mean()
    
    row["GDD_Apr_Sep"] = sub[sub.month.between(4, 9)]["gdd"].sum()

    # square of growing degree days
    row["GDD_Apr_Sep_sq"] = row["GDD_Apr_Sep"] ** 2

    # Vintage outcome
    v = sub[sub.month == 1]["vintage"].values
    if len(v) == 0:
        row["vintage"] = np.nan
    else:
        row["vintage"] = v[0]

    rows.append(row)

year_df = pd.DataFrame(rows).dropna()

st.write("Constructed dataset")
st.dataframe(year_df)

# ----------------------------------
# Target selection
# ----------------------------------

target_mode = st.sidebar.radio(
    "Prediction Target",
    ["Classic only", "Classic + Non Classic"]
)

if target_mode == "Classic only":
    y = (year_df["vintage"] == 1).astype(int)
else:
    y = (year_df["vintage"] > 0).astype(int)

# ----------------------------------
# Variable selection
# ----------------------------------

st.sidebar.header("Choose predictors")

predictors = [
    "GDD_Apr_Sep",
    "GDD_Apr_Sep_sq",
    "Rain_Sep",
    "RainSep_sq",
    "Temp_Jul_Aug",
    "Temp_Jul",
    "TempJul_x_RainSep",
    "TempJulAug_x_RainSep",
    "Rain_Apr_May",
    "Rain_Jun_Aug",
    "Rain_Sep_Oct",
    "DTR_Aug_Sep",
    "Temp_Apr_Jun",
    "Rain_Oct_Feb",
    "Aridity_Index",
    "Aridity_x_RainSep",
    "Aridity_x_RainSep_sq",
    "Tmax_June",
    "Tmax_July"
]

selected = []

for p in predictors:
    if st.sidebar.checkbox(p, True):
        selected.append(p)

if len(selected) == 0:
    st.warning("Select at least one variable")
    st.stop()

X = year_df[selected]
X = sm.add_constant(X)

# ----------------------------------
# Run logit
# ----------------------------------

try:
    model = sm.Logit(y, X).fit(disp=0)
except Exception as e:
    st.error(f"Model could not be estimated: {e}")
    st.stop()

st.header("Model Estimates")

summary_table = pd.DataFrame({
    "coef": model.params,
    "pvalue": model.pvalues,
    "odds_ratio": np.exp(model.params)
})

st.dataframe(summary_table)

# ----------------------------------
# Summary statistics (averages)
# ----------------------------------

st.header("Average of Variables")

avg_table = year_df.mean(numeric_only=True).to_frame(name="Average")

st.dataframe(avg_table)

# ----------------------------------
# Predictions
# ----------------------------------

probs = model.predict(X)
pred = (probs > 0.5).astype(int)

accuracy = accuracy_score(y, pred)

try:
    auc = roc_auc_score(y, probs)
except Exception:
    auc = np.nan

st.header("Diagnostics")
st.write("Accuracy:", accuracy)
st.write("ROC AUC:", auc)
st.write("Pseudo R2:", model.prsquared)
st.write("AIC:", model.aic)
st.write("BIC:", model.bic)

# ----------------------------------
# Marginal effect of September rain
# evaluated at average variables
# ----------------------------------

means = year_df.mean()

rain = means["Rain_Sep"]
tempjul = means["Temp_Jul"]
tempjulaug = means["Temp_Jul_Aug"]
aridity = means["Aridity_Index"]

b = model.params

dz_drain = (
    b.get("Rain_Sep",0)
    + 2*b.get("RainSep_sq",0)*rain
    + b.get("TempJul_x_RainSep",0)*tempjul
    + b.get("TempJulAug_x_RainSep",0)*tempjulaug
    + b.get("Aridity_x_RainSep",0)*aridity
    + 2*b.get("Aridity_x_RainSep_sq",0)*aridity*rain
)

# predicted probability at averages
Xmean = sm.add_constant(pd.DataFrame([means[selected]]))
pmean = model.predict(Xmean)[0]

marginal_effect = pmean * (1 - pmean) * dz_drain

st.header("Marginal Effect of September Rain")
st.write("Marginal effect evaluated at average climate:", round(marginal_effect,4))

# ----------------------------------
# Plot probabilities over time
# ----------------------------------

st.header("Predicted probabilities over time")

plot_df = year_df.copy()
plot_df["prob"] = probs
plot_df["actual"] = y

# declared vintages only
declared = plot_df[plot_df["actual"] == 1].copy()

fig, ax = plt.subplots(figsize=(12, 6))

# blue line: fitted logit probabilities by year
ax.plot(
    plot_df["year"],
    plot_df["prob"],
    linewidth=2.5,
    label="Predicted probability"
)

# dashed line at 50%
ax.axhline(
    0.5,
    linestyle="--",
    linewidth=2,
    label="50% threshold"
)

# red dots only for declared vintages, at their predicted probabilities
ax.scatter(
    declared["year"],
    declared["prob"],
    color="red",
    s=90,
    zorder=3,
    label="Declared vintage"
)

ax.set_xlabel("Year")
ax.set_ylabel("Probability of Vintage")
ax.set_title("Predicted Probability of Classic Port Vintages")
ax.set_ylim(0, 1)
ax.legend()

st.pyplot(fig)


# ----------------------------------
# Misclassifications
# ----------------------------------

results = year_df.copy()
results["prob"] = probs
results["prediction"] = pred
results["actual"] = y

missed_vintages = results[(results.actual == 1) & (results.prediction == 0)]
false_vintages = results[(results.actual == 0) & (results.prediction == 1)]

st.header("Missed vintages")
st.dataframe(missed_vintages[["year", "prob"]])

st.header("Predicted vintages not declared")
st.dataframe(false_vintages[["year", "prob"]])

num_actual_vintages = (results["actual"] == 1).sum()
num_actual_nonvintages = (results["actual"] == 0).sum()

frac_vintages_misclassified = (
    len(missed_vintages) / num_actual_vintages
    if num_actual_vintages > 0 else np.nan
)

frac_nonvintages_misclassified = (
    len(false_vintages) / num_actual_nonvintages
    if num_actual_nonvintages > 0 else np.nan
)

st.header("Classification error rates")
st.write(
    "Fraction of vintages misclassified:",
    f"{frac_vintages_misclassified:.2f}"
)

st.write(
    "Fraction of non-vintages misclassified:",
    f"{frac_nonvintages_misclassified:.2f}"
)

