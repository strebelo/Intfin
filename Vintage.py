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

# ----------------------------------
# Monthly constructed variables
# ----------------------------------

df["tmean"] = (df["tmax"] + df["tmin"]) / 2

base_temp = st.sidebar.slider("GDD Base Temperature", 5, 15, 10)
df["gdd"] = np.maximum(df["tmean"] - base_temp, 0)

# ----------------------------------
# Construct annual variables
# ----------------------------------

years = sorted(df["year"].unique())
rows = []

for y in years:
    sub = df[df["year"] == y]
    prev = df[df["year"] == y - 1]

    row = {}
    row["year"] = y

    row["GDD_Apr_Sep"] = sub[sub["month"].between(4, 9)]["gdd"].sum()
    row["Rain_Sep"] = sub[sub["month"] == 9]["rain"].sum()

    row["Temp_Jul"] = sub[sub["month"] == 7]["tmean"].mean()
    row["Temp_Aug"] = sub[sub["month"] == 8]["tmean"].mean()
    row["Temp_Jul_Aug"] = sub[sub["month"].isin([7, 8])]["tmean"].mean()

    row["TempJul_x_RainSep"] = row["Temp_Jul"] * row["Rain_Sep"]
    row["TempAug_x_RainSep"] = row["Temp_Aug"] * row["Rain_Sep"]

    row["Rain_Apr_May"] = sub[sub["month"].isin([4, 5])]["rain"].sum()
    row["Rain_Jun_Aug"] = sub[sub["month"].isin([6, 7, 8])]["rain"].sum()
    row["Rain_Sep_Oct"] = sub[sub["month"].isin([9, 10])]["rain"].sum()

    row["DTR_Aug_Sep"] = (
        sub[sub["month"].isin([8, 9])]["tmax"] -
        sub[sub["month"].isin([8, 9])]["tmin"]
    ).mean()

    row["Temp_Apr_Jun"] = sub[sub["month"].isin([4, 5, 6])]["tmean"].mean()

    # Rain Oct-Feb
    rain_oct_dec_prev = prev[prev["month"].isin([10, 11, 12])]["rain"].sum()
    rain_jan_feb = sub[sub["month"].isin([1, 2])]["rain"].sum()
    row["Rain_Oct_Feb"] = rain_oct_dec_prev + rain_jan_feb

    # Average temperature from previous October to current August
    temp_oct_dec = prev[prev["month"].isin([10, 11, 12])]["tmean"]
    temp_jan_aug = sub[sub["month"].isin([1, 2, 3, 4, 5, 6, 7, 8])]["tmean"]
    temp_oct_aug = pd.concat([temp_oct_dec, temp_jan_aug])
    avg_temp_oct_aug = temp_oct_aug.mean()

    # Rainfall from previous October to current August
    rain_oct_dec = prev[prev["month"].isin([10, 11, 12])]["rain"].sum()
    rain_jan_aug = sub[sub["month"].isin([1, 2, 3, 4, 5, 6, 7, 8])]["rain"].sum()
    rain_oct_aug = rain_oct_dec + rain_jan_aug

    # Aridity index
    if rain_oct_aug > 0:
        row["Aridity_Index"] = 100 * avg_temp_oct_aug / rain_oct_aug
    else:
        row["Aridity_Index"] = np.nan

    # September rain squared
    row["RainSep_sq"] = row["Rain_Sep"] ** 2

    # Aridity interactions
    row["Aridity_x_RainSep"] = row["Aridity_Index"] * row["Rain_Sep"]
    row["Aridity_x_RainSep_sq"] = row["Aridity_Index"] * row["RainSep_sq"]

    # Max temperatures
    row["Tmax_June"] = sub[sub["month"] == 6]["tmax"].mean()
    row["Tmax_July"] = sub[sub["month"] == 7]["tmax"].mean()
    row["Tmax_August"] = sub[sub["month"] == 8]["tmax"].mean()
    row["Tmax_June_July"] = sub[sub["month"].isin([6, 7])]["tmax"].mean()

    # GDD squared
    row["GDD_Apr_Sep_sq"] = row["GDD_Apr_Sep"] ** 2

    # Vintage outcome
    v = sub[sub["month"] == 1]["vintage"].values
    row["vintage"] = v[0] if len(v) > 0 else np.nan

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
    # Only classic vintages count as 1
    y = (year_df["vintage"] == 1).astype(int)
else:
    # Any positive vintage code counts as 1
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
    "Temp_Jul",
    "Temp_Aug",
    "Temp_Jul_Aug",
    "TempJul_x_RainSep",
    "TempAug_x_RainSep",
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
    "Tmax_July",
    "Tmax_August",
    "Tmax_June_July"
]

default_selected = {
    "GDD_Apr_Sep",
    "GDD_Apr_Sep_sq",
    "Temp_Jul",
    "Temp_Aug",
    "Rain_Apr_May",
    "Rain_Oct_Feb",
    "Aridity_x_RainSep",
    "Tmax_July",
    "Tmax_August"
}

selected = []
for p in predictors:
    if st.sidebar.checkbox(p, value=(p in default_selected)):
        selected.append(p)

if len(selected) == 0:
    st.warning("Select at least one variable")
    st.stop()

# ----------------------------------
# Prediction settings
# ----------------------------------

st.sidebar.header("Prediction Settings")

threshold = st.sidebar.slider(
    "Vintage prediction threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

# ----------------------------------
# Build X and run logit
# ----------------------------------

X = year_df[selected].copy()
X = sm.add_constant(X, has_constant="add")

try:
    model = sm.Logit(y, X).fit(disp=0)
except Exception as e:
    st.error(f"Model could not be estimated: {e}")
    st.stop()

# ----------------------------------
# Model estimates
# ----------------------------------

st.header("Model Estimates")

summary_table = pd.DataFrame({
    "coef": model.params,
    "pvalue": model.pvalues,
    "odds_ratio": np.exp(model.params)
})

st.dataframe(summary_table)

# ----------------------------------
# Predictions
# ----------------------------------

probs = model.predict(X)
pred = (probs > threshold).astype(int)

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
st.write("Prediction threshold:", f"{threshold:.2f}")

# ----------------------------------
# Plot probabilities over time
# ----------------------------------

st.header("Predicted probabilities over time")

plot_df = year_df.copy()
plot_df["prob"] = probs
plot_df["actual"] = y

declared = plot_df[plot_df["actual"] == 1].copy()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    plot_df["year"],
    plot_df["prob"],
    linewidth=2.5,
    label="Predicted probability"
)

ax.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.2f}"
)

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
ax.set_title("Predicted Probability of Port Vintage Declaration")
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

missed_vintages = results[(results["actual"] == 1) & (results["prediction"] == 0)]
false_vintages = results[(results["actual"] == 0) & (results["prediction"] == 1)]

st.header("Missed vintages")
st.dataframe(missed_vintages[["year", "prob"]])

st.header("Predicted vintages not declared")
st.dataframe(false_vintages[["year", "prob"]])

# Counts
num_actual_vintages = (results["actual"] == 1).sum()
num_actual_nonvintages = (results["actual"] == 0).sum()

num_missed_vintages = len(missed_vintages)
num_false_vintages = len(false_vintages)

# Fractions
frac_vintages_misclassified = (
    num_missed_vintages / num_actual_vintages
    if num_actual_vintages > 0 else np.nan
)

frac_nonvintages_misclassified = (
    num_false_vintages / num_actual_nonvintages
    if num_actual_nonvintages > 0 else np.nan
)

# ----------------------------------
# Classification error rates
# ----------------------------------

st.header("Classification error rates")

st.write(
    f"Vintages misclassified: {num_missed_vintages} out of {num_actual_vintages} "
    f"({frac_vintages_misclassified:.2f})"
)

st.write(
    f"Non-vintages misclassified: {num_false_vintages} out of {num_actual_nonvintages} "
    f"({frac_nonvintages_misclassified:.2f})"
)

# ----------------------------------
# Means used for marginal effects
# ----------------------------------

means = X.drop(columns="const", errors="ignore").mean()

# ----------------------------------
# Marginal effect of September rain
# at Aridity = mean + 1 SD
# ----------------------------------

rain = means.get("Rain_Sep", year_df["Rain_Sep"].mean())
tempjul = means.get("Temp_Jul", year_df["Temp_Jul"].mean())
tempaug = means.get("Temp_Aug", year_df["Temp_Aug"].mean())

A_1sd = year_df["Aridity_Index"].mean() + year_df["Aridity_Index"].std()

dz_drain_1sd = (
    model.params.get("Rain_Sep", 0.0)
    + 2 * model.params.get("RainSep_sq", 0.0) * rain
    + model.params.get("TempJul_x_RainSep", 0.0) * tempjul
    + model.params.get("TempAug_x_RainSep", 0.0) * tempaug
    + model.params.get("Aridity_x_RainSep", 0.0) * A_1sd
    + 2 * model.params.get("Aridity_x_RainSep_sq", 0.0) * A_1sd * rain
)

X1_dict = {}
for col in X.columns:
    if col == "const":
        X1_dict[col] = 1.0
    else:
        X1_dict[col] = means.get(col, year_df[col].mean() if col in year_df.columns else 0.0)

# Override components needed for this counterfactual point
if "Rain_Sep" in X1_dict:
    X1_dict["Rain_Sep"] = rain
if "Temp_Jul" in X1_dict:
    X1_dict["Temp_Jul"] = tempjul
if "Temp_Aug" in X1_dict:
    X1_dict["Temp_Aug"] = tempaug
if "RainSep_sq" in X1_dict:
    X1_dict["RainSep_sq"] = rain ** 2
if "Aridity_Index" in X1_dict:
    X1_dict["Aridity_Index"] = A_1sd
if "TempJul_x_RainSep" in X1_dict:
    X1_dict["TempJul_x_RainSep"] = tempjul * rain
if "TempAug_x_RainSep" in X1_dict:
    X1_dict["TempAug_x_RainSep"] = tempaug * rain
if "Aridity_x_RainSep" in X1_dict:
    X1_dict["Aridity_x_RainSep"] = A_1sd * rain
if "Aridity_x_RainSep_sq" in X1_dict:
    X1_dict["Aridity_x_RainSep_sq"] = A_1sd * (rain ** 2)

X1 = pd.DataFrame([X1_dict])[X.columns]
p1 = model.predict(X1).iloc[0]
ME_rain_1sd = p1 * (1 - p1) * dz_drain_1sd

st.header("Marginal Effect of September Rain at Aridity = Mean + 1 SD")
st.write(f"{ME_rain_1sd:.4f}")

# ----------------------------------
# Marginal effect of September rain
# at Aridity = mean + 2 SD
# ----------------------------------

A_2sd = year_df["Aridity_Index"].mean() + 2 * year_df["Aridity_Index"].std()

dz_drain_2sd = (
    model.params.get("Rain_Sep", 0.0)
    + 2 * model.params.get("RainSep_sq", 0.0) * rain
    + model.params.get("TempJul_x_RainSep", 0.0) * tempjul
    + model.params.get("TempAug_x_RainSep", 0.0) * tempaug
    + model.params.get("Aridity_x_RainSep", 0.0) * A_2sd
    + 2 * model.params.get("Aridity_x_RainSep_sq", 0.0) * A_2sd * rain
)

X2_dict = X1_dict.copy()
if "Aridity_Index" in X2_dict:
    X2_dict["Aridity_Index"] = A_2sd
if "Aridity_x_RainSep" in X2_dict:
    X2_dict["Aridity_x_RainSep"] = A_2sd * rain
if "Aridity_x_RainSep_sq" in X2_dict:
    X2_dict["Aridity_x_RainSep_sq"] = A_2sd * (rain ** 2)

X2 = pd.DataFrame([X2_dict])[X.columns]
p2 = model.predict(X2).iloc[0]
ME_rain_2sd = p2 * (1 - p2) * dz_drain_2sd

st.header("Marginal Effect of September Rain at Aridity = Mean + 2 SD")
st.write(f"{ME_rain_2sd:.4f}")

# ----------------------------------
# General marginal effects table
# evaluated at the mean
# with interactions and quadratic terms handled properly
# ----------------------------------

def linear_index_derivative(var_name, params, means_dict):
    d = 0.0

    # Direct effect
    d += params.get(var_name, 0.0)

    # Own square terms
    square_map = {
        "Rain_Sep": "RainSep_sq",
        "GDD_Apr_Sep": "GDD_Apr_Sep_sq"
    }

    if var_name in square_map:
        sq_name = square_map[var_name]
        d += 2 * params.get(sq_name, 0.0) * means_dict.get(var_name, 0.0)

    # Standard interaction terms
    interaction_map = {
        "Rain_Sep": [
            ("TempJul_x_RainSep", "Temp_Jul"),
            ("TempAug_x_RainSep", "Temp_Aug"),
            ("Aridity_x_RainSep", "Aridity_Index")
        ],
        "Temp_Jul": [
            ("TempJul_x_RainSep", "Rain_Sep")
        ],
        "Temp_Aug": [
            ("TempAug_x_RainSep", "Rain_Sep")
        ],
        "Aridity_Index": [
            ("Aridity_x_RainSep", "Rain_Sep")
        ]
    }

    if var_name in interaction_map:
        for interaction_term, other_var in interaction_map[var_name]:
            d += params.get(interaction_term, 0.0) * means_dict.get(other_var, 0.0)

    # Interaction involving Aridity * Rain_Sep^2
    if var_name == "Rain_Sep":
        d += (
            2
            * params.get("Aridity_x_RainSep_sq", 0.0)
            * means_dict.get("Aridity_Index", 0.0)
            * means_dict.get("Rain_Sep", 0.0)
        )

    if var_name == "Aridity_Index":
        d += (
            params.get("Aridity_x_RainSep_sq", 0.0)
            * (means_dict.get("Rain_Sep", 0.0) ** 2)
        )

    return d

# Build mean covariate vector consistent with X
Xmean_dict = {}
for col in X.columns:
    if col == "const":
        Xmean_dict[col] = 1.0
    else:
        Xmean_dict[col] = means.get(col, 0.0)

# Keep constructed terms internally consistent if they are present
rain_mean = Xmean_dict.get("Rain_Sep", means.get("Rain_Sep", 0.0))
tempjul_mean = Xmean_dict.get("Temp_Jul", means.get("Temp_Jul", 0.0))
tempaug_mean = Xmean_dict.get("Temp_Aug", means.get("Temp_Aug", 0.0))
aridity_mean = Xmean_dict.get("Aridity_Index", means.get("Aridity_Index", 0.0))
gdd_mean = Xmean_dict.get("GDD_Apr_Sep", means.get("GDD_Apr_Sep", 0.0))

if "RainSep_sq" in Xmean_dict:
    Xmean_dict["RainSep_sq"] = rain_mean ** 2
if "GDD_Apr_Sep_sq" in Xmean_dict:
    Xmean_dict["GDD_Apr_Sep_sq"] = gdd_mean ** 2
if "TempJul_x_RainSep" in Xmean_dict:
    Xmean_dict["TempJul_x_RainSep"] = tempjul_mean * rain_mean
if "TempAug_x_RainSep" in Xmean_dict:
    Xmean_dict["TempAug_x_RainSep"] = tempaug_mean * rain_mean
if "Aridity_x_RainSep" in Xmean_dict:
    Xmean_dict["Aridity_x_RainSep"] = aridity_mean * rain_mean
if "Aridity_x_RainSep_sq" in Xmean_dict:
    Xmean_dict["Aridity_x_RainSep_sq"] = aridity_mean * (rain_mean ** 2)

Xmean = pd.DataFrame([Xmean_dict])[X.columns]
p_at_mean = model.predict(Xmean).iloc[0]

base_variables_for_me = [
    "GDD_Apr_Sep",
    "Rain_Sep",
    "Temp_Jul",
    "Temp_Aug",
    "Temp_Jul_Aug",
    "Rain_Apr_May",
    "Rain_Jun_Aug",
    "Rain_Sep_Oct",
    "DTR_Aug_Sep",
    "Temp_Apr_Jun",
    "Rain_Oct_Feb",
    "Aridity_Index",
    "Tmax_June",
    "Tmax_July",
    "Tmax_August",
    "Tmax_June_July"
]

me_rows = []
for var in base_variables_for_me:
    if var in year_df.columns:
        dzdx = linear_index_derivative(var, model.params, Xmean_dict)
        me = p_at_mean * (1 - p_at_mean) * dzdx

        me_rows.append({
            "variable": var,
            "mean_value": Xmean_dict.get(var, np.nan),
            "dz_dx_at_mean": dzdx,
            "marginal_effect_at_mean": me
        })

me_table = pd.DataFrame(me_rows)

if not me_table.empty:
    me_table["mean_value"] = me_table["mean_value"].round(3)
    me_table["dz_dx_at_mean"] = me_table["dz_dx_at_mean"].round(4)
    me_table["marginal_effect_at_mean"] = me_table["marginal_effect_at_mean"].round(4)
    me_table = me_table.sort_values(
        "marginal_effect_at_mean",
        key=lambda s: np.abs(s),
        ascending=False
    )

    st.header("Marginal Effects Table")
    st.write("All controls evaluated at their sample means. Interaction and quadratic terms are taken into account.")
    st.dataframe(me_table, use_container_width=True)
