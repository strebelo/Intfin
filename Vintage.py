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
required_cols = ["year","month","tmax","tmin","rain","vintage"]

if not all(col in df.columns for col in required_cols):
    st.error("Spreadsheet must contain: year, month, tmax, tmin, rain, vintage")
    st.stop()

df["tmean"] = (df["tmax"] + df["tmin"]) / 2

base_temp = st.sidebar.slider("GDD Base Temperature", 5,15,10)

df["gdd"] = np.maximum(df["tmean"] - base_temp,0)

# ----------------------------------
# Construct annual variables
# ----------------------------------

years = sorted(df.year.unique())
rows = []

for y in years:

    sub = df[df.year == y]

    prev = df[df.year == y-1]

    row = {}

    row["year"] = y

    row["GDD_Apr_Sep"] = sub[sub.month.between(4,9)]["gdd"].sum()

    row["Rain_Sep"] = sub[sub.month == 9]["rain"].sum()

    row["Temp_Jul_Aug"] = sub[sub.month.isin([7,8])]["tmean"].mean()

    row["Temp_Jul"] = sub[sub.month == 7]["tmean"].mean()

    row["Rain_Apr_May"] = sub[sub.month.isin([4,5])]["rain"].sum()

    row["Rain_Jun_Aug"] = sub[sub.month.isin([6,7,8])]["rain"].sum()

    row["Rain_Sep_Oct"] = sub[sub.month.isin([9,10])]["rain"].sum()

    row["DTR_Aug_Sep"] = (sub[sub.month.isin([8,9])]["tmax"] -
                         sub[sub.month.isin([8,9])]["tmin"]).mean()

    row["Temp_Apr_Jun"] = sub[sub.month.isin([4,5,6])]["tmean"].mean()

    # Rain Oct-Feb
    rain_oct_dec = prev[prev.month.isin([10,11,12])]["rain"].sum()
    rain_jan_feb = sub[sub.month.isin([1,2])]["rain"].sum()

    row["Rain_Oct_Feb"] = rain_oct_dec + rain_jan_feb

    # Vintage outcome
    v = sub[sub.month==1]["vintage"].values

    if len(v)==0:
        row["vintage"] = np.nan
    else:
        row["vintage"] = v[0]

    rows.append(row)

year_df = pd.DataFrame(rows)

year_df = year_df.dropna()

st.write("Constructed dataset")
st.dataframe(year_df)

# ----------------------------------
# Target selection
# ----------------------------------

target_mode = st.sidebar.radio(
"Prediction Target",
["Classic only","Classic + Non Classic"]
)

if target_mode == "Classic only":
    y = (year_df["vintage"]==1).astype(int)

else:
    y = (year_df["vintage"]>0).astype(int)

# ----------------------------------
# Variable selection
# ----------------------------------

st.sidebar.header("Choose predictors")

predictors = [
"GDD_Apr_Sep",
"Rain_Sep",
"Temp_Jul_Aug",
"Temp_Jul",
"Rain_Apr_May",
"Rain_Jun_Aug",
"Rain_Sep_Oct",
"DTR_Aug_Sep",
"Temp_Apr_Jun",
"Rain_Oct_Feb"
]

selected = []

for p in predictors:

    if st.sidebar.checkbox(p,True):
        selected.append(p)

if len(selected)==0:
    st.warning("Select at least one variable")
    st.stop()

X = year_df[selected]

X = sm.add_constant(X)

# ----------------------------------
# Run logit
# ----------------------------------

model = sm.Logit(y,X).fit(disp=0)

st.header("Model Estimates")

summary_table = pd.DataFrame({
"coef":model.params,
"pvalue":model.pvalues,
"odds_ratio":np.exp(model.params)
})

st.dataframe(summary_table)

# ----------------------------------
# Predictions
# ----------------------------------

probs = model.predict(X)

pred = (probs > 0.5).astype(int)

accuracy = accuracy_score(y,pred)

try:
    auc = roc_auc_score(y,probs)
except:
    auc = np.nan

st.header("Diagnostics")

st.write("Accuracy:",accuracy)

st.write("ROC AUC:",auc)

st.write("Pseudo R2:",model.prsquared)

st.write("AIC:",model.aic)

st.write("BIC:",model.bic)

# ----------------------------------
# Plot probabilities
# ----------------------------------

fig, ax = plt.subplots()

ax.plot(year_df["year"],probs,label="Predicted probability")

ax.scatter(year_df["year"],y,color="red",label="Actual vintage")

ax.set_ylabel("Probability")

ax.legend()

st.pyplot(fig)

# ----------------------------------
# Misclassifications
# ----------------------------------

results = year_df.copy()

results["prob"] = probs

results["prediction"] = pred

results["actual"] = y

missed_vintages = results[(results.actual==1) & (results.prediction==0)]

false_vintages = results[(results.actual==0) & (results.prediction==1)]

st.header("Missed vintages")

st.dataframe(missed_vintages[["year","prob"]])

st.header("Predicted vintages not declared")

st.dataframe(false_vintages[["year","prob"]])
