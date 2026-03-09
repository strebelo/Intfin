import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Port Vintage Prediction - Random Forest", layout="wide")

st.title("Port Vintage Declaration Prediction - Random Forest")

st.markdown(
    """
Upload a monthly climate Excel file with columns:

- `year`
- `month`
- `tmax`
- `tmin`
- `rain`
- `vintage`

where vintage is coded as:

- `1` = classic vintage
- `0.75` = non-classic vintage
- `0` = no vintage
"""
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

target_mode = st.sidebar.radio(
    "Target variable",
    ["Classic only", "Classic + non-classic"],
    index=0
)

gdd_threshold = st.sidebar.slider(
    "Growing Degree Days threshold (°C)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

n_estimators = st.sidebar.slider(
    "Number of trees",
    min_value=50,
    max_value=1000,
    value=300,
    step=50
)

max_depth = st.sidebar.selectbox(
    "Max tree depth",
    options=[None, 2, 3, 4, 5, 6, 8, 10],
    index=0
)

min_samples_leaf = st.sidebar.slider(
    "Minimum samples per leaf",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

# -----------------------------
# Helper functions
# -----------------------------
def prepare_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["year", "month", "tmax", "tmin", "rain", "vintage"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
    df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce")
    df["vintage"] = pd.to_numeric(df["vintage"], errors="coerce")

    # Replace rainfall missing code if present
    df["rain"] = df["rain"].replace(-99.9, np.nan)

    # Basic checks
    df = df.dropna(subset=["year", "month"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return df


def annual_from_monthly(df: pd.DataFrame, gdd_base: float) -> pd.DataFrame:
    df = df.copy()
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2
    df["dtr"] = df["tmax"] - df["tmin"]

    annual_rows = []

    years = sorted(df["year"].unique())

    for y in years:
        this_year = df[df["year"] == y].copy()

        # Previous year for Oct-Dec when constructing Oct-Feb rain
        prev_year = df[df["year"] == y - 1].copy()

        if this_year.empty:
            continue

        def avg_temp(months):
            tmp = this_year[this_year["month"].isin(months)]["tmean"]
            return tmp.mean() if len(tmp) > 0 else np.nan

        def sum_rain(months):
            tmp = this_year[this_year["month"].isin(months)]["rain"]
            return tmp.sum(min_count=1) if len(tmp) > 0 else np.nan

        # GDD: Apr-Sep using monthly mean temp above threshold
        gdd_months = this_year[this_year["month"].isin([4, 5, 6, 7, 8, 9])].copy()
        if len(gdd_months) > 0:
            gdd = np.maximum(gdd_months["tmean"] - gdd_base, 0).sum()
        else:
            gdd = np.nan

        # Rain Oct-Feb for vintage year y:
        # Oct-Dec from previous year, Jan-Feb from current year
        oct_dec_prev = prev_year[prev_year["month"].isin([10, 11, 12])]["rain"]
        jan_feb_this = this_year[this_year["month"].isin([1, 2])]["rain"]
        rain_oct_feb = pd.concat([oct_dec_prev, jan_feb_this]).sum(min_count=1)

        # Annual target: should be constant within year
        vintage_values = this_year["vintage"].dropna().unique()
        vintage_value = vintage_values[0] if len(vintage_values) > 0 else np.nan

        row = {
            "year": y,
            "gdd_apr_sep": gdd,
            "rain_sep": sum_rain([9]),
            "temp_jul_aug": avg_temp([7, 8]),
            "temp_jul": avg_temp([7]),
            "rain_apr_may": sum_rain([4, 5]),
            "rain_jun_aug": sum_rain([6, 7, 8]),
            "rain_sep_oct": sum_rain([9, 10]),
            "dtr_aug_sep": this_year[this_year["month"].isin([8, 9])]["dtr"].mean(),
            "temp_apr_jun": avg_temp([4, 5, 6]),
            "rain_oct_feb": rain_oct_feb,
            "vintage_value": vintage_value,
        }
        annual_rows.append(row)

    annual_df = pd.DataFrame(annual_rows).sort_values("year").reset_index(drop=True)
    return annual_df


def build_target(df_annual: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "Classic only":
        return (df_annual["vintage_value"] == 1).astype(int)
    else:
        return (df_annual["vintage_value"] > 0).astype(int)


def loo_random_forest_predictions(X: pd.DataFrame, y: pd.Series, params: dict):
    loo = LeaveOneOut()
    probs = np.zeros(len(X))
    preds = np.zeros(len(X), dtype=int)

    imputer = SimpleImputer(strategy="median")

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        model = RandomForestClassifier(**params)
        model.fit(X_train_imp, y_train)

        probs[test_idx[0]] = model.predict_proba(X_test_imp)[0, 1]
        preds[test_idx[0]] = int(probs[test_idx[0]] >= 0.5)

    return probs, preds


# -----------------------------
# Main app
# -----------------------------
if uploaded_file is None:
    st.info("Please upload your Excel file in the sidebar.")
    st.stop()

try:
    raw_df = pd.read_excel(uploaded_file)
    df = prepare_monthly_data(raw_df)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df.head(12))

# Annual variables
annual_df = annual_from_monthly(df, gdd_threshold)

feature_map = {
    "Growing Degree Days April-September": "gdd_apr_sep",
    "Rain September": "rain_sep",
    "Temperature July-August": "temp_jul_aug",
    "Temperature July": "temp_jul",
    "Rain April-May": "rain_apr_may",
    "Rain June-August": "rain_jun_aug",
    "Rain September-October": "rain_sep_oct",
    "Diurnal temperature range August-September": "dtr_aug_sep",
    "Temperature April-June": "temp_apr_jun",
    "Rain October-February": "rain_oct_feb",
}

default_features = [
    "Growing Degree Days April-September",
    "Rain September",
    "Temperature July-August",
    "Rain September-October",
]

selected_feature_labels = st.sidebar.multiselect(
    "Choose explanatory variables",
    options=list(feature_map.keys()),
    default=default_features,
)

if len(selected_feature_labels) == 0:
    st.warning("Please select at least one explanatory variable.")
    st.stop()

selected_features = [feature_map[x] for x in selected_feature_labels]

annual_df["target"] = build_target(annual_df, target_mode)

model_df = annual_df[["year", "target", "vintage_value"] + selected_features].copy()

st.subheader("Annual dataset used in the model")
st.dataframe(model_df)

# Drop rows with missing target
model_df = model_df.dropna(subset=["target"]).reset_index(drop=True)

X = model_df[selected_features].copy()
y = model_df["target"].astype(int)

# Need at least two classes
if y.nunique() < 2:
    st.error("The target variable has only one class in the current sample.")
    st.stop()

rf_params = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_leaf": min_samples_leaf,
    "random_state": random_state,
    "class_weight": "balanced",
}

# Leave-one-out predictions for honest out-of-sample diagnostics
probs_loo, preds_loo = loo_random_forest_predictions(X, y, rf_params)

# Fit full-sample model for feature importances
imputer_full = SimpleImputer(strategy="median")
X_imp_full = imputer_full.fit_transform(X)

rf_full = RandomForestClassifier(**rf_params)
rf_full.fit(X_imp_full, y)

# Metrics
acc = accuracy_score(y, preds_loo)
auc = roc_auc_score(y, probs_loo)
cm = confusion_matrix(y, preds_loo)

# Results table
results_df = model_df[["year", "target", "vintage_value"]].copy()
results_df["pred_prob"] = probs_loo
results_df["pred_class"] = preds_loo
results_df["actual_label"] = np.where(results_df["target"] == 1, "Vintage", "No vintage")
results_df["pred_label"] = np.where(results_df["pred_class"] == 1, "Vintage", "No vintage")
results_df["correct"] = results_df["target"] == results_df["pred_class"]

# -----------------------------
# Diagnostics
# -----------------------------
st.subheader("Model diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{auc:.3f}")
c2.metric("Accuracy", f"{acc:.3f}")
c3.metric("Years in sample", f"{len(results_df)}")

st.markdown(
    """
**Interpretation of ROC-AUC:**  
This is the probability that the model assigns a higher vintage probability to a true vintage year than to a non-vintage year.

- **0.5** = no predictive ability  
- **0.8** = quite good  
- **0.9+** = very strong
"""
)

# Confusion matrix
st.subheader("Confusion matrix")
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No vintage", "Vintage"])
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# -----------------------------
# Feature importances
# -----------------------------
st.subheader("Feature importances (full-sample Random Forest)")

importance_df = pd.DataFrame({
    "Variable": selected_feature_labels,
    "Importance": rf_full.feature_importances_
}).sort_values("Importance", ascending=False)

st.dataframe(importance_df, use_container_width=True)

fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
ax_imp.bar(importance_df["Variable"], importance_df["Importance"])
ax_imp.set_ylabel("Importance")
ax_imp.set_title("Random Forest Feature Importances")
ax_imp.tick_params(axis="x", rotation=45)
plt.tight_layout()
st.pyplot(fig_imp)

# -----------------------------
# Probability plot
# -----------------------------
st.subheader("Predicted vintage probability by year")

vintage_mask = results_df["target"] == 1

fig_prob, ax_prob = plt.subplots(figsize=(12, 5))
ax_prob.plot(results_df["year"], results_df["pred_prob"], marker="o")
ax_prob.scatter(
    results_df.loc[vintage_mask, "year"],
    results_df.loc[vintage_mask, "pred_prob"],
    color="red",
    s=80,
    label="Actual vintage years"
)
ax_prob.axhline(0.5, linestyle="--")
ax_prob.set_xlabel("Year")
ax_prob.set_ylabel("Predicted probability of vintage")
ax_prob.set_title("Random Forest predicted probabilities")
ax_prob.legend()
st.pyplot(fig_prob)

# -----------------------------
# Misclassifications
# -----------------------------
st.subheader("Misclassified years")

misclassified = results_df[~results_df["correct"]].copy()

if len(misclassified) == 0:
    st.success("No misclassified years at the 0.5 threshold.")
else:
    st.dataframe(
        misclassified[["year", "actual_label", "pred_label", "pred_prob"]],
        use_container_width=True
    )

# False negatives / false positives
st.subheader("Detailed classification tables")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Missed vintages**")
    missed_vintages = results_df[(results_df["target"] == 1) & (results_df["pred_class"] == 0)].copy()
    if len(missed_vintages) == 0:
        st.write("None")
    else:
        st.dataframe(missed_vintages[["year", "pred_prob"]], use_container_width=True)

with col2:
    st.markdown("**Years the model says should have been vintage**")
    false_vintages = results_df[(results_df["target"] == 0) & (results_df["pred_class"] == 1)].copy()
    if len(false_vintages) == 0:
        st.write("None")
    else:
        st.dataframe(false_vintages[["year", "pred_prob"]], use_container_width=True)

# -----------------------------
# Full prediction table
# -----------------------------
st.subheader("Full prediction table")
st.dataframe(
    results_df[["year", "actual_label", "pred_label", "pred_prob", "correct"]],
    use_container_width=True
)
