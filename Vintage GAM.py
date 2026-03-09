import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pygam import LogisticGAM, s
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Port Vintage Prediction with GAM", layout="wide")

st.title("Port Vintage Declaration Prediction — GAM Model")

st.markdown("""
This app estimates a **Generalized Additive Model (GAM)** for Port vintage declaration using climate variables constructed from monthly data.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# -----------------------------
# Read data
# -----------------------------
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head())

required_cols = ["year", "month", "tmax", "tmin", "rain", "vintage"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"Spreadsheet must contain these columns: {required_cols}")
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# Clean types
for col in ["year", "month", "tmax", "tmin", "rain", "vintage"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["year", "month", "tmax", "tmin", "rain", "vintage"]).copy()
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

# -----------------------------
# Create monthly climate measures
# -----------------------------
df["tmean"] = (df["tmax"] + df["tmin"]) / 2
df["gdd_month"] = np.maximum(df["tmean"] - 10, 0)

# -----------------------------
# Annual aggregation
# -----------------------------
def build_yearly_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    years = sorted(monthly_df["year"].unique())

    for y in years:
        this_year = monthly_df[monthly_df["year"] == y]
        prev_year = monthly_df[monthly_df["year"] == y - 1]

        # Need months from current year and previous year for Oct-Feb rain
        # Oct-Feb for year y means Oct-Dec of y-1 plus Jan-Feb of y
        oct_dec_prev = prev_year[prev_year["month"].isin([10, 11, 12])]
        jan_feb_this = this_year[this_year["month"].isin([1, 2])]
        oct_feb = pd.concat([oct_dec_prev, jan_feb_this], axis=0)

        apr_sep = this_year[this_year["month"].isin([4, 5, 6, 7, 8, 9])]
        jul_aug = this_year[this_year["month"].isin([7, 8])]
        july = this_year[this_year["month"] == 7]
        august = this_year[this_year["month"] == 8]
        september = this_year[this_year["month"] == 9]

        # skip incomplete years
        if len(this_year) < 12:
            continue

        row = {
            "year": y,
            "gdd_apr_sep": apr_sep["gdd_month"].sum() if not apr_sep.empty else np.nan,
            "temp_jul_aug": jul_aug["tmean"].mean() if not jul_aug.empty else np.nan,
            "temp_july": july["tmean"].mean() if not july.empty else np.nan,
            "temp_august": august["tmean"].mean() if not august.empty else np.nan,
            "rain_september": september["rain"].sum() if not september.empty else np.nan,
            "rain_oct_feb": oct_feb["rain"].sum() if not oct_feb.empty else np.nan,
            "dtr_apr_sep": (apr_sep["tmax"] - apr_sep["tmin"]).mean() if not apr_sep.empty else np.nan,
            "vintage_raw": this_year["vintage"].max()
        }

        rows.append(row)

    return pd.DataFrame(rows)

year_df = build_yearly_dataset(df)

if year_df.empty:
    st.error("Could not construct annual dataset from the uploaded file.")
    st.stop()

st.subheader("Annual Data Preview")
st.dataframe(year_df.head())

# -----------------------------
# Target selection
# -----------------------------
st.sidebar.header("Model Settings")

target_choice = st.sidebar.radio(
    "Choose target variable",
    ["Classic vintages only", "Classic + non-classic vintages"]
)

if target_choice == "Classic vintages only":
    year_df["target"] = (year_df["vintage_raw"] == 1).astype(int)
else:
    year_df["target"] = (year_df["vintage_raw"] > 0).astype(int)

feature_options = [
    "gdd_apr_sep",
    "temp_jul_aug",
    "temp_july",
    "temp_august",
    "rain_september",
    "rain_oct_feb",
    "dtr_apr_sep"
]

selected_features = st.sidebar.multiselect(
    "Choose explanatory variables",
    feature_options,
    default=["gdd_apr_sep", "temp_jul_aug", "rain_september"]
)

if len(selected_features) == 0:
    st.warning("Please select at least one explanatory variable.")
    st.stop()

# -----------------------------
# Prepare model data
# -----------------------------
model_df = year_df[["year", "target"] + selected_features].dropna().copy()

if len(model_df) < 10:
    st.error("Not enough complete annual observations after dropping missing values.")
    st.stop()

X = model_df[selected_features].values
y = model_df["target"].values

# -----------------------------
# GAM specification
# -----------------------------
# One smooth term per selected feature
terms = s(0)
for i in range(1, len(selected_features)):
    terms = terms + s(i)

# User controls
st.sidebar.header("GAM Controls")
n_splines = st.sidebar.slider("Number of splines per variable", min_value=4, max_value=15, value=8)
lam = st.sidebar.selectbox("Smoothing penalty (lam)", options=[0.1, 1, 3, 10, 30, 100], index=2)
threshold = st.sidebar.slider("Classification threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

# -----------------------------
# Fit model
# -----------------------------
try:
    gam = LogisticGAM(terms, n_splines=n_splines, lam=lam)
    gam.fit(X, y)
except Exception as e:
    st.error(f"Error fitting GAM: {e}")
    st.stop()

# Predictions
pred_prob = gam.predict_proba(X)
pred_class = (pred_prob >= threshold).astype(int)

# Metrics
acc = accuracy_score(y, pred_class)

try:
    auc = roc_auc_score(y, pred_prob)
except:
    auc = np.nan

cm = confusion_matrix(y, pred_class)

# -----------------------------
# Results
# -----------------------------
st.subheader("Model Diagnostics")

col1, col2, col3 = st.columns(3)
col1.metric("Observations", len(model_df))
col2.metric("Accuracy", f"{acc:.3f}")
col3.metric("ROC AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

with st.expander("Confusion Matrix"):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    st.dataframe(cm_df)

with st.expander("GAM Summary"):
    summary_text = []
    summary_text.append("Selected variables:")
    for v in selected_features:
        summary_text.append(f"- {v}")
    summary_text.append("")
    summary_text.append("Pseudo R-squared and other diagnostics from pyGAM:")
    stats = gam.statistics_

    for key, val in stats.items():
        if np.isscalar(val):
            summary_text.append(f"{key}: {val}")
    st.text("\n".join(summary_text))

# -----------------------------
# Plot predicted probabilities and actual vintages
# -----------------------------
plot_df = model_df.copy()
plot_df["pred_prob"] = pred_prob
plot_df["pred_class"] = pred_class

actual_vintage_years = plot_df.loc[plot_df["target"] == 1, "year"]
actual_vintage_probs = plot_df.loc[plot_df["target"] == 1, "pred_prob"]

st.subheader("Predicted Probability of Vintage by Year")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(plot_df["year"], plot_df["pred_prob"], linewidth=2, label="GAM predicted probability")
ax.scatter(
    actual_vintage_years,
    actual_vintage_probs,
    color="red",
    s=60,
    label="Actual vintage declared",
    zorder=3
)

ax.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold = {threshold:.2f}")
ax.set_xlabel("Year")
ax.set_ylabel("Probability of Vintage")
ax.set_title("GAM Predicted Probability and Actual Vintage Declarations")
ax.set_ylim(-0.02, 1.02)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# -----------------------------
# Misclassifications
# -----------------------------
st.subheader("Misclassifications")

false_negatives = plot_df[(plot_df["target"] == 1) & (plot_df["pred_class"] == 0)].copy()
false_positives = plot_df[(plot_df["target"] == 0) & (plot_df["pred_class"] == 1)].copy()

col_fn, col_fp = st.columns(2)

with col_fn:
    st.markdown("**Missed vintages (actual vintage, model predicted no vintage)**")
    if false_negatives.empty:
        st.write("None")
    else:
        st.dataframe(false_negatives[["year", "pred_prob"] + selected_features].sort_values("year"))

with col_fp:
    st.markdown("**Years model says should have been vintage (predicted vintage, but not declared)**")
    if false_positives.empty:
        st.write("None")
    else:
        st.dataframe(false_positives[["year", "pred_prob"] + selected_features].sort_values("year"))

# -----------------------------
# Variable effect plots
# -----------------------------
st.subheader("Estimated Smooth Effects")

for i, feature in enumerate(selected_features):
    fig, ax = plt.subplots(figsize=(8, 4))
    XX = gam.generate_X_grid(term=i)
    pdep = gam.partial_dependence(term=i, X=XX)
    confi = gam.partial_dependence(term=i, X=XX, width=0.95)[1]

    ax.plot(XX[:, i], pdep, linewidth=2)
    ax.plot(XX[:, i], confi[:, 0], linestyle="--", linewidth=1)
    ax.plot(XX[:, i], confi[:, 1], linestyle="--", linewidth=1)

    ax.set_title(f"Smooth effect of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Partial effect on log-odds")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# -----------------------------
# Downloadable results table
# -----------------------------
st.subheader("Download Results")

results_df = plot_df.copy()
results_df["actual"] = y
results_df["predicted_probability"] = pred_prob
results_df["predicted_class"] = pred_class

csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name="gam_vintage_results.csv",
    mime="text/csv"
)
