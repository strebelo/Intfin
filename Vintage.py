import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score

st.title("Port Vintage Logit Model")

st.write("""
This app estimates logistic regressions predicting whether a year is a classic vintage.
Upload the Excel file with monthly data containing these columns:

- year
- month
- tmax
- tmin
- rain
- vintage

where vintage = 1 for classic vintage years and 0 otherwise.
""")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["year", "month", "tmax", "tmin", "rain", "vintage"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace common missing-value codes
    df.loc[df["rain"] <= -99, "rain"] = np.nan
    df.loc[df["tmax"] <= -99, "tmax"] = np.nan
    df.loc[df["tmin"] <= -99, "tmin"] = np.nan

    # -----------------------------
    # Construct monthly variables
    # -----------------------------
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2
    df["gdd_month"] = np.maximum(df["tmean"] - 10, 0)

    # -----------------------------
    # Build yearly dataset
    # -----------------------------
    rows = []
    years = sorted(df["year"].dropna().unique())

    for y in years:
        d = df[df["year"] == y].copy()
        prev = df[df["year"] == y - 1].copy()

        vintage_vals = d["vintage"].dropna().unique()
        if len(vintage_vals) == 0:
            continue

        rows.append({
            "year": int(y),
            "vintage": int(vintage_vals[0]),
            "gdd_apr_sep": d.loc[d["month"].between(4, 9), "gdd_month"].sum(),
            "rain_sep": d.loc[d["month"] == 9, "rain"].sum(),
            "rain_apr_jun": d.loc[d["month"].between(4, 6), "rain"].sum(),
            "temp_jul_aug": d.loc[d["month"].between(7, 8), "tmean"].mean(),
            "rain_oct_feb": (
                prev.loc[prev["month"].between(10, 12), "rain"].sum()
                + d.loc[d["month"].between(1, 2), "rain"].sum()
            ),
            "temp_jul": d.loc[d["month"] == 7, "tmean"].mean(),
            "temp_aug": d.loc[d["month"] == 8, "tmean"].mean(),
            "rain_apr": d.loc[d["month"] == 4, "rain"].sum(),
            "rain_may": d.loc[d["month"] == 5, "rain"].sum(),
            "rain_jun": d.loc[d["month"] == 6, "rain"].sum(),
        })

    year_df = pd.DataFrame(rows)
    year_df["gdd_sq"] = year_df["gdd_apr_sep"] ** 2
    year_df["aug_x_sep_rain"] = year_df["temp_aug"] * year_df["rain_sep"]

    # Drop rows with missing vintage
    year_df = year_df.dropna(subset=["vintage"]).copy()

    # -----------------------------
    # Model choices
    # -----------------------------
    model_options = {
        "Best default model": [
            "gdd_apr_sep", "gdd_sq", "rain_sep",
            "rain_apr_jun", "temp_jul_aug", "rain_oct_feb"
        ],
        "Simple harvest model": [
            "rain_sep"
        ],
        "Monthly spring rain model": [
            "rain_apr", "rain_may", "rain_jun", "rain_sep"
        ],
        "July-August split model": [
            "rain_apr_jun", "temp_jul", "temp_aug", "rain_sep", "rain_oct_feb"
        ],
        "Interaction model": [
            "rain_apr_jun", "temp_jul", "temp_aug", "rain_sep",
            "aug_x_sep_rain", "rain_oct_feb"
        ]
    }

    model_name = st.selectbox("Choose model", list(model_options.keys()), index=0)
    feature_cols = model_options[model_name]

    model_df = year_df[["year", "vintage"] + feature_cols].dropna().copy()

    X = model_df[feature_cols]
    y = model_df["vintage"]

    # -----------------------------
    # Estimate logistic regression
    # -----------------------------
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_const)
    result = model.fit(disp=False)

    # Predicted probabilities
    model_df["predicted_probability"] = result.predict(X_const)

    # Binary predictions at threshold 0.5
    threshold = 0.5
    model_df["predicted_class"] = (model_df["predicted_probability"] >= threshold).astype(int)

    # -----------------------------
    # Diagnostics
    # -----------------------------
    auc = roc_auc_score(y, model_df["predicted_probability"])
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

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Estimated coefficients")
    st.dataframe(coef_table)

    st.subheader("Model diagnostics")
    diag_df = pd.DataFrame({
        "Metric": [
            "Pseudo R-squared",
            "Log-Likelihood",
            "LR test p-value",
            "ROC AUC",
            "Brier Score",
            "Precision",
            "Recall"
        ],
        "Value": [
            result.prsquared,
            result.llf,
            result.llr_pvalue,
            auc,
            brier,
            precision,
            recall
        ]
    })
    st.dataframe(diag_df)

    st.subheader("Predicted probability of vintage by year")
    st.dataframe(
        model_df[["year", "vintage", "predicted_probability", "predicted_class"] + feature_cols]
        .sort_values("year")
        .reset_index(drop=True)
    )

    st.subheader("Regression summary")
    st.text(result.summary())

import matplotlib.pyplot as plt

# ------------------------------------------------
# Compute predicted probabilities
# ------------------------------------------------

X_plot = year_df[feature_cols].dropna().copy()
X_plot_const = sm.add_constant(X_plot, has_constant="add")

year_df.loc[X_plot.index, "predicted_probability"] = result.predict(X_plot_const)

plot_df = year_df.dropna(subset=["predicted_probability"]).copy()

# -----------------------------------------
# Predicted probability vs actual vintages
# -----------------------------------------

import matplotlib.pyplot as plt

# Predicted probabilities from the fitted model
X_const = sm.add_constant(X, has_constant="add")
predicted_prob = result.predict(X_const)

# Build plotting dataframe
plot_df = pd.DataFrame({
    "year": model_df["year"],
    "vintage": y,
    "predicted_probability": predicted_prob
})

# Plot
st.subheader("Predicted probability of vintage by year")

fig, ax = plt.subplots(figsize=(10,5))

# Line for predicted probability
ax.plot(
    plot_df["year"],
    plot_df["predicted_probability"],
    color="blue",
    linewidth=2,
    label="Predicted probability"
)

# Highlight actual vintages
vintage_points = plot_df[plot_df["vintage"] == 1]

ax.scatter(
    vintage_points["year"],
    vintage_points["predicted_probability"],
    color="red",
    s=80,
    label="Declared vintage"
)

# Reference threshold
ax.axhline(0.5, linestyle="--", color="gray")

ax.set_xlabel("Year")
ax.set_ylabel("Probability of Vintage")
ax.set_title("Predicted Probability of Classic Port Vintage")

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
