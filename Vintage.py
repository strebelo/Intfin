import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Port Vintage Prediction", layout="wide")
st.title("Port Vintage Declaration Prediction")
st.markdown(
    """
Upload yearly-monthly weather data and estimate models that predict whether a year was declared Vintage.

This app is built for small samples such as 1940–2011 and supports:
- Feature construction from monthly temperature and rainfall data
- Logistic regression
- L1-penalized logistic regression (LASSO-style selection)
- Random forest
- Gradient boosting
- Cross-validated predictive evaluation
- Probability scoring for each year

**Expected data structure:** one row per **year-month**, with columns for year, month, weather variables, and outcome.
If your data format is different, you can still use the app by mapping columns in the sidebar.
"""
)

# =========================
# Helpers
# =========================
MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


def safe_month_to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        x = int(x)
        return x if 1 <= x <= 12 else np.nan
    s = str(x).strip().lower()
    if s.isdigit():
        v = int(s)
        return v if 1 <= v <= 12 else np.nan
    return MONTH_NAME_TO_NUM.get(s, np.nan)


def normalize_binary_outcome(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
        "vintage": 1,
        "non-vintage": 0,
        "non vintage": 0,
        "declared": 1,
        "not declared": 0,
    }
    out = []
    for val in series:
        if pd.isna(val):
            out.append(np.nan)
        elif isinstance(val, (int, np.integer, float, np.floating)) and val in [0, 1]:
            out.append(int(val))
        else:
            out.append(mapping.get(str(val).strip().lower(), np.nan))
    return pd.Series(out, index=series.index)


def season_months(name: str) -> List[int]:
    seasons = {
        "Winter (Dec-Feb)": [12, 1, 2],
        "Spring (Mar-May)": [3, 4, 5],
        "Early Summer (Jun-Jul)": [6, 7],
        "Late Summer (Aug-Sep)": [8, 9],
        "Summer (Jun-Aug)": [6, 7, 8],
        "Growing Season (Apr-Sep)": [4, 5, 6, 7, 8, 9],
        "Harvest Window (Sep-Oct)": [9, 10],
    }
    return seasons[name]


def weighted_mean(values: pd.Series, weights: Optional[pd.Series]) -> float:
    valid = values.notna()
    if weights is not None:
        valid = valid & weights.notna() & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    if weights is None:
        return values[valid].mean()
    return np.average(values[valid], weights=weights[valid])


def weighted_sum(values: pd.Series, weights: Optional[pd.Series]) -> float:
    valid = values.notna()
    if valid.sum() == 0:
        return np.nan
    # Rainfall should normally be summed directly. We do not weight the total rainfall by days observed;
    # instead, we simply sum available monthly totals. The days-observed variable is mainly useful as a data-quality indicator.
    return values[valid].sum()


def add_gdd_proxy(df: pd.DataFrame, tmax_col: str, tmin_col: str) -> pd.DataFrame:
    out = df.copy()
    out["tmean_month"] = (out[tmax_col] + out[tmin_col]) / 2.0
    out["gdd_month_base10"] = np.maximum(out["tmean_month"] - 10.0, 0.0)
    return out


def build_yearly_features(
    df: pd.DataFrame,
    year_col: str,
    month_col: str,
    tmax_col: str,
    tmin_col: str,
    rain_col: str,
    days_obs_col: Optional[str],
    vintage_col: str,
    missing_code: float,
    use_weights: bool,
    selected_feature_flags: Dict[str, bool],
) -> pd.DataFrame:
    data = df.copy()
    for col in [tmax_col, tmin_col, rain_col] + ([days_obs_col] if days_obs_col else []):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data.loc[data[col] == missing_code, col] = np.nan

    data[month_col] = data[month_col].apply(safe_month_to_num)
    data = data.dropna(subset=[year_col, month_col])
    data[year_col] = pd.to_numeric(data[year_col], errors="coerce")
    data = data.dropna(subset=[year_col])
    data[year_col] = data[year_col].astype(int)
    data[month_col] = data[month_col].astype(int)

    data[vintage_col] = normalize_binary_outcome(data[vintage_col])
    data = add_gdd_proxy(data, tmax_col, tmin_col)

    grouped = []
    for year, g in data.groupby(year_col):
        row = {"year": year}
        row["vintage"] = g[vintage_col].dropna().iloc[0] if g[vintage_col].dropna().shape[0] > 0 else np.nan
        w = g[days_obs_col] if (days_obs_col and use_weights) else None

        # Data coverage diagnostics
        row["months_present"] = g[[month_col, tmax_col, tmin_col, rain_col]].dropna(subset=[month_col]).shape[0]
        row["months_with_any_weather"] = g[[tmax_col, tmin_col, rain_col]].notna().any(axis=1).sum()
        if days_obs_col:
            row["total_days_observed"] = g[days_obs_col].fillna(0).sum()

        # Monthly features
        for m in range(1, 13):
            gm = g[g[month_col] == m]
            wm = gm[days_obs_col] if (days_obs_col and use_weights) else None
            if selected_feature_flags.get("monthly_tmax", False):
                row[f"tmax_{MONTH_LABELS[m]}"] = weighted_mean(gm[tmax_col], wm) if gm.shape[0] else np.nan
            if selected_feature_flags.get("monthly_tmin", False):
                row[f"tmin_{MONTH_LABELS[m]}"] = weighted_mean(gm[tmin_col], wm) if gm.shape[0] else np.nan
            if selected_feature_flags.get("monthly_rain", False):
                row[f"rain_{MONTH_LABELS[m]}"] = weighted_sum(gm[rain_col], wm) if gm.shape[0] else np.nan

        # Seasonal engineered features
        season_defs = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "late_summer": [8, 9],
            "growing": [4, 5, 6, 7, 8, 9],
            "harvest": [9, 10],
        }
        for sname, months in season_defs.items():
            gs = g[g[month_col].isin(months)]
            ws = gs[days_obs_col] if (days_obs_col and use_weights) else None
            if selected_feature_flags.get("seasonal_temps", False):
                row[f"tmax_{sname}"] = weighted_mean(gs[tmax_col], ws) if gs.shape[0] else np.nan
                row[f"tmin_{sname}"] = weighted_mean(gs[tmin_col], ws) if gs.shape[0] else np.nan
                row[f"tmean_{sname}"] = weighted_mean((gs[tmax_col] + gs[tmin_col]) / 2.0, ws) if gs.shape[0] else np.nan
            if selected_feature_flags.get("seasonal_rain", False):
                row[f"rain_{sname}"] = weighted_sum(gs[rain_col], ws) if gs.shape[0] else np.nan
            if selected_feature_flags.get("gdd", False):
                row[f"gdd_{sname}"] = weighted_sum(gs["gdd_month_base10"], None) if gs.shape[0] else np.nan

        # Derived contrasts and event-style features
        if selected_feature_flags.get("temperature_spreads", False):
            row["diurnal_summer"] = row.get("tmax_summer", np.nan) - row.get("tmin_summer", np.nan)
            row["diurnal_harvest"] = row.get("tmax_harvest", np.nan) - row.get("tmin_harvest", np.nan)
            row["spring_to_summer_warming"] = row.get("tmean_summer", np.nan) - row.get("tmean_spring", np.nan)

        if selected_feature_flags.get("rain_ratios", False):
            summer_rain = row.get("rain_summer", np.nan)
            harvest_rain = row.get("rain_harvest", np.nan)
            row["harvest_vs_summer_rain_ratio"] = (
                harvest_rain / summer_rain if pd.notna(harvest_rain) and pd.notna(summer_rain) and summer_rain != 0 else np.nan
            )

        if selected_feature_flags.get("quality_index", False):
            # A simple hand-built index; coefficients are not estimated here.
            # Positive loading on growing-season warmth, negative loading on late rain.
            row["douro_weather_index"] = (
                0.5 * row.get("tmean_growing", 0)
                + 0.3 * row.get("gdd_growing", 0)
                - 0.02 * row.get("rain_late_summer", 0)
                - 0.02 * row.get("rain_harvest", 0)
            )

        grouped.append(row)

    yearly = pd.DataFrame(grouped).sort_values("year").reset_index(drop=True)
    return yearly


def choose_cv(n_obs: int, y: pd.Series, requested: str):
    if requested == "Leave-one-out":
        return LeaveOneOut(), "Leave-one-out"
    n_splits = min(5, int(y.value_counts().min()))
    if n_splits < 2:
        return LeaveOneOut(), "Leave-one-out (fallback because the minority class is too small for stratified folds)"
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), f"Stratified {n_splits}-fold"


def make_model(model_name: str, class_weight_option: str, random_state: int):
    class_weight = None if class_weight_option == "None" else "balanced"

    if model_name == "Logistic regression":
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            class_weight=class_weight,
            random_state=random_state,
        )
        return model, True

    if model_name == "L1 logistic (LASSO-style)":
        model = LogisticRegression(
            penalty="l1",
            C=0.5,
            solver="liblinear",
            max_iter=5000,
            class_weight=class_weight,
            random_state=random_state,
        )
        return model, True

    if model_name == "Random forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=random_state,
        )
        return model, False

    if model_name == "Gradient boosting":
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state,
        )
        return model, False

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(X: pd.DataFrame, y: pd.Series, model_name: str, cv_name: str, class_weight_option: str, random_state: int):
    model, needs_scaling = make_model(model_name, class_weight_option, random_state)

    numeric_features = X.columns.tolist()
    numeric_transformer_steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        numeric_transformer_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_transformer_steps), numeric_features)
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    cv, cv_used = choose_cv(len(y), y, cv_name)

    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "cv_used": cv_used,
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }

    pipe.fit(X, y)

    feature_importance = None
    coefficients = None

    if model_name in ["Logistic regression", "L1 logistic (LASSO-style)"]:
        fitted_model = pipe.named_steps["model"]
        coefs = fitted_model.coef_.ravel()
        coefficients = pd.DataFrame({
            "feature": X.columns,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }).sort_values("abs_coefficient", ascending=False)
    else:
        try:
            perm = permutation_importance(pipe, X, y, n_repeats=30, random_state=random_state, scoring="roc_auc")
            feature_importance = pd.DataFrame({
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False)
        except Exception:
            pass

    results = pd.DataFrame({
        "year": X.index,
        "actual_vintage": y.values,
        "predicted_probability": proba,
        "predicted_class_0_5": pred,
    })

    return pipe, metrics, results, coefficients, feature_importance


def download_df(df: pd.DataFrame, name: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=name, mime="text/csv")


# =========================
# Sidebar controls
# =========================
st.sidebar.header("1) Upload data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload your file to begin. A template description appears below.")
    st.markdown(
        """
### Recommended data layout
Each row should represent one month in one year. Example:

| year | month | tmax | tmin | rain | days_obs | vintage |
|---|---:|---:|---:|---:|---:|---:|
| 1940 | 1 | 12.3 | 5.1 | 110.2 | 31 | 0 |
| 1940 | 2 | 13.1 | 6.2 | 95.0 | 29 | 0 |
| ... | ... | ... | ... | ... | ... | ... |
| 1963 | 9 | 29.0 | 15.5 | 8.0 | 30 | 1 |

Notes:
- Missing data may be coded as **-99.9**.
- `vintage` may be coded as 0/1, yes/no, vintage/non-vintage, or declared/not declared.
- `days_obs` is optional but useful.
"""
    )
    st.stop()

# Read file
try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.sidebar.header("2) Map columns")
all_cols = raw.columns.tolist()

def pick_col(label: str, default_candidates: List[str], optional: bool = False):
    options = ["<None>"] + all_cols if optional else all_cols
    default_idx = 0
    for c in default_candidates:
        if c in all_cols:
            default_idx = options.index(c) if optional else all_cols.index(c)
            break
    choice = st.sidebar.selectbox(label, options, index=default_idx)
    return None if choice == "<None>" else choice

year_col = pick_col("Year column", ["year", "Year", "ANO", "ano"])
month_col = pick_col("Month column", ["month", "Month", "mes", "MES"])
tmax_col = pick_col("Average max daily air temperature", ["tmax", "TMAX", "avg_max_temp", "max_temp"])
tmin_col = pick_col("Average min daily air temperature", ["tmin", "TMIN", "avg_min_temp", "min_temp"])
rain_col = pick_col("Total rainfall", ["rain", "rainfall", "precip", "PRECIP"])
days_obs_col = pick_col("Days with observations (optional)", ["days_obs", "n_days", "obs_days"], optional=True)
vintage_col = pick_col("Vintage outcome column", ["vintage", "declared", "Vintage", "outcome"])

st.sidebar.header("3) Cleaning and feature construction")
missing_code = st.sidebar.number_input("Missing value code", value=-99.9, format="%.4f")
use_weights = st.sidebar.checkbox("Use days observed as weights for temperature averages", value=True)

st.sidebar.markdown("**Choose engineered features**")
selected_feature_flags = {
    "monthly_tmax": st.sidebar.checkbox("Monthly max temperatures", value=False),
    "monthly_tmin": st.sidebar.checkbox("Monthly min temperatures", value=False),
    "monthly_rain": st.sidebar.checkbox("Monthly rainfall totals", value=False),
    "seasonal_temps": st.sidebar.checkbox("Seasonal temperatures", value=True),
    "seasonal_rain": st.sidebar.checkbox("Seasonal rainfall", value=True),
    "gdd": st.sidebar.checkbox("Growing degree-day proxy", value=True),
    "temperature_spreads": st.sidebar.checkbox("Temperature spreads and contrasts", value=True),
    "rain_ratios": st.sidebar.checkbox("Rain ratios", value=False),
    "quality_index": st.sidebar.checkbox("Hand-built Douro weather index", value=True),
}

st.sidebar.header("4) Modeling")
model_name = st.sidebar.selectbox(
    "Model",
    [
        "Logistic regression",
        "L1 logistic (LASSO-style)",
        "Random forest",
        "Gradient boosting",
    ],
)
cv_name = st.sidebar.selectbox("Cross-validation", ["Leave-one-out", "Stratified k-fold"], index=0)
class_weight_option = st.sidebar.selectbox("Class imbalance handling", ["Balanced", "None"], index=0)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# =========================
# Main data display
# =========================
st.subheader("Raw data preview")
st.dataframe(raw.head(20), use_container_width=True)

# Build yearly features
try:
    yearly = build_yearly_features(
        df=raw,
        year_col=year_col,
        month_col=month_col,
        tmax_col=tmax_col,
        tmin_col=tmin_col,
        rain_col=rain_col,
        days_obs_col=days_obs_col,
        vintage_col=vintage_col,
        missing_code=missing_code,
        use_weights=use_weights,
        selected_feature_flags=selected_feature_flags,
    )
except Exception as e:
    st.error(f"Feature construction failed: {e}")
    st.stop()

st.subheader("Yearly feature table")
st.dataframe(yearly.head(20), use_container_width=True)
download_df(yearly, "yearly_features.csv", "Download yearly feature table as CSV")

# Data quality summary
st.subheader("Data quality checks")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Years in sample", int(yearly.shape[0]))
with col2:
    st.metric("Declared Vintage years", int(yearly["vintage"].fillna(0).sum()))
with col3:
    st.metric("Non-Vintage years", int((yearly["vintage"] == 0).sum()))

missing_summary = yearly.isna().mean().sort_values(ascending=False).reset_index()
missing_summary.columns = ["variable", "share_missing"]
st.dataframe(missing_summary.head(20), use_container_width=True)

# Feature selection for model
st.subheader("Model feature selection")
excluded_cols = ["year", "vintage"]
auto_feature_candidates = [c for c in yearly.columns if c not in excluded_cols]
default_features = [
    c for c in auto_feature_candidates
    if any(key in c for key in ["summer", "spring", "harvest", "growing", "late_summer", "douro_weather_index", "gdd"])
]
selected_features = st.multiselect(
    "Choose predictors",
    options=auto_feature_candidates,
    default=default_features[:12] if len(default_features) > 0 else auto_feature_candidates[:10],
)

analysis = yearly.dropna(subset=["vintage"]).copy()
if len(selected_features) == 0:
    st.warning("Select at least one predictor.")
    st.stop()

X = analysis[selected_features].copy()
y = analysis["vintage"].astype(int).copy()
X.index = analysis["year"]

# Correlations for quick inspection
st.subheader("Quick exploratory view")
if X.shape[1] >= 2:
    corr = analysis[["vintage"] + selected_features].corr(numeric_only=True)
    st.dataframe(corr[["vintage"]].sort_values("vintage", ascending=False), use_container_width=True)
else:
    st.info("Select at least two predictors to see a richer correlation view.")

# Fit model
st.subheader("Model results")
try:
    fitted_pipe, metrics, cv_results, coefficients, feature_importance = evaluate_model(
        X=X,
        y=y,
        model_name=model_name,
        cv_name=cv_name,
        class_weight_option=class_weight_option,
        random_state=int(random_state),
    )
except Exception as e:
    st.error(f"Model estimation failed: {e}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if pd.notna(metrics["roc_auc"]) else "NA")
m2.metric("Brier score", f"{metrics['brier_score']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("Precision", f"{metrics['precision']:.3f}")

st.caption(f"Cross-validation used: {metrics['cv_used']}")

st.markdown("**Predicted probabilities by year**")
st.dataframe(cv_results.sort_values("predicted_probability", ascending=False), use_container_width=True)
download_df(cv_results, "vintage_predictions.csv", "Download yearly predictions as CSV")

# Confusion matrix and interpretation
pred_05 = cv_results["predicted_class_0_5"].values
cm = confusion_matrix(y, pred_05)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
st.markdown("**Confusion matrix at 0.5 threshold**")
st.dataframe(cm_df, use_container_width=True)

# Coefficients or importance
if coefficients is not None:
    st.markdown("**Estimated coefficients**")
    st.dataframe(coefficients, use_container_width=True)
    download_df(coefficients, "logistic_coefficients.csv", "Download coefficients as CSV")
    st.info(
        "For logistic models, a positive coefficient means that higher values of the variable increase the probability of a Vintage declaration, holding the other included variables fixed."
    )

if feature_importance is not None:
    st.markdown("**Permutation feature importance**")
    st.dataframe(feature_importance, use_container_width=True)
    download_df(feature_importance, "feature_importance.csv", "Download feature importance as CSV")

# In-sample fitted probabilities from full model
full_proba = fitted_pipe.predict_proba(X)[:, 1]
full_fit = pd.DataFrame({
    "year": X.index,
    "actual_vintage": y.values,
    "fitted_probability_full_sample": full_proba,
}).sort_values("fitted_probability_full_sample", ascending=False)

st.markdown("**Full-sample fitted probabilities**")
st.dataframe(full_fit, use_container_width=True)
download_df(full_fit, "full_sample_fitted_probabilities.csv", "Download full-sample fitted probabilities")

# Simple rule-based threshold explorer
st.subheader("Vintage threshold explorer")
st.markdown(
    "This is a simple descriptive screen, not a formal model. It lets you see whether years above or below a chosen weather threshold were more likely to be declared Vintage."
)
threshold_var = st.selectbox("Choose threshold variable", options=selected_features)
threshold_value = st.slider(
    "Threshold value",
    min_value=float(np.nanmin(analysis[threshold_var])),
    max_value=float(np.nanmax(analysis[threshold_var])),
    value=float(np.nanmedian(analysis[threshold_var])),
)
direction = st.radio("Vintage more likely when variable is", ["Above threshold", "Below threshold"], horizontal=True)

if direction == "Above threshold":
    mask = analysis[threshold_var] >= threshold_value
else:
    mask = analysis[threshold_var] <= threshold_value

share_vintage = analysis.loc[mask, "vintage"].mean() if mask.sum() > 0 else np.nan
st.write(f"Years satisfying the rule: **{int(mask.sum())}**")
st.write(f"Share declared Vintage among those years: **{share_vintage:.3f}**" if pd.notna(share_vintage) else "No years satisfy the selected rule.")

# Documentation / interpretation section
st.subheader("How to use this sensibly")
st.markdown(
    """
### Practical advice
- Start with **seasonal variables** rather than all monthly variables. With about 72 years of data, too many predictors will overfit.
- Use **leave-one-out cross-validation** as the default benchmark because the sample is small.
- Focus on **ROC AUC, Brier score, recall, and precision**, not just accuracy, because declared Vintage years are uncommon.
- Compare:
  - **Logistic regression** for interpretability
  - **L1 logistic** for sparse selection
  - **Random forest** or **gradient boosting** for nonlinear effects and interactions
- Treat results as **predictive**, not necessarily causal.

### A good starting predictor set
A good first pass is often:
- `tmean_growing`
- `gdd_growing`
- `rain_spring`
- `rain_late_summer`
- `rain_harvest`
- `douro_weather_index`

### Missing data
This app converts your missing code, such as `-99.9`, into missing values and then imputes model inputs with the **median** of each variable inside the cross-validation pipeline.
For a publication-quality paper, you may want to compare this with a more tailored climatological imputation strategy.
"""
)

st.subheader("Optional next steps")
st.markdown(
    """
Useful extensions you may want to add later:
1. A **Bayesian logistic model** with informative priors.
2. A **declared vs not-declared vs classic/non-classic** multinomial model if you later split the outcome more finely.
3. **SHAP values** for nonlinear model interpretation.
4. Hyperparameter tuning for the random forest and boosting models.
5. Region-specific weather indices if you later add sub-regional station data.
"""
)
