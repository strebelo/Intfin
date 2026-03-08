import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# USER SETTINGS
# ============================================================

FILE_PATH = "Data Vintages.xlsx"   # change if needed
SHEET_NAME = "Data"

# Default model choice:
# "best_logit" is the recommended default
MODEL_NAME = "best_logit"

# Other built-in options:
# "baseline_sep_rain"
# "monthly_rain"
# "jul_aug_split"
# "interaction_model"
# "huglin_model"
# "random_forest"

BASE_TEMP_GDD = 10.0
CLASSIFICATION_THRESHOLD = 0.50

# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

# Standardize column names
df.columns = [str(c).strip().lower() for c in df.columns]

# Keep only needed columns
needed = ["year", "month", "tmax", "tmin", "rain", "days_obs", "vintage"]
df = df[[c for c in needed if c in df.columns]].copy()

# Convert to numeric
for c in ["year", "month", "tmax", "tmin", "rain", "days_obs", "vintage"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Treat sentinel missing values as NaN
df.loc[df["rain"] <= -99, "rain"] = np.nan
df.loc[df["tmax"] <= -99, "tmax"] = np.nan
df.loc[df["tmin"] <= -99, "tmin"] = np.nan

# Monthly mean temperature
df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0

# Growing degree days by month (base 10C)
df["gdd_month"] = np.maximum(df["tmean"] - BASE_TEMP_GDD, 0.0)

# Huglin monthly contribution (simple monthly approximation)
# HI_month = ((Tmean - 10) + (Tmax - 10))/2, floored at 0
df["huglin_month"] = np.maximum(((df["tmean"] - 10.0) + (df["tmax"] - 10.0)) / 2.0, 0.0)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def yearly_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    years = sorted(monthly_df["year"].dropna().astype(int).unique())

    for y in years:
        d = monthly_df[monthly_df["year"] == y].copy()

        # Annual outcome: assume vintage is constant within year
        vintage_vals = d["vintage"].dropna().unique()
        vintage = int(vintage_vals[0]) if len(vintage_vals) > 0 else np.nan

        # Core variables
        gdd_apr_sep = d.loc[d["month"].between(4, 9), "gdd_month"].sum(skipna=True)
        huglin_apr_sep = d.loc[d["month"].between(4, 9), "huglin_month"].sum(skipna=True)

        rain_sep = d.loc[d["month"] == 9, "rain"].sum(skipna=True)
        rain_apr = d.loc[d["month"] == 4, "rain"].sum(skipna=True)
        rain_may = d.loc[d["month"] == 5, "rain"].sum(skipna=True)
        rain_jun = d.loc[d["month"] == 6, "rain"].sum(skipna=True)
        rain_apr_jun = d.loc[d["month"].between(4, 6), "rain"].sum(skipna=True)

        rain_jul_aug = d.loc[d["month"].between(7, 8), "rain"].sum(skipna=True)

        temp_jul = d.loc[d["month"] == 7, "tmean"].mean(skipna=True)
        temp_aug = d.loc[d["month"] == 8, "tmean"].mean(skipna=True)
        temp_jul_aug = d.loc[d["month"].between(7, 8), "tmean"].mean(skipna=True)

        # Cool night index proxy: September minimum temperature
        cool_night_sep = d.loc[d["month"] == 9, "tmin"].mean(skipna=True)

        # Extreme heat counts (optional)
        days_gt_35 = d.loc[d["month"].between(7, 8) & (d["tmax"] > 35), "month"].count()

        # Winter rain: Oct-Dec of previous year + Jan-Feb of current year
        prev = monthly_df[monthly_df["year"] == (y - 1)].copy()
        rain_oct_feb = (
            prev.loc[prev["month"].between(10, 12), "rain"].sum(skipna=True)
            + d.loc[d["month"].between(1, 2), "rain"].sum(skipna=True)
        )

        rows.append({
            "year": y,
            "vintage": vintage,
            "gdd_apr_sep": gdd_apr_sep,
            "gdd_apr_sep_sq": gdd_apr_sep ** 2,
            "huglin_apr_sep": huglin_apr_sep,
            "rain_sep": rain_sep,
            "rain_apr": rain_apr,
            "rain_may": rain_may,
            "rain_jun": rain_jun,
            "rain_apr_jun": rain_apr_jun,
            "rain_jul_aug": rain_jul_aug,
            "temp_jul": temp_jul,
            "temp_aug": temp_aug,
            "temp_jul_aug": temp_jul_aug,
            "cool_night_sep": cool_night_sep,
            "days_gt_35_jul_aug": days_gt_35,
            "rain_oct_feb": rain_oct_feb,
        })

    out = pd.DataFrame(rows)

    # Interaction
    out["aug_x_sep_rain"] = out["temp_aug"] * out["rain_sep"]

    return out


year_df = yearly_features(df)

# Drop first year if prior Oct-Dec unavailable
# More generally, drop rows with missing vintage
year_df = year_df.dropna(subset=["vintage"]).copy()

# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODEL_SPECS = {
    "baseline_sep_rain": [
        "rain_sep"
    ],
    "monthly_rain": [
        "rain_apr", "rain_may", "rain_jun", "rain_sep"
    ],
    "jul_aug_split": [
        "rain_apr_jun", "temp_jul", "temp_aug", "rain_sep", "rain_oct_feb"
    ],
    "interaction_model": [
        "rain_apr_jun", "temp_jul", "temp_aug", "rain_sep", "aug_x_sep_rain", "rain_oct_feb"
    ],
    "huglin_model": [
        "huglin_apr_sep", "rain_sep", "rain_oct_feb"
    ],
    "best_logit": [
        "gdd_apr_sep",
        "gdd_apr_sep_sq",
        "rain_sep",
        "rain_apr_jun",
        "temp_jul_aug",
        "rain_oct_feb"
    ],
    "random_forest": [
        "rain_apr_jun",
        "temp_jul",
        "temp_aug",
        "rain_sep",
        "rain_oct_feb",
        "gdd_apr_sep"
    ],
}

if MODEL_NAME not in MODEL_SPECS:
    raise ValueError(f"Unknown MODEL_NAME '{MODEL_NAME}'. Choose from: {list(MODEL_SPECS.keys())}")

feature_cols = MODEL_SPECS[MODEL_NAME]

# Prepare estimation sample
model_df = year_df[["year", "vintage"] + feature_cols].dropna().copy()
model_df["vintage"] = model_df["vintage"].astype(int)

X = model_df[feature_cols].copy()
y = model_df["vintage"].copy()

# ============================================================
# DIAGNOSTICS HELPERS
# ============================================================

def classification_metrics(y_true, p_hat, threshold=0.5):
    y_pred = (p_hat >= threshold).astype(int)

    auc = roc_auc_score(y_true, p_hat) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, p_hat)

    # Handle zero-division safely
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "ROC_AUC": auc,
        "Brier": brier,
        "Precision": precision,
        "Recall": recall,
    }

def leave_one_out_logit(X, y):
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        X_train_const = sm.add_constant(X_train, has_constant="add")
        X_test_const = sm.add_constant(X_test, has_constant="add")

        try:
            res = sm.Logit(y_train, X_train_const).fit(disp=False)
            probs[test_idx[0]] = float(res.predict(X_test_const).iloc[0])
        except Exception:
            # fallback: penalized sklearn logistic if separation / convergence issues
            clf = LogisticRegression(max_iter=5000, solver="lbfgs")
            clf.fit(X_train, y_train)
            probs[test_idx[0]] = float(clf.predict_proba(X_test)[:, 1][0])

    return probs

def leave_one_out_rf(X, y):
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=3,
            random_state=42
        )
        rf.fit(X_train, y_train)
        probs[test_idx[0]] = float(rf.predict_proba(X_test)[:, 1][0])

    return probs

# ============================================================
# ESTIMATE MODEL
# ============================================================

print("=" * 80)
print(f"MODEL: {MODEL_NAME}")
print("Features:", feature_cols)
print("=" * 80)

if MODEL_NAME != "random_forest":
    X_const = sm.add_constant(X, has_constant="add")
    logit_model = sm.Logit(y, X_const)
    logit_res = logit_model.fit(disp=False)

    print("\nLOGIT ESTIMATES\n")
    print(logit_res.summary())

    # In-sample fitted probabilities
    model_df["p_hat_in_sample"] = logit_res.predict(X_const)

    # Leave-one-out predicted probabilities
    model_df["p_hat_loo"] = leave_one_out_logit(X, y)

    # Diagnostics
    metrics_in = classification_metrics(y, model_df["p_hat_in_sample"], threshold=CLASSIFICATION_THRESHOLD)
    metrics_loo = classification_metrics(y, model_df["p_hat_loo"], threshold=CLASSIFICATION_THRESHOLD)

    print("\nIN-SAMPLE METRICS")
    for k, v in metrics_in.items():
        print(f"{k:>12}: {v:.4f}")

    print("\nLEAVE-ONE-OUT METRICS")
    for k, v in metrics_loo.items():
        print(f"{k:>12}: {v:.4f}")

    print("\nLIKELIHOOD-BASED DIAGNOSTICS")
    print(f"Log-Likelihood      : {logit_res.llf:.4f}")
    print(f"Pseudo R-squared    : {logit_res.prsquared:.4f}")
    print(f"LR test p-value     : {logit_res.llr_pvalue:.4g}")
    print(f"N observations      : {int(logit_res.nobs)}")

else:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42
    )
    rf.fit(X, y)

    model_df["p_hat_in_sample"] = rf.predict_proba(X)[:, 1]
    model_df["p_hat_loo"] = leave_one_out_rf(X, y)

    metrics_in = classification_metrics(y, model_df["p_hat_in_sample"], threshold=CLASSIFICATION_THRESHOLD)
    metrics_loo = classification_metrics(y, model_df["p_hat_loo"], threshold=CLASSIFICATION_THRESHOLD)

    print("\nRANDOM FOREST FEATURE IMPORTANCE\n")
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(imp)

    print("\nIN-SAMPLE METRICS")
    for k, v in metrics_in.items():
        print(f"{k:>12}: {v:.4f}")

    print("\nLEAVE-ONE-OUT METRICS")
    for k, v in metrics_loo.items():
        print(f"{k:>12}: {v:.4f}")

# ============================================================
# OUTPUT YEAR-BY-YEAR PROBABILITIES
# ============================================================

print("\nYEAR-BY-YEAR PREDICTED PROBABILITIES")
cols_to_show = ["year", "vintage", "p_hat_in_sample", "p_hat_loo"] + feature_cols
print(model_df[cols_to_show].sort_values("year").to_string(index=False))

# Save results
out_name = f"vintage_model_results_{MODEL_NAME}.xlsx"
with pd.ExcelWriter(out_name, engine="openpyxl") as writer:
    model_df.sort_values("year").to_excel(writer, sheet_name="year_probs", index=False)

    if MODEL_NAME == "random_forest":
        imp_df = pd.DataFrame({
            "variable": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        imp_df.to_excel(writer, sheet_name="feature_importance", index=False)
    else:
        coef_df = pd.DataFrame({
            "variable": logit_res.params.index,
            "coef": logit_res.params.values,
            "std_err": logit_res.bse.values,
            "z_stat": logit_res.tvalues.values,
            "p_value": logit_res.pvalues.values
        })
        coef_df.to_excel(writer, sheet_name="coefficients", index=False)

        metrics_df = pd.DataFrame([
            {"sample": "in_sample", **metrics_in},
            {"sample": "leave_one_out", **metrics_loo},
        ])
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

print(f"\nResults written to: {out_name}")
