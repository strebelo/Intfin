"""Streamlit app: Annual Log FX Changes — Normal vs. Fat Tails.

Run: streamlit run fx_volatility.py
Input: one monthly exchange rate per row, with Date and SpotRate columns.
AnnualLogChange = 100 * (log(S_t) - log(S_{t-12})), in log percentage points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


def parse_input_file(uploaded_file) -> pd.DataFrame:
    """Read the first Excel worksheet or a comma-delimited CSV."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix == ".csv":
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    elif suffix == ".xls":
        df = pd.read_excel(uploaded_file, engine="xlrd")
    else:
        raise ValueError("Upload an .xlsx, .xls, or .csv file.")
    if df.empty or len(df.columns) < 2:
        raise ValueError("The file must contain date and exchange-rate columns.")
    return df


def coerce_datetime(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Parse Excel dates, date strings, YYYYMMDD, and Excel serial dates.

    Numeric values are never interpreted as nanoseconds since 1970.
    No removed infer_datetime_format argument is used.
    """
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return pd.to_datetime(series, errors="coerce")

    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    numeric_mask = numeric.notna()
    dates = pd.to_datetime(
        text.mask(numeric_mask),
        format="mixed",
        errors="coerce",
        dayfirst=dayfirst,
    )

    ymd = text.str.fullmatch(r"\d{8}", na=False)
    if ymd.any():
        dates.loc[ymd] = pd.to_datetime(
            text.loc[ymd], format="%Y%m%d", errors="coerce"
        )

    serial = numeric_mask & ~ymd & numeric.between(1, 100000)
    if serial.any():
        dates.loc[serial] = pd.to_datetime(
            numeric.loc[serial].astype(float),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return dates


def normalize_colname(value) -> str:
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


def guess_columns(df: pd.DataFrame):
    lookup = {normalize_colname(col): col for col in df.columns}
    date_col = next(
        (lookup[n] for n in ["date", "month", "period", "observationdate"]
         if n in lookup), None
    )
    price_col = next(
        (lookup[n] for n in ["spotrate", "spot", "rate", "price",
                            "exchangerate", "fx", "spotusd", "usdusd"]
         if n in lookup), None
    )

    if date_col is None:
        for col in df.columns:
            if col == price_col:
                continue
            values = df[col].dropna()
            if values.empty:
                continue
            # Do not mistake an ordinary numeric price column for dates.
            numbers = pd.to_numeric(values, errors="coerce")
            if numbers.notna().mean() > 0.8:
                plausible = numbers.between(20000, 80000) | numbers.between(
                    19000101, 21001231
                )
                if plausible.mean() < 0.8:
                    continue
            if coerce_datetime(values).notna().mean() > 0.8:
                date_col = col
                break

    if price_col is None:
        for col in df.columns:
            if col != date_col:
                values = df[col].dropna()
                if len(values) and pd.to_numeric(
                    values, errors="coerce"
                ).notna().mean() > 0.8:
                    price_col = col
                    break
    return date_col, price_col


def compute_annual_log_changes(
    df: pd.DataFrame,
    date_col,
    price_col,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Use exact calendar-month lags; do not interpolate missing prices."""
    if date_col == price_col:
        raise ValueError("Choose different columns for dates and exchange rates.")

    dates = coerce_datetime(df[date_col], dayfirst=dayfirst)
    rates = pd.to_numeric(df[price_col], errors="coerce").astype(float)

    bad_date = dates.isna() & rates.notna()
    if bad_date.any():
        raise ValueError(
            f"{int(bad_date.sum())} numeric exchange-rate rows have missing or "
            "unrecognized dates. Check the date column and date convention."
        )

    invalid_rate = dates.notna() & (
        rates.isna() | ~np.isfinite(rates) | (rates <= 0)
    )
    if invalid_rate.any():
        examples = dates.loc[invalid_rate].dt.strftime("%Y-%m-%d").head(5)
        raise ValueError(
            "Exchange rates must be positive numbers. Invalid or missing "
            "rates on: " + ", ".join(examples) + "."
        )

    # Undated, nonnumeric source notes and blank rows are excluded.
    ignored_rows = int(dates.isna().sum())
    clean = pd.DataFrame({"Date": dates, "SpotRate": rates})
    clean = clean.dropna(subset=["Date"]).sort_values("Date")
    if clean.empty:
        raise ValueError("No valid monthly observations were found.")

    clean["Month"] = clean["Date"].dt.to_period("M")
    duplicates = clean["Month"].duplicated(keep=False)
    if duplicates.any():
        examples = clean.loc[duplicates, "Month"].astype(str).unique()[:5]
        raise ValueError(
            "More than one observation in a month: " + ", ".join(examples)
            + ". Supply one monthly rate per month."
        )

    calendar = pd.period_range(
        clean["Month"].min(), clean["Month"].max(), freq="M"
    )
    monthly = clean.set_index("Month")["SpotRate"].reindex(calendar)
    annual = 100.0 * (np.log(monthly) - np.log(monthly.shift(12)))

    result = pd.DataFrame({
        "Date": calendar.to_timestamp(),
        "SpotRate": monthly.to_numpy(),
        "AnnualLogChange": annual.to_numpy(),
    })
    result.attrs["ignored_rows"] = ignored_rows
    result.attrs["missing_months"] = int(monthly.isna().sum())
    result.attrs["input_observations"] = len(clean)
    return result


def purged_folds(month_numbers: np.ndarray):
    """Blocked folds with an 11-month exclusion on either side of each block.

    Neighboring 12-month changes share monthly returns. Excluding training
    observations less than 12 months from the validation block reduces this
    source of leakage. It does not remove all possible serial dependence.
    """
    folds = []
    for _, test in KFold(n_splits=5, shuffle=False).split(month_numbers):
        lower, upper = month_numbers[test].min(), month_numbers[test].max()
        train = np.flatnonzero(
            (month_numbers < lower - 11) | (month_numbers > upper + 11)
        )
        if len(train) >= 8 and len(test) >= 2:
            folds.append((train, test))
    if len(folds) < 2:
        raise ValueError(
            "The sample is too short for bandwidth cross-validation with "
            "12-month changes. Use a longer history."
        )
    return folds


def mean_log_likelihood(estimator, values):
    return float(estimator.score(values) / len(values))


def fit_density(values: np.ndarray, month_numbers: np.ndarray):
    values = np.asarray(values, dtype=float)
    if len(values) < 20:
        raise ValueError("At least 20 annual changes are needed for this comparison.")
    if not np.isfinite(values).all():
        raise ValueError("Annual changes contain a nonfinite value.")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd < 1e-10:
        raise ValueError("Annual changes have too little variation to fit a density.")

    folds = purged_folds(np.asarray(month_numbers))
    bandwidths = sd * np.geomspace(0.02, 2.0, 45)
    search = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=folds,
        scoring=mean_log_likelihood,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(values.reshape(-1, 1))
    bandwidth = float(search.best_params_["bandwidth"])
    grid = np.linspace(
        min(values.min(), mean - 4 * sd) - 0.5 * sd,
        max(values.max(), mean + 4 * sd) + 0.5 * sd,
        800,
    )
    return {
        "mean": mean,
        "sd": sd,
        "bandwidth": bandwidth,
        "folds": len(folds),
        "boundary": bool(search.best_index_ in (0, len(bandwidths) - 1)),
        "grid": grid,
        "kde": np.exp(search.best_estimator_.score_samples(grid.reshape(-1, 1))),
        "normal": stats.norm.pdf(grid, loc=mean, scale=sd),
    }


def make_figure(values, density, tail_cutoff):
    mean, sd = density["mean"], density["sd"]
    grid = density["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    left, right = axes
    left.hist(values, bins="fd", density=True, color="#5B8DB8",
              alpha=0.3, edgecolor="white", label="Observed changes")
    left.plot(grid, density["kde"], color="#8B1E2D", linewidth=2.5,
              label="Gaussian KDE")
    left.plot(grid, density["normal"], color="#174A7E", linewidth=2,
              linestyle="--", label="Normal benchmark")
    for cutoff in (mean - tail_cutoff * sd, mean + tail_cutoff * sd):
        left.axvline(cutoff, color="#666666", linewidth=1, linestyle=":")
    left.set(title="Distribution of annual FX changes",
             xlabel="Annual log change (log percentage points)", ylabel="Density")
    left.legend(fontsize=8, loc="upper left")

    theoretical, observed = stats.probplot(values, dist="norm", fit=False)
    right.scatter(theoretical, observed, s=15, alpha=0.7, color="#8B1E2D")
    right.plot(theoretical, mean + sd * theoretical, "--", color="#174A7E",
               linewidth=2, label="Normal reference")
    right.set(title="Normal Q–Q plot", xlabel="Standard normal quantile",
              ylabel="Observed annual log change (log percentage points)")
    right.legend(fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def main():
    import streamlit as st

    st.set_page_config(page_title="FX Annual Log Changes — Normal vs Fat Tails",
                       layout="wide")
    st.title("Annual FX changes: normal distribution and fat tails")
    st.latex(r"g_t = 100\,[\log(S_t)-\log(S_{t-12})]")
    st.caption(
        "Upload monthly exchange rates. With local currency per U.S. dollar, "
        "positive changes mean depreciation of the local currency. "
        "Log percentage points differ from ordinary percentage changes."
    )

    uploaded = st.file_uploader("Choose an exchange-rate file",
                                type=["xlsx", "xls", "csv"])
    if uploaded is None:
        st.info("Upload Colombian_Peso.xlsx, Polish_Zloty.xlsx, or another monthly file.")
        return

    try:
        raw = parse_input_file(uploaded)
    except Exception as exc:
        st.error(f"Problem reading the file: {exc}")
        return

    date_guess, rate_guess = guess_columns(raw)
    columns = list(raw.columns)
    with st.sidebar:
        st.header("Data settings")
        date_col = st.selectbox(
            "Date column", columns,
            index=columns.index(date_guess) if date_guess is not None else 0,
        )
        rate_col = st.selectbox(
            "Exchange-rate column", columns,
            index=columns.index(rate_guess) if rate_guess is not None else 1,
        )
        dayfirst = st.checkbox("Day first for ambiguous text dates", value=False)
        sampling = st.radio("Annual-change sample", [
            "All months (overlapping)", "One calendar month per year"
        ])
        selected_month = 12
        if sampling == "One calendar month per year":
            selected_month = st.selectbox(
                "Ending month", list(range(1, 13)), index=11,
                format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
            )
        tail_cutoff = st.slider("Tail threshold (standard deviations from the mean)",
                                1.0, 4.0, 2.0, 0.5)

    try:
        monthly = compute_annual_log_changes(raw, date_col, rate_col, dayfirst)
    except Exception as exc:
        st.error(f"Problem computing annual log changes: {exc}")
        return

    if monthly.attrs["ignored_rows"]:
        st.caption(f"Ignored {monthly.attrs['ignored_rows']} blank or source-note rows.")
    if monthly.attrs["missing_months"]:
        st.warning(
            f"{monthly.attrs['missing_months']} calendar months are missing. "
            "No prices were filled in. An annual change is included only when "
            "both its current price and its price exactly 12 months earlier exist."
        )

    sample = monthly.dropna(subset=["AnnualLogChange"]).copy()
    if sampling == "One calendar month per year":
        sample = sample.loc[sample["Date"].dt.month == selected_month].copy()
    values = sample["AnnualLogChange"].to_numpy(dtype=float)
    month_numbers = sample["Date"].dt.to_period("M").array.asi8

    try:
        cached_fit = st.cache_data(show_spinner=False)(fit_density)
        with st.spinner("Selecting the KDE bandwidth by cross-validation..."):
            density = cached_fit(values, month_numbers)
    except Exception as exc:
        st.error(f"Problem fitting the distributions: {exc}")
        return

    mean, sd = density["mean"], density["sd"]
    st.caption(
        f"{uploaded.name} | Annual changes ending "
        f"{sample['Date'].min():%b %Y} to {sample['Date'].max():%b %Y}"
    )
    metrics = st.columns(4)
    metrics[0].metric("Annual changes", f"{len(values):,}")
    metrics[1].metric("Mean (log pp)", f"{mean:.2f}")
    metrics[2].metric("Standard deviation (log pp)", f"{sd:.2f}")
    metrics[3].metric("Excess kurtosis", f"{stats.kurtosis(values, fisher=True, bias=False):.2f}")

    fig = make_figure(values, density, tail_cutoff)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The normal benchmark uses the sample mean and standard deviation. "
        f"KDE bandwidth: {density['bandwidth']:.3f} log percentage points. "
        f"Selected using {density['folds']} chronological validation blocks, "
        "excluding training changes within 11 months of each block."
    )
    st.caption(
        "KDE smoothing can widen the plotted tails. Compare observed tail "
        "frequencies and the Q–Q plot as well as the smoothed curve."
    )
    if density["boundary"]:
        st.warning("The chosen bandwidth is at the edge of the search range. "
                   "Interpret the smoothed density with care.")

    empirical_tail = float(np.mean(np.abs(values - mean) > tail_cutoff * sd))
    normal_tail = float(2 * stats.norm.sf(tail_cutoff))
    st.subheader(f"Changes more than {tail_cutoff:g} standard deviations from the mean")
    left, middle, right = st.columns(3)
    left.metric("Observed share", f"{empirical_tail:.2%}")
    middle.metric("Normal benchmark", f"{normal_tail:.2%}")
    right.metric("Observed / normal", f"{empirical_tail / normal_tail:.2f}×")
    st.caption(
        "These are descriptive comparisons. A zero observed tail share does not "
        "imply zero risk, and excess kurtosis alone does not establish fat tails."
    )
    if sampling == "All months (overlapping)":
        st.info(
            "Successive annual changes share 11 months of returns and are not "
            "independent observations. Try one calendar month per year to remove "
            "mechanical overlap; other serial dependence may remain."
        )

    with st.expander("Annual changes over time and underlying data"):
        st.line_chart(sample.set_index("Date")[["AnnualLogChange"]])
        st.dataframe(monthly)
    st.download_button(
        "Download selected annual changes (CSV)",
        data=sample.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_annual_log_changes.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
    out = df[[date_col, price_col]].copy()
    out[date_col] = coerce_datetime(out[date_col])
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col)
    out = out.set_index(date_col)
    out = out.loc[:, [price_col]].astype(float)
    out["log_price"] = np.log(out[price_col])
    out["annual_log_change"] = out["log_price"] - out["log_price"].shift(12)
    out = out.dropna(subset=["annual_log_change"])
    return out

# ===== KDE (Gaussian kernel) with cross-validated bandwidth =====

def fit_kde_cv_gaussian(x: np.ndarray, cv_folds: int = 5):
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0:
        raise ValueError("Standard deviation is zero; KDE cannot be fit.")

    z = ((x - mu) / sigma).reshape(-1, 1)
    bandwidths = np.linspace(0.1, 1.5, 25)

    n = len(z)
    k = max(2, min(cv_folds, max(2, n // 5)))
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=cv,
        n_jobs=-1
    )
    grid.fit(z)

    best_kde = grid.best_estimator_
    best_bw = float(grid.best_params_["bandwidth"])
    return best_kde, mu, sigma, best_bw

def evaluate_kde_pdf_on_grid(best_kde: KernelDensity, grid_x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    zgrid = ((grid_x - mu) / sigma).reshape(-1, 1)
    logpdf_z = best_kde.score_samples(zgrid)
    pdf_x = np.exp(logpdf_z) / sigma
    return pdf_x

def normal_pdf(x, mu, sigma):
    return stats.norm.pdf(x, loc=mu, scale=sigma)

# --- FIXED TAIL INTEGRATION (Option A) ---
def tail_prob_from_pdf(grid_x: np.ndarray, pdf_x: np.ndarray, mu: float, sigma: float, k: float = 1.96) -> float:
    left = mu - k * sigma
    right = mu + k * sigma
    left_mask = grid_x < left
    right_mask = grid_x > right
    area_left = float(np.trapz(pdf_x[left_mask], grid_x[left_mask])) if np.any(left_mask) else 0.0
    area_right = float(np.trapz(pdf_x[right_mask], grid_x[right_mask])) if np.any(right_mask) else 0.0
    return area_left + area_right

def tail_prob_normal(k: float = 1.96) -> float:
    return float(2 * (1 - stats.norm.cdf(k)))

# -------------------------------
# Forecast section (wrapped to avoid NameError if moved)
# -------------------------------

def render_forecast_section(mu, sigma, panel, price_col):
    # -------------------------------
    # Forecast: Spot path & 95% Normal CI by month
    # -------------------------------
    st.subheader("Forecast: 95% Normal-confidence interval for the spot by month")

    # Inputs
    default_spot = float(panel[price_col].iloc[-1]) if len(panel) else 1.0
    spot_now = st.number_input("Current spot (S₀)", min_value=0.0, value=round(default_spot, 6), format="%.6f")
    horizon_m = st.number_input("Horizon (months)", min_value=1, max_value=240, value=12, step=1)

    # Choose drift source
    drift_choice = st.radio(
        "Drift used in the CI",
        options=["Historical average (μ)", "User-specified annual drift"],
        index=0,
        horizontal=True,
        help="Drift is the annual log change. If you choose a custom drift, enter it as a decimal (e.g., 0.02 ≈ 2% per year)."
    )

    if drift_choice == "User-specified annual drift":
        user_mu_annual = st.number_input(
            "Annual log drift μ (decimal, e.g., 0.02 ≈ 2%)",
            value=float(mu),
            format="%.6f"
        )
        mu_used = float(user_mu_annual)
    else:
        mu_used = float(mu)

    # Compute only if valid
    if spot_now <= 0.0:
        st.warning("Please enter a positive current spot to compute the forecast.")
        return

    # Annual -> monthly scaling under Normal i.i.d. log changes assumption
    mu_month = mu_used / 12.0
    sigma_month = sigma / np.sqrt(12.0)

    # For month h: cumulative mean and std of log change
    months = np.arange(1, int(horizon_m) + 1, dtype=int)
    mu_h = months * mu_month
    sigma_h = sigma * np.sqrt(months / 12.0)  # same as sigma_month * sqrt(months)

    # Point forecast and 95% CI for the SPOT (level), using log-normal mapping
    point = spot_now * np.exp(mu_h)
    lower = spot_now * np.exp(mu_h - 1.96 * sigma_h)
    upper = spot_now * np.exp(mu_h + 1.96 * sigma_h)

    # Future dates (month starts) based on last data timestamp, if available
    last_date = panel.index.max() if isinstance(panel.index, pd.DatetimeIndex) and len(panel.index) else None
    if pd.notna(last_date):
        # Next month start then monthly
        future_dates = pd.date_range((last_date + pd.offsets.MonthBegin(1)).replace(day=1), periods=len(months), freq="MS")
    else:
        future_dates = pd.RangeIndex(1, len(months) + 1, name="Month")

    # Table
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "month_ahead": months,
        "spot_point": point,
        "spot_lower_95": lower,
        "spot_upper_95": upper
    })

    st.dataframe(
        forecast_df.assign(
            spot_point=lambda d: d["spot_point"].round(6),
            spot_lower_95=lambda d: d["spot_lower_95"].round(6),
            spot_upper_95=lambda d: d["spot_upper_95"].round(6),
        ),
        use_container_width=True
    )

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast table (CSV)",
        data=csv_bytes,
        file_name="fx_spot_normal_CI_forecast.csv",
        mime="text/csv"
    )

    # -------------------------------
    # Plot the forecast with CI band
    # (Legend below; extra space between plot and x-axis)
    # -------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.0))

    x_axis = np.arange(len(months))
    ax2.plot(x_axis, point, linewidth=2, label="Spot point forecast (Normal)")
    ax2.fill_between(x_axis, lower, upper, alpha=0.2, label="95% CI (Normal)")

    # Robust ticks (≈12 ticks max) — avoids slicing bugs
    step = max(1, len(x_axis) // 12)
    tick_idx = np.arange(0, len(x_axis), step)

    if isinstance(future_dates, pd.DatetimeIndex):
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([d.strftime("%Y-%m") for d in future_dates[tick_idx]], rotation=45, ha="right")
        ax2.set_xlabel("Forecast month")
    else:
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([str(m) for m in months[tick_idx]], rotation=0, ha="center")
        ax2.set_xlabel("Months ahead")

    ax2.set_ylabel("Spot")
    ax2.set_title("95% confidence interval implied by normal distribution")
    ax2.grid(True, linestyle=":", linewidth=0.8)

    # >>> Extra space between graph and x-axis labels <<<
    ax2.tick_params(axis="x", pad=10)   # more gap from axis line to tick labels
    ax2.xaxis.labelpad = 14            # more gap from tick labels to axis label
    ax2.margins(y=0.10)                # add vertical breathing room inside axes

    # Legend BELOW the chart, pushed well away from x-labels
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
        fontsize="small"
    )

    # Tight layout, then add generous bottom margin for labels + legend
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.46)

    st.pyplot(fig2)

# -------------------------------
# UI
# -------------------------------

st.title("Distribution of changes in exchange rates")

st.markdown(
    r"""
Upload **monthly** spot exchange rates (CSV or Excel).  
We compute the annual log change $\Delta \log S_t = \log S_t - \log S_{t-12}$, then compare a Normal model with a Kernel Density Estimation (Gaussian kernel, bandwidth via cross-validation) and report tail risks.
"""
)

uploaded = st.file_uploader("Upload CSV or Excel with a Date column and a Spot Rate column", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Awaiting file upload…")
    st.stop()

try:
    raw = parse_input_file(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if raw.empty:
    st.error("The uploaded file is empty.")
    st.stop()

date_guess, price_guess = guess_columns(raw)

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Date column", options=list(raw.columns),
                            index=(list(raw.columns).index(date_guess) if date_guess in raw.columns else 0))
with col2:
    price_col = st.selectbox("Spot rate column", options=list(raw.columns),
                             index=(list(raw.columns).index(price_guess) if price_guess in raw.columns else (1 if len(raw.columns) > 1 else 0)))

try:
    panel = compute_annual_log_changes(raw, date_col, price_col)
except Exception as e:
    st.error(f"Problem computing annual log changes: {e}")
    st.stop()

if panel["annual_log_change"].empty:
    st.warning("Not enough data to compute 12-month log changes.")
    st.stop()

x = panel["annual_log_change"].dropna().values
mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
skew, kurt = float(stats.skew(x, bias=False)), float(stats.kurtosis(x, fisher=False, bias=False))

jb_stat, jb_p = stats.jarque_bera(x)
sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)

try:
    kde_cv, mu_kde, sigma_kde, best_bw = fit_kde_cv_gaussian(x, cv_folds=5)
except Exception as e:
    st.error(f"KDE fit failed: {e}")
    st.stop()

st.subheader("Summary statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean (μ)", f"{mu:.4f}")
c2.metric("Std (σ)", f"{sigma:.4f}")
c3.metric("Skewness (Normal = 0)", f"{skew:.4f}")
c4.metric("Kurtosis (Normal = 3)", f"{kurt:.4f}")
st.caption(f"Best KDE bandwidth (z-scale): {best_bw:.3f}")

st.subheader("Normality tests")
t1, t2 = st.columns(2)
t1.write("**Jarque–Bera**")
t1.write(f"stat = {jb_stat:.3f}, p = {jb_p:.3g}")
t2.write("**Shapiro–Wilk**")
t2.write("n ≤ 5000 required" if np.isnan(sh_stat) else f"stat = {sh_stat:.3f}, p = {sh_p:.3g}")

# -------------------------------
# Distribution Plot (with toggles)
# -------------------------------
st.subheader("Distribution of changes in the spot exchange rate")
ocol1, ocol2, ocol3 = st.columns(3)
show_normal = ocol1.checkbox("Show Normal distribution", True)
show_kde = ocol2.checkbox("Show Kernel Density Estimation (CV Gaussian)", True)
show_ci = ocol3.checkbox("Show 95% Normal interval (μ ± 1.96σ)", True)

pad_stds = 1.0
xmin, xmax = np.min(x), np.max(x)
lo = min(mu - 4 * sigma, xmin) - pad_stds * sigma
hi = max(mu + 4 * sigma, xmax) + pad_stds * sigma
grid = np.linspace(lo - 0.5 * sigma, hi + 0.5 * sigma, 4000)

pdf_norm = normal_pdf(grid, mu, sigma) if show_normal else None
pdf_kde = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde) if show_kde else None

fig1, ax1 = plt.subplots(figsize=(7, 4.25))
ax1.hist(x, bins="auto", density=True, alpha=0.6, edgecolor="black", label="Histogram")

if show_ci:
    left, right = mu - 1.96 * sigma, mu + 1.96 * sigma
    ax1.axvspan(left, right, alpha=0.15, label="95% Normal interval (μ ± 1.96σ)")
if show_normal:
    ax1.plot(grid, pdf_norm, linewidth=2, label="Normal PDF")
if show_kde:
    ax1.plot(grid, pdf_kde, linewidth=2, linestyle="--", label="Kernel Density Estimation (CV Gaussian) PDF")

ax1.set_xlabel("Annual log change")
ax1.set_ylabel("Density")
ax1.set_title("Histogram with Optional Normal / KDE Overlays")
ax1.grid(True, linestyle=":", linewidth=0.8)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize="small")
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
st.pyplot(fig1)

# -------------------------------
# Tail probabilities beyond μ ± 1.96σ
# -------------------------------
st.subheader("Tail probabilities beyond μ ± 1.96σ")
p_tail_norm = tail_prob_normal(1.96)
pdf_kde_full = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde)
p_tail_kde = tail_prob_from_pdf(grid, pdf_kde_full, mu, sigma, 1.96)
threshold = 1.96 * sigma
p_tail_emp = float(np.mean(np.abs(x - mu) > threshold))

t1, t2, t3 = st.columns(3)
t1.metric("Normal model", f"{p_tail_norm:.4f}")
t2.metric("KDE (CV Gaussian)", f"{p_tail_kde:.4f}")
t3.metric("Empirical proportion", f"{p_tail_emp:.4f}")
st.caption("Under a Normal distribution, P(|Z|>1.96) ≈ 0.0500. Differences vs. CV Gaussian KDE highlight fat/thin tails.")

# -------------------------------
# Forecast: Spot path & 95% Normal CI by month (with drift choice)
# -------------------------------
render_forecast_section(mu, sigma, panel, price_col)

# -------------------------------
# Diagnostics (temporary; safe to delete/comment out)
# -------------------------------
# with st.expander("Diagnostics (temporary; safe to delete)"):
#     total_area = float(np.trapz(pdf_kde_full, grid))
#     st.write(f"DEBUG — KDE total area over grid: **{total_area:.4f}**")
#     st.write(f"DEBUG — grid range: **[{grid.min():.6f}, {grid.max():.6f}]**")
#     st.write(f"DEBUG — μ = {mu:.6f}, σ = {sigma:.6f}, 95% bounds: "
#              f"[{(mu - 1.96*sigma):.6f}, {(mu + 1.96*sigma):.6f}]")
#     st.write(f"DEBUG — Tail (Normal): **{p_tail_norm:.4f}**")
#     st.write(f"DEBUG — Tail (KDE, trapezoid fixed): **{p_tail_kde:.4f}**")
#     st.write(f"DEBUG — Tail (Empirical): **{p_tail_emp:.4f}**")
#
#     try:
#         z_samp = kde_cv.sample(100_000, random_state=0)
#         x_samp = mu_kde + sigma_kde * z_samp.ravel()
#         p_tail_mc = float(np.mean(np.abs(x_samp - mu) > 1.96 * sigma))
#         st.write(f"DEBUG — Tail (KDE Monte Carlo ~100k): **{p_tail_mc:.4f}**")
#     except Exception as e:
#         st.write(f"DEBUG — KDE Monte Carlo sampling failed: {e}")
#
#     if st.checkbox("Run bandwidth sensitivity check (z-scale)", value=False):
#         bws = [0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20]
#         rows = []
#         z = ((x - mu) / sigma).reshape(-1, 1)
#         for bw in bws:
#             kde_tmp = KernelDensity(kernel="gaussian", bandwidth=bw).fit(z)
#             pdf_tmp = evaluate_kde_pdf_on_grid(kde_tmp, grid, mu, sigma)
#             tail_tmp = tail_prob_from_pdf(grid, pdf_tmp, mu, sigma, 1.96)
#             area_tmp = float(np.trapz(pdf_tmp, grid))
#             rows.append({"bandwidth_z": bw, "kde_tail": tail_tmp, "total_area": area_tmp})
#         st.dataframe(pd.DataFrame(rows))
