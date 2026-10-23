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
