# CyclesSpectralAnalysis.py
# Streamlit app for spectral analysis of a monthly time series (cycles in years)

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import detrend, windows
import plotly.express as px

st.set_page_config(page_title="Spectral Analysis (Cycles in Years)", layout="wide")

st.title("Spectral Analysis of a Monthly Time Series")
st.markdown(
    """
This app takes a **monthly** time series, computes its spectrum, and shows you
which **cycle lengths (in years)** have the most power.

**Steps performed:**
1. Read the series and sort by date.
2. Clean the data (remove NaNs / infinities).
3. Detrend the series (optional).
4. Apply a Hann window (optional).
5. Compute the FFT and periodogram.
6. Convert frequencies (cycles/year) into periods in years.
"""
)

# --- File upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Choose columns")

    # Guess date column if possible
    default_date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            default_date_col = c
            break

    if default_date_col is not None:
        date_index = df.columns.get_loc(default_date_col)
    else:
        date_index = 0  # fallback

    date_col = st.selectbox(
        "Date column (monthly index)",
        df.columns,
        index=date_index
    )

    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        st.error("No value columns found (only a date column). Please upload a CSV with at least one numeric series column.")
        st.stop()

    value_col = st.selectbox("Value column (time series)", value_cols)

    # Parse dates and sort
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Could not parse the date column '{date_col}'. Error: {e}")
        st.stop()

    df = df.sort_values(date_col).reset_index(drop=True)

    # --- Cleaning and numeric conversion ---
    st.subheader("2. Data cleaning")

    # Force numeric, coercing bad values to NaN
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Replace infinities with NaN, then drop NaNs
    df_clean = df[[date_col, value_col]].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    removed_rows = len(df) - len(df_clean)
    if removed_rows > 0:
        st.info(f"Removed {removed_rows} rows with NaN or infinite values in '{value_col}'.")

    if len(df_clean) < 16:
        st.error(
            "After cleaning, fewer than 16 valid observations remain. "
            "Please provide a longer or cleaner series."
        )
        st.stop()

    # Use the cleaned data going forward
    df = df_clean.sort_values(date_col).reset_index(drop=True)
    x = df[value_col].values.astype(float)
    N = len(x)

    st.write("First rows of the cleaned series:")
    st.dataframe(df[[date_col, value_col]].head())
    st.write(f"Number of valid observations after cleaning: **{N}**")

    # --- Preprocessing options ---
    st.subheader("3. Preprocessing")

    detrend_type = st.selectbox(
        "Detrend type",
        ["linear", "constant (remove mean)", "none"],
        index=0
    )
    apply_window = st.checkbox("Apply Hann window", value=True)

    x_proc = x.copy()

    # Safety check before detrending
    if not np.all(np.isfinite(x_proc)):
        st.error(
            "The series still contains NaN or infinite values even after cleaning. "
            "Please check your data."
        )
        st.stop()

    # Detrend
    if detrend_type == "linear":
        x_proc = detrend(x_proc, type="linear")
    elif detrend_type == "constant (remove mean)":
        x_proc = detrend(x_proc, type="constant")
    # else "none": do nothing

    # Windowing
    if apply_window:
        w = windows.hann(N)
        x_proc = x_proc * w

    # Final safety check
    if not np.all(np.isfinite(x_proc)):
        st.error(
            "The processed series (after detrending/windowing) contains NaN or infinite values. "
            "This should not happen; please inspect your data."
        )
        st.stop()

    # --- FFT and spectrum ---
    st.subheader("4. Spectrum and cycles in years")

    # Sample spacing in years: monthly data => 1/12 year
    d = 1.0 / 12.0

    # Real FFT (only nonnegative frequencies)
    X = np.fft.rfft(x_proc)
    freqs = np.fft.rfftfreq(N, d=d)  # frequencies in cycles per year

    # Periodogram (power)
    power = np.abs(X) ** 2

    # Remove the zero frequency (mean) to avoid infinite period
    positive = freqs > 0
    freqs_pos = freqs[positive]
    power_pos = power[positive]

    if len(freqs_pos) == 0:
        st.error("No positive frequencies found (this should not happen with reasonable data).")
        st.stop()

    # Convert to period in years
    periods_years = 1.0 / freqs_pos

    # --- Period range filter ---
    st.markdown("**Filter the period range (in years)**")

    # Reasonable defaults: 0.5 to 20 years
    min_period = st.number_input("Min period (years)", value=0.5, step=0.1)
    max_period = st.number_input("Max period (years)", value=20.0, step=0.5)

    if min_period <= 0:
        st.error("Min period must be > 0.")
        st.stop()
    if max_period <= min_period:
        st.error("Max period must be greater than min period.")
        st.stop()

    mask_range = (periods_years >= min_period) & (periods_years <= max_period)
    periods_plot = periods_years[mask_range]
    power_plot = power_pos[mask_range]

    if len(periods_plot) == 0:
        st.info(
            "No frequencies fall in the selected period range. "
            "Try widening the range."
        )
        st.stop()

    # Sort by period for nicer plotting (short to long)
    sort_idx = np.argsort(periods_plot)
    periods_plot = periods_plot[sort_idx]
    power_plot = power_plot[sort_idx]

    # --- Plot using Plotly ---
    spec_df = pd.DataFrame({
        "Period (years)": periods_plot,
        "Power": power_plot
    })

    fig = px.line(
        spec_df,
        x="Period (years)",
        y="Power",
        title="Spectral Power by Period (Years)"
    )
    fig.update_layout(
        xaxis_title="Period (years)",
        yaxis_title="Power (arbitrary units)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Show top peaks ---
    st.subheader("5. Dominant cycles (peaks)")

    n_peaks = st.slider("Number of peaks to report", min_value=1, max_value=10, value=5)

    # Find peaks: simply take the top n_peaks by power within the selected range
    idx_sorted = np.argsort(power_plot)[::-1]  # descending power
    n_report = min(n_peaks, len(idx_sorted))
    idx_top = idx_sorted[:n_report]

    peaks_periods = periods_plot[idx_top]
    peaks_power = power_plot[idx_top]

    peaks_df = pd.DataFrame({
        "Rank": np.arange(1, n_report + 1),
        "Period (years)": peaks_periods,
        "Power": peaks_power
    }).sort_values("Rank")

    st.write(
        "These are the most powerful cycles (largest spectral peaks) "
        f"in the selected period range [{min_period}, {max_period}] years:"
    )
    st.dataframe(peaks_df)

else:
    st.info("Upload a CSV file to begin.")
