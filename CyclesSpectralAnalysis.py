# CyclesSpectralAnalysis.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import detrend, windows
import plotly.express as px

st.set_page_config(page_title="Spectral Analysis (Cycles in Years)", layout="wide")

st.title("Spectral Analysis of a Monthly Time Series")
st.markdown(
    """
This app takes a monthly time series, computes its spectrum, and shows you
which **cycle lengths (in years)** have the most power.

**Steps performed:**
1. Read the series and sort by date.
2. Detrend the series (linear detrend).
3. Apply a Hann window.
4. Compute the FFT and periodogram.
5. Convert frequencies (cycles/year) into periods in years.
"""
)

# --- File upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Choose columns")

    # Guess date and value columns
    default_date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            default_date_col = c
            break

    date_col = st.selectbox("Date column (monthly index)", df.columns, index=(df.columns.get_loc(default_date_col) if default_date_col in df.columns else 0))
    value_cols = [c for c in df.columns if c != date_col]
    value_col = st.selectbox("Value column (time series)", value_cols)

    # Parse dates and sort
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Show a quick preview
    st.write("First rows of the selected series:")
    st.dataframe(df[[date_col, value_col]].head())

    # Extract the series
    x = df[value_col].astype(float).values
    N = len(x)

    if N < 16:
        st.warning("You need at least ~16 observations to get anything meaningful.")
    else:
        st.subheader("2. Preprocessing")

        detrend_type = st.selectbox("Detrend type", ["linear", "constant (remove mean)", "none"])
        apply_window = st.checkbox("Apply Hann window", value=True)

        x_proc = x.copy()
        if detrend_type == "linear":
            x_proc = detrend(x_proc, type="linear")
        elif detrend_type == "constant (remove mean)":
            x_proc = detrend(x_proc, type="constant")
        # else 'none': do nothing

        if apply_window:
            w = windows.hann(N)
            x_proc = x_proc * w

        st.write("Number of observations:", N)

        # --- FFT and spectrum ---
        st.subheader("3. Spectrum and cycles in years")

        # Sample spacing in years: monthly data => 1/12 year
        d = 1.0 / 12.0

        # Real FFT (only nonnegative frequencies)
        X = np.fft.rfft(x_proc)
        freqs = np.fft.rfftfreq(N, d=d)  # frequencies in cycles per year

        # Periodogram (power)
        # Scaling choice is not critical for identifying peaks, so we use |X|^2
        power = np.abs(X) ** 2

        # Remove the zero frequency (mean) to avoid infinite period
        positive = freqs > 0
        freqs_pos = freqs[positive]
        power_pos = power[positive]

        # Convert to period in years
        periods_years = 1.0 / freqs_pos

        # Let user restrict period range (years)
        st.markdown("**Filter the period range (in years)**")
        min_period = st.number_input("Min period (years)", value=0.5, step=0.1)
        max_period = st.number_input("Max period (years)", value=20.0, step=0.5)

        mask_range = (periods_years >= min_period) & (periods_years <= max_period)
        periods_plot = periods_years[mask_range]
        power_plot = power_pos[mask_range]

        # Sort by period (optional, for nicer plotting from short to long)
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
        fig.update_layout(xaxis_title="Period (years)", yaxis_title="Power (arbitrary units)")
        st.plotly_chart(fig, use_container_width=True)

        # --- Show top peaks ---
        st.subheader("4. Dominant cycles (peaks)")

        n_peaks = st.slider("Number of peaks to report", min_value=1, max_value=10, value=5)

        # Find peaks by taking top n_peaks values in the filtered range
        if len(power_plot) > 0:
            idx_sorted = np.argsort(power_plot)[::-1]  # descending
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
            st.info("No frequencies in the selected period range. Try widening it.")
else:
    st.info("Upload a CSV file to begin.")
