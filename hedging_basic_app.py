#!/usr/bin/env python3
# basic_cashflow_app.py
# Streamlit app: Simulate FX and plot expected cash flow over time (basic version)
# Uses Monte Carlo GBM for FX; cash flow = foreign revenue * spot FX.
# No hedging in this basic version.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Expected Cash Flow (Basic)", layout="centered")

st.title("Expected Cash Flow Over Time — Basic Version")

with st.sidebar:
    st.header("Simulation Settings")
    T = st.number_input("Horizon (periods)", min_value=4, max_value=120, value=24, step=1)
    revenue_fcy = st.number_input("Revenue per period (FCY)", min_value=0.0, value=1_000_000.0, step=10_000.0, format="%.2f")
    S0 = st.number_input("Initial FX rate (HCY/FCY)", min_value=0.0001, value=1.10, step=0.01, format="%.4f")
    mu_annual = st.number_input("Drift μ (annualized, %)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1) / 100.0
    sigma_annual = st.number_input("Volatility σ (annualized, %)", min_value=0.01, max_value=200.0, value=12.0, step=0.1) / 100.0
    periods_per_year = st.selectbox("Periods per year", options=[12, 4, 2, 1, 52], index=0)
    n_sims = st.number_input("Simulations", min_value=100, max_value=20000, value=5000, step=100)
    seed = st.number_input("Random seed (optional, 0 = none)", min_value=0, max_value=1_000_000, value=0, step=1)

dt = 1.0 / periods_per_year
mu = mu_annual
sigma = sigma_annual

if seed != 0:
    np.random.seed(int(seed))

# Simulate GBM for FX
# dS/S = mu dt + sigma dW
def simulate_gbm_paths(S0, mu, sigma, T, dt, n_sims):
    n_steps = int(T)
    S = np.empty((n_sims, n_steps+1), dtype=float)
    S[:, 0] = S0
    drift = (mu - 0.5 * sigma**2) * dt
    shock_scale = sigma * np.sqrt(dt)
    for t in range(1, n_steps+1):
        Z = np.random.normal(size=n_sims)
        S[:, t] = S[:, t-1] * np.exp(drift + shock_scale * Z)
    return S

S_paths = simulate_gbm_paths(S0, mu, sigma, T, dt, n_sims)

# Cash flow in home currency each period
CF_paths = revenue_fcy * S_paths[:, 1:]  # periods 1..T
expected_CF = CF_paths.mean(axis=0)

df = pd.DataFrame({
    "Period": np.arange(1, T+1),
    "Expected Cash Flow (HCY)": expected_CF
})

st.subheader("Expected Cash Flow Over Time")
fig, ax = plt.subplots()
ax.plot(df["Period"], df["Expected Cash Flow (HCY)"])
ax.set_xlabel("Period")
ax.set_ylabel("Expected Cash Flow (Home Currency)")
ax.set_title("Expected Cash Flow Over Time (Mean across simulations)")
st.pyplot(fig, clear_figure=True)

st.subheader("Summary Table")
st.dataframe(df.style.format({"Expected Cash Flow (HCY)": "{:,.2f}"}), use_container_width=True)

st.caption("Model: GBM for FX. Expected cash flow computed as mean revenue×spot across Monte Carlo paths. "
           "Basic version omits hedging and financing effects.")
