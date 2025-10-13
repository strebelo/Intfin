#!/usr/bin/env python3
# advanced_cashflow_app.py
# Streamlit app: Simulate FX and plot expected cash flow over time (advanced version)
# Strategies: No hedge, Forward hedge (constant fraction), Option hedge (puts on FX) via Monte Carlo.
# Shows expected cash flow over time for the selected strategy and parameters.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm

st.set_page_config(page_title="Expected Cash Flow (Advanced)", layout="wide")

st.title("Expected Cash Flow Over Time — Advanced Version")

with st.sidebar:
    st.header("Core Settings")
    T = st.number_input("Horizon (periods)", min_value=4, max_value=120, value=24, step=1)
    periods_per_year = st.selectbox("Periods per year", options=[12, 4, 2, 1, 52], index=0)
    revenue_fcy = st.number_input("Revenue per period (FCY)", min_value=0.0, value=1_000_000.0, step=10_000.0, format="%.2f")
    S0 = st.number_input("Initial FX rate (HCY/FCY)", min_value=0.0001, value=1.10, step=0.01, format="%.4f")
    mu_annual = st.number_input("Drift μ (annualized, %)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1) / 100.0
    sigma_annual = st.number_input("Volatility σ (annualized, %)", min_value=0.01, max_value=200.0, value=12.0, step=0.1) / 100.0
    n_sims = st.number_input("Simulations", min_value=100, max_value=50000, value=10000, step=100)
    seed = st.number_input("Random seed (0 = none)", min_value=0, max_value=1_000_000, value=0, step=1)

    st.header("Rates (for forwards & discounting)")
    r_dom_annual = st.number_input("Home risk-free (annual, %)", min_value=-10.0, max_value=50.0, value=4.0, step=0.1)/100.0
    r_for_annual = st.number_input("Foreign risk-free (annual, %)", min_value=-10.0, max_value=50.0, value=2.0, step=0.1)/100.0

    st.header("Hedging Strategy")
    strategy = st.selectbox("Strategy", ["No hedge", "Forward hedge (constant fraction)", "Put option hedge (constant fraction)"])
    hedge_frac = st.slider("Hedge fraction", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.header("Options (if using puts)")
    moneyness = st.selectbox("Put strike as % of forward", options=[80, 90, 95, 100, 105], index=3)
    # Black-Scholes uses annualized vol and time in years; we'll value each-period put on S_t with maturity = dt.
    # Premium is paid upfront at each period on notional = hedge_frac * revenue_fcy * S_forward (converted to HCY).

dt = 1.0 / periods_per_year
mu = mu_annual
sigma = sigma_annual
r_dom = r_dom_annual
r_for = r_for_annual

if seed != 0:
    np.random.seed(int(seed))

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

def forward_rate(S_prev, r_dom, r_for, dt):
    # Covered interest parity
    return S_prev * (1 + r_dom*dt) / (1 + r_for*dt)

def bs_put_price(S, K, r_dom, r_for, sigma, tau):
    # Black-Scholes for FX options (domestic discounting, foreign as dividend yield)
    if sigma * sqrt(tau) == 0:
        return max(K - S, 0.0)
    d1 = (log(S/K) + (r_dom - r_for + 0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = d1 - sigma*sqrt(tau)
    # Put price = K*e^{-r_d tau} * N(-d2) - S*e^{-r_f tau} * N(-d1)
    return K*np.exp(-r_dom*tau) * norm.cdf(-d2) - S*np.exp(-r_for*tau) * norm.cdf(-d1)

# Simulate FX paths
S_paths = simulate_gbm_paths(S0, mu, sigma, T, dt, n_sims)  # shape (n_sims, T+1)

# Prepare arrays
CF_nohedge = revenue_fcy * S_paths[:, 1:]  # HCY
expected_CF_nohedge = CF_nohedge.mean(axis=0)

# Forward hedge: at each period t-1, lock in hedge_frac of next period's revenue at forward rate F_{t-1,t}
# Realized CF_t = (1-h) * rev * S_t + h * rev * F_{t-1,t}
F_matrix = np.zeros((n_sims, T))
for t in range(1, T+1):
    F_matrix[:, t-1] = forward_rate(S_paths[:, t-1], r_dom, r_for, dt)

CF_forward = (1-hedge_frac) * revenue_fcy * S_paths[:, 1:] + hedge_frac * revenue_fcy * F_matrix
expected_CF_forward = CF_forward.mean(axis=0)

# Put option hedge: buy a put on S_t with maturity dt and strike = m% of forward F_{t-1,t}
# Premium paid upfront at t-1 (discounted value in HCY); payoff at t is max(K - S_t, 0) on notional = hedge_frac * revenue_fcy
# Realized CF_t = (1-h) * rev * S_t + h * rev * (S_t + payoff) - premium_cost_upfront_discounted_forwarded_to_t
# Since payoff adds only when S_t<K, equivalently: h * rev * (S_t + (K - S_t)^+) = h*rev*(max(S_t, K))
# We'll compute cash flow at t net of premium; premium at t-1 is: price_per_unit * notional_size_in_FCY? Careful:
# We treat the option written on 1 FCY worth of S; so notional in FCY = hedge_frac*revenue_fcy; premium in HCY at t-1 = price(S_{t-1}, K, ...)
# We'll carry premium to time t with (1 + r_dom*dt).
K_matrix = moneyness/100.0 * F_matrix
premium_matrix = np.zeros((n_sims, T))
for t in range(1, T+1):
    S_prev = S_paths[:, t-1]
    K = K_matrix[:, t-1]
    # Price each path's put:
    prices = np.array([bs_put_price(S_prev[i], K[i], r_dom, r_for, sigma, dt) for i in range(n_sims)])
    premium_matrix[:, t-1] = prices * hedge_frac * revenue_fcy  # HCY at t-1

# Realized CF under put hedge:
# At time t: (1-h)*rev*S_t + h*rev*max(S_t, K)  - premium*(1+r_dom*dt)
payoff_component = np.maximum(S_paths[:, 1:], K_matrix)
CF_put = (1-hedge_frac) * revenue_fcy * S_paths[:, 1:] + hedge_frac * revenue_fcy * payoff_component - premium_matrix * (1 + r_dom*dt)
expected_CF_put = CF_put.mean(axis=0)

# Assemble results
df = pd.DataFrame({"Period": np.arange(1, T+1)})
df["No hedge (E[CF])"] = expected_CF_nohedge
df["Forward hedge (E[CF])"] = expected_CF_forward
df["Put hedge (E[CF])"] = expected_CF_put

st.subheader("Expected Cash Flow Over Time")

col1, col2 = st.columns([2,1])
with col1:
    fig, ax = plt.subplots()
    if strategy == "No hedge":
        ax.plot(df["Period"], df["No hedge (E[CF])"], label="No hedge")
    elif strategy == "Forward hedge (constant fraction)":
        ax.plot(df["Period"], df["Forward hedge (E[CF])"], label="Forward hedge")
    else:
        ax.plot(df["Period"], df["Put hedge (E[CF])"], label="Put hedge")
    ax.set_xlabel("Period")
    ax.set_ylabel("Expected Cash Flow (Home Currency)")
    ax.set_title(f"Expected Cash Flow Over Time — {strategy}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with col2:
    st.markdown("### Key Averages")
    st.write(pd.DataFrame({
        "Strategy": ["No hedge", "Forward hedge", "Put hedge"],
        "Mean per period (HCY)": [
            df["No hedge (E[CF])"].mean(),
            df["Forward hedge (E[CF])"].mean(),
            df["Put hedge (E[CF])"].mean()
        ]
    }).style.format({"Mean per period (HCY)": "{:,.2f}"}))

st.subheader("Expected Cash Flow Table")
st.dataframe(df.style.format({
    "No hedge (E[CF])": "{:,.2f}",
    "Forward hedge (E[CF])": "{:,.2f}",
    "Put hedge (E[CF])": "{:,.2f}",
}), use_container_width=True)

st.caption(
    "Model: GBM for FX; covered interest parity for forwards; Black–Scholes for period puts with foreign rate as dividend yield. "
    "Expected cash flow computed as the mean across Monte Carlo simulations. "
    "Premiums are paid upfront each period and carried to the cash flow date at the domestic risk-free rate."
)
