# supercycle.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------

def ar1_path(T, rho, mu=1.0, sigma=0.05, seed=0, positive_bump_periods=5, bump_size=1.0):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    if positive_bump_periods > 0:
        eps[:positive_bump_periods] = bump_size
    alpha = np.empty(T)
    alpha[0] = mu
    for t in range(1, T):
        alpha[t] = (1 - rho) * mu + rho * alpha[t - 1] + sigma * eps[t]
    alpha[0] = (1 - rho) * mu + rho * mu + sigma * eps[0]
    return alpha

def equilibrium_price_path(T, k, alpha_c, alpha_i, theta_c, theta_i, p_init=None):
    if p_init is None:
        p_init = np.ones(k)
    p = np.empty(T)
    p[:k] = p_init
    for t in range(k, T):
        denom = alpha_i[t - k] * (p[t - k] ** theta_i)
        denom = max(denom, 1e-10)
        p_t = (alpha_c[t] / denom) ** (1.0 / theta_c)
        p[t] = max(p_t, 1e-10)
    return p

def aggregate_investment(alpha_i, p, theta_i):
    return alpha_i * (p ** theta_i)

def individual_path(T, k, r, p, i_rule, a0=0.0, i_hist=None):
    if i_hist is None:
        i_hist = np.zeros(k)
    i = np.empty(T)
    a = np.empty(T + 1)
    a[0] = a0
    for t in range(T):
        i[t] = max(i_rule(t, p[t]), 0.0)
        delivered = i[t - k] if t - k >= 0 else i_hist[t - k]
        cash_in = p[t] * delivered
        a[t + 1] = a[t] * (1.0 + r) + cash_in - i[t]
    return i, a

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Investment-Lag Model", layout="wide")
st.title("Investment-Lag Commodity Model")

with st.sidebar:
    T = st.number_input("Horizon T", 10, 1000, 200, 10)
    k = st.number_input("Lag k (periods)", 1, 24, 4, 1)

    theta_c = st.number_input("Demand elasticity θ_c", 0.1, 10.0, 1.0, 0.1)
    theta_i = st.number_input("Investment elasticity θ_i", 0.1, 10.0, 1.0, 0.1)

    rho_c = st.slider("ρ_c (demand persistence)", 0.0, 0.99, 0.8, 0.01)
    rho_i = st.slider("ρ_i (investment persistence)", 0.0, 0.99, 0.8, 0.01)
    mu_c = st.number_input("μ_c (mean demand)", 0.01, 10.0, 1.0, 0.01)
    mu_i = st.number_input("μ_i (mean investment)", 0.01, 10.0, 1.0, 0.01)
    sigma_c = st.number_input("σ_c (demand shocks)", 0.0, 2.0, 0.05, 0.01)
    sigma_i = st.number_input("σ_i (investment shocks)", 0.0, 2.0, 0.05, 0.01)

    bump_periods = st.slider("Positive shock periods", 0, 50, 10, 1)
    bump_size = st.number_input("Bump size", 0.0, 5.0, 1.5, 0.1)
    seed = st.number_input("Random seed", 0, 10_000, 1234, 1)

    a0 = st.number_input("Initial assets a₀", -1e6, 1e6, 0.0, 100.0)
    p_init_val = st.number_input("Initial price (×k)", 1e-6, 1e6.0, 1.0, 0.1)
    i_hist_val = st.number_input("Pre-sample investment (×k)", 0.0, 1e6, 0.0, 10.0)

    rule_type = st.selectbox("Decision Rule", ["Scaled aggregate", "Price power", "Constant"], 0)
    chi = st.number_input("χ (scale factor)", 0.0, 5.0, 1.0, 0.1)
    gamma = st.number_input("γ (price coefficient)", 0.0, 5.0, 1.0, 0.1)
    phi = st.number_input("φ (price exponent)", 0.0, 5.0, 1.0, 0.1)
    const_i = st.number_input("Constant i_t", 0.0, 1e6, 0.0, 10.0)
    r = st.number_input("Interest rate r", -0.99, 10.0, 0.01, 0.01)

# -----------------------------
# Simulation
# -----------------------------

alpha_c = ar1_path(T, rho_c, mu_c, sigma_c, seed + 1, bump_periods, bump_size)
alpha_i = ar1_path(T, rho_i, mu_i, sigma_i, seed + 2, bump_periods, bump_size)
p = equilibrium_price_path(T, k, alpha_c, alpha_i, theta_c, theta_i, np.full(k, p_init_val))
I_agg = aggregate_investment(alpha_i, p, theta_i)

if rule_type == "Scaled aggregate":
    i_rule = lambda t, p_t: chi * I_agg[t]
elif rule_type == "Price power":
    i_rule = lambda t, p_t: gamma * (p_t ** phi)
else:
    i_rule = lambda t, p_t: const_i

inherited_i_hist = np.full(k, i_hist_val)
i_ind, a_path = individual_path(T, k, r, p, i_rule, a0, inherited_i_hist)

df = pd.DataFrame({
    "t": np.arange(T),
    "p_t": p,
    "I_agg": I_agg,
    "i_t": i_ind,
    "a_t": a_path[1:],
})

# -----------------------------
# Plots
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(df["t"], df["p_t"], linewidth=2)
    ax1.set_xlabel("t")
    ax1.set_ylabel("Price p_t")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(df["t"], df["I_agg"], '--', label="Aggregate I_t", linewidth=1.5)
    ax2.plot(df["t"], df["i_t"], label="Individual i_t", linewidth=2)
    ax2.set_xlabel("t")
    ax2.set_ylabel("Investment")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

with col3:
    fig3, ax3 = plt.subplots()
    ax3.plot(df["t"], df["a_t"], linewidth=2)
    ax3.set_xlabel("t")
    ax3.set_ylabel("Assets a_t")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, clear_figure=True)

# -----------------------------
# Data table and download
# -----------------------------

with st.expander("Show data"):
    st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("Download data", csv, "simulation.csv", "text/csv")
