# supercycle.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------

def ar1_path(T, rho, mu=1.0, sigma=0.05, seed=0, positive_bump_periods=5, bump_size=1.0):
    """
    Generate an AR(1) path for alpha_t with optional initial positive shock series.
    alpha_t = (1-rho)*mu + rho*alpha_{t-1} + sigma*eps_t

    We implement a deterministic positive bump for the first 'positive_bump_periods' periods
    by setting eps_t = +bump_size for t < positive_bump_periods, else eps_t ~ N(0,1).
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    if positive_bump_periods > 0:
        eps[:positive_bump_periods] = bump_size

    alpha = np.empty(T)
    alpha[0] = mu  # start at mean; bump comes via eps[0]
    for t in range(1, T):
        alpha[t] = (1 - rho) * mu + rho * alpha[t - 1] + sigma * eps[t]
    # Apply the first step explicitly using eps[0]
    alpha[0] = (1 - rho) * mu + rho * mu + sigma * eps[0]
    return alpha


def equilibrium_price_path(T, k, alpha_c, alpha_i, theta_c, theta_i, p_init=None):
    """
    From market clearing:
        alpha_c,t * p_t^{-theta_c} = alpha_i,t-k * p_{t-k}^{theta_i}
    Solve forward for p_t given p_{t-k}.

    For t < k, we use provided p_init (array of length k). If None, use ones.
    """
    if p_init is None:
        p_init = np.ones(k)
    assert len(p_init) == k, "p_init must have length k"

    p = np.empty(T)
    # Fill first k with provided history
    p[:k] = p_init

    for t in range(k, T):
        denom = alpha_i[t - k] * (p[t - k] ** theta_i)
        # Guard against division by zero or negative
        denom = max(denom, 1e-10)
        p_t = (alpha_c[t] / denom) ** (1.0 / theta_c)
        p[t] = max(p_t, 1e-10)
    return p


def aggregate_investment(alpha_i, p, theta_i):
    return alpha_i * (p ** theta_i)


def individual_path(T, k, r, p, i_rule, a0=0.0, i_hist=None):
    """
    Simulate the individual's investment i_t and assets a_t.

    Inputs
    ------
    T, k, r: horizon, investment lag, gross interest rate net of 1 (i.e., assets grow as a*(1+r))
    p: price path (length T)
    i_rule: function(t, p_t) -> i_t (individual policy rule)
    a0: initial assets a_0
    i_hist: array-like length k of inherited investments for periods t=-k,...,-1

    Returns
    -------
    i: (T,) individual investment decisions
    a: (T+1,) assets, with a[0] = a0
    """
    if i_hist is None:
        i_hist = np.zeros(k)
    assert len(i_hist) == k, "i_hist must have length k"

    i = np.empty(T)
    a = np.empty(T + 1)
    a[0] = a0

    for t in range(T):
        # Decision at time t
        i[t] = max(i_rule(t, p[t]), 0.0)

        # Income from selling past investment decided k periods ago
        if t - k >= 0:
            delivered = i[t - k]
        else:
            delivered = i_hist[t - k]  # indexing negative into history
        cash_in = p[t] * delivered

        # Asset law of motion
        a[t + 1] = a[t] * (1.0 + r) + cash_in - i[t]

    return i, a


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Investment-Lag Model", layout="wide")
st.title("Investment-Lag Commodity Model with Individual Decisions")

with st.sidebar:
    st.header("Model Parameters")
    T = st.number_input("Horizon T", min_value=10, max_value=1000, value=200, step=10)
    k = st.number_input("Lag k (periods)", min_value=1, max_value=24, value=4, step=1)

    st.subheader("Elasticities")
    theta_c = st.number_input("Demand elasticity exponent (theta_c)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    theta_i = st.number_input("Investment elasticity exponent (theta_i)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    st.subheader("AR(1) Processes for alphas")
    rho_c = st.slider("rho_c (persistence, demand)", 0.0, 0.99, 0.8, 0.01)
    rho_i = st.slider("rho_i (persistence, investment)", 0.0, 0.99, 0.8, 0.01)
    mu_c = st.number_input("mu_c (mean level for alpha_c)", 0.01, 10.0, 1.0, 0.01)
    mu_i = st.number_input("mu_i (mean level for alpha_i)", 0.01, 10.0, 1.0, 0.01)
    sigma_c = st.number_input("sigma_c (std of shocks to alpha_c)", 0.0, 2.0, 0.05, 0.01)
    sigma_i = st.number_input("sigma_i (std of shocks to alpha_i)", 0.0, 2.0, 0.05, 0.01)

    st.subheader("Positive Shock Initialization")
    bump_periods = st.slider("# of early positive shock periods", 0, 50, 10, 1)
    bump_size = st.number_input("bump size in eps (std dev units)", 0.0, 5.0, 1.5, 0.1)
    seed = st.number_input("Random seed", 0, 10_000, 1234, 1)

    st.subheader("Initial Conditions")
    a0 = st.number_input("Initial assets a_0", -1e6, 1e6, 0.0, 100.0, format="%.2f")
    p_init_val = st.number_input("Initial pre-sample price (repeated k times)", 1e-6, 1e6.0, 1.0, 0.1, format="%.6f")
    i_hist_val = st.number_input("Inherited pre-sample investment (repeated k times)", 0.0, 1e6, 0.0, 10.0, format="%.2f")

    st.subheader("Individual Decision Rule")
    st.caption("Default: scale the aggregate investment rule by a factor χ.")
    rule_type = st.selectbox("Rule", ["Scaled aggregate investment", "Price power rule", "Constant"], index=0)
    chi = st.number_input("χ (scale vs aggregate I)", 0.0, 5.0, 1.0, 0.1)
    gamma = st.number_input("γ (coefficient for price power rule)", 0.0, 5.0, 1.0, 0.1)
    phi = st.number_input("φ (price exponent for price power rule)", 0.0, 5.0, 1.0, 0.1)
    const_i = st.number_input("Constant i_t", 0.0, 1e6, 0.0, 10.0)

    r = st.number_input("Interest rate r (per period)", -0.99, 10.0, 0.01, 0.01)

# Generate AR(1) paths
alpha_c = ar1_path(
    T=T,
    rho=rho_c,
    mu=mu_c,
    sigma=sigma_c,
    seed=int(seed) + 1,
    positive_bump_periods=bump_periods,
    bump_size=bump_size,
)
alpha_i = ar1_path(
    T=T,
    rho=rho_i,
    mu=mu_i,
    sigma=sigma_i,
    seed=int(seed) + 2,
    positive_bump_periods=bump_periods,
    bump_size=bump_size,
)

# Equilibrium price path given the market clearing with lag k
p = equilibrium_price_path(
    T=T,
    k=k,
    alpha_c=alpha_c,
    alpha_i=alpha_i,
    theta_c=theta_c,
    theta_i=theta_i,
    p_init=np.full(k, p_init_val, dtype=float),
)

# Aggregate investment (for reference and for default rule)
I_agg = aggregate_investment(alpha_i, p, theta_i)

# Define the individual's decision rule
if rule_type == "Scaled aggregate investment":
    def i_rule(t, p_t, I_agg=I_agg, chi=chi):
        return chi * I_agg[t]
elif rule_type == "Price power rule":
    def i_rule(t, p_t, gamma=gamma, phi=phi):
        return gamma * (p_t ** phi)
else:  # Constant
    def i_rule(t, p_t, const_i=const_i):
        return const_i

# Simulate individual's path
inherited_i_hist = np.full(k, i_hist_val, dtype=float)
i_ind, a_path = individual_path(
    T=T,
    k=k,
    r=r,
    p=p,
    i_rule=i_rule,
    a0=a0,
    i_hist=inherited_i_hist,
)

# Assemble DataFrame for easy display / download
idx = np.arange(T)
df = pd.DataFrame(
    {
        "t": idx,
        "alpha_c": alpha_c,
        "alpha_i": alpha_i,
        "p": p,
        "I_agg": I_agg,
        "i_ind": i_ind,
        "a": a_path[1:],  # align asset at end of period t
    }
)

# -----------------------------
# Layout & Plots
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Equilibrium Price p_t")
    fig1, ax1 = plt.subplots()
    ax1.plot(df["t"], df["p"], linewidth=2)
    ax1.set_xlabel("t")
    ax1.set_ylabel("p_t")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True)

with col2:
    st.subheader("Investment Decisions")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["t"], df["I_agg"], linestyle='--', label="Aggregate I_t", linewidth=1.5)
    ax2.plot(df["t"], df["i_ind"], label="Individual i_t", linewidth=2)
    ax2.set_xlabel("t")
    ax2.set_ylabel("Investment")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

with col3:
    st.subheader("Assets a_t")
    fig3, ax3 = plt.subplots()
    ax3.plot(df["t"], df["a"], linewidth=2)
    ax3.set_xlabel("t")
    ax3.set_ylabel("a_t")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, clear_figure=True)

# -----------------------------
# Display table and downloads
# -----------------------------

with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("Download CSV", csv, file_name="investment_lag_simulation.csv", mime="text/csv")

st.caption(
    """
    Notes:
    - Market clearing enforces α_{c,t} p_t^{-θ_c} = α_{i,t-k} p_{t-k}^{θ_i}. Prices for t<k are initialized from the sidebar.
    - The individual is atomistic: i_t does not affect p_t. Assets evolve as a_{t+1} = a_t(1+r) + p_t i_{t-k} - i_t.
    - Use the sidebar to change k, elasticities, AR(1) parameters, and the individual policy.
    """
)
