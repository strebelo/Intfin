import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Helper: compute forward via CIP
# -----------------------------
def forward_rate(S, R_dom, R_for, T):
    """
    Covered interest parity with simple interest:
    F = S * (1 + R_dom * T) / (1 + R_for * T)
    """
    return S * (1.0 + R_dom * T) / (1.0 + R_for * T)


# ----------------------------------
# Simulation of profit for a given h
# ----------------------------------
def simulate_profits(
    h,
    P0,
    S0,
    m,
    R_usd,
    sigma_p,
    sigma_s,
    T,
    R_brl=None,
    N=5000,
    seed=None
):
    """
    Simulate profits in BRL at t+60 for a hedge ratio h in [0,1].
    Prices follow lognormal diffusion:

        log(P_T / P0) ~ N(0, sigma_p^2 * T)
        log(S_T / S0) ~ N(0, sigma_s^2 * T)

    where sigma_p and sigma_s are annual volatilities of log-returns.
    """
    rng = np.random.default_rng(seed)

    # Forward rate
    if R_brl is not None:
        F = forward_rate(S0, R_brl, R_usd, T)
    else:
        F = S0

    # Draw shocks for log-returns
    eps_p = rng.normal(0.0, 1.0, size=N)
    eps_s = rng.normal(0.0, 1.0, size=N)

    # Geometric Brownian motion (GBM)
    P_T = P0 * np.exp(sigma_p * np.sqrt(T) * eps_p)
    S_T = S0 * np.exp(sigma_s * np.sqrt(T) * eps_s)

    # Revenue in BRL at t+60
    revenue = (1.0 + m) * P_T * (h * F + (1.0 - h) * S_T)

    # Cost in BRL at t+60 – financed in USD
    cost = P0 * S0 * (1.0 + R_usd * T)

    profits = revenue - cost
    return profits


# =============================
# Streamlit UI
# =============================

st.title("FX Hedging of Fertilizer Sales")

# -------------------------
# Explanation of the formulas
# -------------------------
st.markdown(r"""
## **Model Overview**

A firm buys fertilizer in USD at time \(t\) and sells it in Brazil 60 days later.
Let:

- \(P_t\): fertilizer price in USD  
- \(S_t\): BRL/USD exchange rate  
- \(m\): gross margin on fertilizer  
- \(R^{BRL}\), \(R^{USD}\): BRL and USD interest rates  

The firm earns revenue in BRL at \(t+60\):

\[
\text{Revenue} = (1+m)\,P_{t+60}\,S_{t+60}.
\]

If it hedges a fraction \(h\) using a forward contract at rate \(F_{t,t+60}\):

\[
\text{Revenue} = (1+m)P_{t+60}\big( hF_{t,t+60} + (1-h)S_{t+60} \big).
\]

The USD cost of purchasing the fertilizer grows at the USD interest rate:

\[
\text{Cost} = P_t S_t \big( 1 + R^{USD}T \big).
\]

The profit is:

\[
\Pi = (1+m)P_{t+60}\big( hF + (1-h)S_{t+60} \big)
      - P_t S_t (1 + R^{USD}T).
\]

### **Price and FX Dynamics**

We assume **lognormal** evolution for both the fertilizer price and the exchange rate:

\[
\log\left(\frac{P_{t+60}}{P_t}\right)
   \sim N\big(0,\sigma_P^2 T\big), \qquad
\log\left(\frac{S_{t+60}}{S_t}\right)
   \sim N\big(0,\sigma_S^2 T\big).
\]

Thus:

\[
P_{t+60} = P_t \exp(\sigma_P\sqrt{T}Z_P), \quad
S_{t+60} = S_t \exp(\sigma_S\sqrt{T}Z_S),
\]

with \(Z_P, Z_S \sim N(0,1)\) independent.
""")

# -------------------------
# Sidebar: Parameters
# -------------------------

st.sidebar.header("Model parameters")

P0 = st.sidebar.number_input("Initial fertilizer price P₀ (USD)", value=100.0, min_value=0.0)
S0 = st.sidebar.number_input("Spot FX S₀ (BRL per USD)", value=5.0, min_value=0.0)

m = st.sidebar.number_input("Gross margin m (e.g., 0.10 = 10%)", value=0.12, min_value=0.0, max_value=5.0)

sigma_p = st.sidebar.number_input("Annual volatility of log P (σₚ)", value=0.30, min_value=0.0, max_value=5.0)
sigma_s = st.sidebar.number_input("Annual volatility of log S (σₛ)", value=0.10, min_value=0.0, max_value=5.0)

R_brl = st.sidebar.number_input("BRL interest rate R", value=0.10, min_value=-1.0, max_value=5.0)
R_usd = st.sidebar.number_input("USD interest rate R*", value=0.05, min_value=-1.0, max_value=5.0)

days = st.sidebar.number_input("Horizon (days)", value=60, min_value=1, max_value=365)
T = days / 360.0

N = st.sidebar.number_input("Number of simulations", value=5000, min_value=100, max_value=200000, step=100)
seed = st.sidebar.number_input("Random seed", value=123, min_value=0, max_value=10_000_000)

h_user = st.sidebar.slider("Hedge ratio h", 0.0, 1.0, 0.5, 0.05)

hedge_grid = np.linspace(0.0, 1.0, 11)

# -------------------------
# Run simulation
# -------------------------

if st.button("Run simulation"):
    profits_unhedged = simulate_profits(
        h=0.0, P0=P0, S0=S0, m=m,
        R_usd=R_usd, sigma_p=sigma_p, sigma_s=sigma_s,
        T=T, R_brl=R_brl, N=int(N), seed=int(seed)
    )

    profits_hedged = simulate_profits(
        h=h_user, P0=P0, S0=S0, m=m,
        R_usd=R_usd, sigma_p=sigma_p, sigma_s=sigma_s,
        T=T, R_brl=R_brl, N=int(N), seed=int(seed)+1
    )

    def summarize(profits):
        mean = np.mean(profits)
        std = np.std(profits, ddof=1)
        prob_loss = np.mean(profits < 0.0)
        return mean, std, prob_loss

    mean_u, std_u, ploss_u = summarize(profits_unhedged)
    mean_h, std_h, ploss_h = summarize(profits_hedged)

    st.subheader("Summary statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Unhedged (h = 0)**")
        st.write(f"Mean profit (BRL): {mean_u:,.2f}")
        st.write(f"Volatility (std dev): {std_u:,.2f}")
        st.write(f"P(profit < 0): {ploss_u:.4f}")

    with col2:
        st.markdown(f"**Hedged (h = {h_user:.2f})**")
        st.write(f"Mean profit (BRL): {mean_h:,.2f}")
        st.write(f"Volatility (std dev): {std_h:,.2f}")
        st.write(f"P(profit < 0): {ploss_h:.4f}")

    # Histogram
    st.subheader("Distribution of profits")

    fig1, ax1 = plt.subplots()
    ax1.hist(profits_unhedged, bins=50, alpha=0.6, density=True, label="Unhedged")
    ax1.hist(profits_hedged, bins=50, alpha=0.6, density=True, label=f"Hedged (h={h_user:.2f})")
    ax1.set_xlabel("Profit (BRL)")
    ax1.set_ylabel("Density")
    ax1.legend()
    st.pyplot(fig1)

    # Mean–Volatility trade-off
    st.subheader("Mean–volatility trade-off across hedge ratios")

    means, vols = [], []

    for h in hedge_grid:
        prof = simulate_profits(
            h=h, P0=P0, S0=S0, m=m,
            R_usd=R_usd, sigma_p=sigma_p, sigma_s=sigma_s,
            T=T, R_brl=R_brl, N=int(N), seed=int(seed)
        )
        means.append(np.mean(prof))
        vols.append(np.std(prof, ddof=1))

    fig2, ax2 = plt.subplots()
    ax2.plot(vols, means, marker="o")

    for h, x, y in zip(hedge_grid, vols, means):
        ax2.annotate(f"h={h:.1f}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax2.set_xlabel("Profit volatility (std dev)")
    ax2.set_ylabel("Mean profit (BRL)")
    st.pyplot(fig2)

# -------------------------
# Disclaimer
# -------------------------
st.markdown("---")
st.markdown("### **This app is for educational purposes only.**")
