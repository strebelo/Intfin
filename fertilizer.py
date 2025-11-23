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
    """
    rng = np.random.default_rng(seed)

    # Forward rate: if R_brl is provided, use CIP. Otherwise, set F = S0.
    if R_brl is not None:
        F = forward_rate(S0, R_brl, R_usd, T)
    else:
        F = S0

    # Draw shocks for P and S (independent standard normals)
    eps_p = rng.normal(0.0, 1.0, size=N)
    eps_s = rng.normal(0.0, 1.0, size=N)

    # Random-walk in levels with annual vol scaled by sqrt(T)
    P_T = P0 + sigma_p * np.sqrt(T) * eps_p
    S_T = S0 + sigma_s * np.sqrt(T) * eps_s

    # Revenue in BRL at t+60
    revenue = (1.0 + m) * P_T * (h * F + (1.0 - h) * S_T)

    # Cost in BRL at t+60 – financed in USD at R_usd
    cost = P0 * S0 * (1.0 + R_usd * T)

    profits = revenue - cost
    return profits


# =============================
# Streamlit app
# =============================

st.title("FX Hedging of Fertilizer Sales")

st.markdown(
    """
This app simulates the **profit in BRL** from buying fertilizer in USD today and
selling it in Brazil in 60 days, with an optional FX hedge.

- It compares **unhedged** and **hedged** positions.
- It plots the **mean–volatility trade-off** as a function of the hedge ratio \(h\).
"""
)

# ---- Parameters sidebar ----
st.sidebar.header("Model parameters")

P0 = st.sidebar.number_input("Initial fertilizer price P₀ (USD)", value=100.0, min_value=0.0)
S0 = st.sidebar.number_input("Spot FX S₀ (BRL per USD)", value=5.0, min_value=0.0)

m = st.sidebar.number_input("Gross margin m (e.g., 0.2 = 20%)", value=0.20, min_value=0.0, max_value=5.0)

sigma_p = st.sidebar.number_input("Annual volatility of P (σ_p)", value=0.30, min_value=0.0, max_value=5.0)
sigma_s = st.sidebar.number_input("Annual volatility of S (σ_s)", value=0.20, min_value=0.0, max_value=5.0)

R_brl = st.sidebar.number_input("BRL interest rate R (APR)", value=0.10, min_value=-1.0, max_value=5.0)
R_usd = st.sidebar.number_input("USD interest rate R* (APR)", value=0.05, min_value=-1.0, max_value=5.0)

days = st.sidebar.number_input("Horizon (days)", value=60, min_value=1, max_value=365)
T = days / 360.0

N = st.sidebar.number_input("Number of simulations", value=5000, min_value=100, max_value=200000, step=100)
seed = st.sidebar.number_input("Random seed", value=123, min_value=0, max_value=10_000_000)

h_user = st.sidebar.slider("Hedge ratio h for comparison (0 = unhedged, 1 = fully hedged)", 0.0, 1.0, 0.5, 0.05)

# Grid of hedge ratios for mean–vol plot
hedge_grid = np.linspace(0.0, 1.0, 11)  # 0, 0.1, ..., 1.0

if st.button("Run simulation"):
    # --- Simulate unhedged and hedged (at chosen h_user) ---
    profits_unhedged = simulate_profits(
        h=0.0,
        P0=P0,
        S0=S0,
        m=m,
        R_usd=R_usd,
        sigma_p=sigma_p,
        sigma_s=sigma_s,
        T=T,
        R_brl=R_brl,
        N=int(N),
        seed=int(seed),
    )

    profits_hedged = simulate_profits(
        h=h_user,
        P0=P0,
        S0=S0,
        m=m,
        R_usd=R_usd,
        sigma_p=sigma_p,
        sigma_s=sigma_s,
        T=T,
        R_brl=R_brl,
        N=int(N),
        seed=int(seed) + 1,
    )

    # --- Summary statistics function ---
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
        st.write(f"Volatility (std dev, BRL): {std_u:,.2f}")
        st.write(f"P(profit < 0): {ploss_u:.4f}")

    with col2:
        st.markdown(f"**Hedged (h = {h_user:.2f})**")
        st.write(f"Mean profit (BRL): {mean_h:,.2f}")
        st.write(f"Volatility (std dev, BRL): {std_h:,.2f}")
        st.write(f"P(profit < 0): {ploss_h:.4f}")

    # --- Histogram comparison ---
    st.subheader("Distribution of profits")

    fig1, ax1 = plt.subplots()
    ax1.hist(profits_unhedged, bins=50, alpha=0.6, density=True, label="Unhedged (h = 0)")
    ax1.hist(profits_hedged, bins=50, alpha=0.6, density=True, label=f"Hedged (h = {h_user:.2f})")
    ax1.set_xlabel("Profit in BRL at t + horizon")
    ax1.set_ylabel("Density")
    ax1.set_title("Profit distribution: unhedged vs hedged")
    ax1.legend()
    st.pyplot(fig1)

    # --- Mean vs volatility for different h ---
    st.subheader("Mean–volatility trade-off as a function of h")

    means = []
    vols = []

    for h in hedge_grid:
        profits_h = simulate_profits(
            h=h,
            P0=P0,
            S0=S0,
            m=m,
            R_usd=R_usd,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
            T=T,
            R_brl=R_brl,
            N=int(N),
            seed=int(seed),
        )
        means.append(np.mean(profits_h))
        vols.append(np.std(profits_h, ddof=1))

    means = np.array(means)
    vols = np.array(vols)

    fig2, ax2 = plt.subplots()
    ax2.plot(vols, means, marker="o")
    for h, x, y in zip(hedge_grid, vols, means):
        ax2.annotate(f"h={h:.1f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_xlabel("Volatility of profit (std dev, BRL)")
    ax2.set_ylabel("Mean profit (BRL)")
    ax2.set_title("Mean–volatility trade-off across hedge ratios h")
    st.pyplot(fig2)

else:
    st.info("Set the parameters in the sidebar and click **Run simulation**.")
