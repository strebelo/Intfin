import numpy as np

import matplotlib
# Use a non-interactive backend so nothing hangs on GUI backends
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st


# -----------------------------
# Helper: compute forward via CIP
# -----------------------------
def forward_rate(S, R_dom, R_for, T):
    """
    Covered interest parity with simple interest:
    F = S * (1 + R_dom * T) / (1 + R_for * T)
    where R_dom is the BRL rate and R_for is the USD rate.
    """
    return S * (1.0 + R_dom * T) / (1.0 + R_for * T)


# ----------------------------------
# Simulate terminal price and FX once
# ----------------------------------
def simulate_paths_terminal(P0, S0, sigma_p, sigma_s, T, N, seed):
    """
    Simulate P_{t+T} and S_{t+T} under lognormal dynamics:

        log(P_T / P0) ~ N(0, sigma_p^2 * T)
        log(S_T / S0) ~ N(0, sigma_s^2 * T)
    """
    rng = np.random.default_rng(seed)
    eps_p = rng.normal(0.0, 1.0, size=N)
    eps_s = rng.normal(0.0, 1.0, size=N)

    P_T = P0 * np.exp(sigma_p * np.sqrt(T) * eps_p)
    S_T = S0 * np.exp(sigma_s * np.sqrt(T) * eps_s)

    return P_T, S_T


# ----------------------------------
# Simulate full price path to get average
# ----------------------------------
def simulate_paths_with_avg(P0, S0, sigma_p, sigma_s, T, days, N, seed):
    """
    Simulate daily GBM path for P_t over 'days' steps to compute both:
      - terminal price P_T
      - average price over the period

    For FX we only need S_T, so we simulate it as a single-step GBM.
    """
    rng = np.random.default_rng(seed)

    # --- price path for P_t ---
    n_steps = int(max(1, days))   # at least 1 step
    dt = T / n_steps

    # shocks for log-returns of P
    eps_p = rng.normal(0.0, 1.0, size=(N, n_steps))

    # log-price path
    logP0 = np.log(P0)
    logP_increments = sigma_p * np.sqrt(dt) * eps_p
    logP_path = logP0 + np.cumsum(logP_increments, axis=1)

    P_path = np.exp(logP_path)       # shape (N, n_steps)
    P_T = P_path[:, -1]
    P_avg = P_path.mean(axis=1)

    # --- FX terminal value (single step GBM) ---
    eps_s = rng.normal(0.0, 1.0, size=N)
    S_T = S0 * np.exp(sigma_s * np.sqrt(T) * eps_s)

    return P_T, P_avg, S_T


def main():
    st.set_page_config(page_title="Fertilizer FX Hedging", layout="centered")
    st.title("FX Hedging of Fertilizer Sales")

    st.write("✅ App loaded. Adjust parameters in the sidebar and click **Run simulation**.")

    # -------------------------
    # Sidebar: Parameters
    # -------------------------
    st.sidebar.header("Model parameters")

    P0 = st.sidebar.number_input(
        "Initial fertilizer price P₀ (USD)",
        value=100.0,
        min_value=0.0,
    )
    S0 = st.sidebar.number_input(
        "Spot FX S₀ (BRL per USD)",
        value=5.0,
        min_value=0.0,
    )

    m = st.sidebar.number_input(
        "Gross margin m (e.g., 0.10 = 10%)",
        value=0.12,
        min_value=0.0,
        max_value=5.0,
    )

    sigma_p = st.sidebar.number_input(
        "Annual volatility of log P (σₚ)",
        value=0.30,
        min_value=0.0,
        max_value=5.0,
    )
    sigma_s = st.sidebar.number_input(
        "Annual volatility of log S (σₛ)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
    )

    R_brl = st.sidebar.number_input(
        "BRL interest rate R",
        value=0.15,
        min_value=-1.0,
        max_value=5.0,
    )
    R_usd = st.sidebar.number_input(
        "USD interest rate R*",
        value=0.05,
        min_value=-1.0,
        max_value=5.0,
    )

    days = st.sidebar.number_input(
        "Horizon (days)",
        value=60,
        min_value=1,
        max_value=365,
    )
    T = days / 360.0

    N = st.sidebar.number_input(
        "Number of simulations",
        value=5000,
        min_value=100,
        max_value=200000,
        step=100,
    )
    seed = st.sidebar.number_input(
        "Random seed",
        value=123,
        min_value=0,
        max_value=10_000_000,
    )

    h_user = st.sidebar.slider(
        "Hedge ratio h",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    # *** NEW: choose how the sale price is indexed ***
    price_mode = st.sidebar.radio(
        "Sale price indexation",
        options=["Terminal price P(T)", "Average price over [t, t+T]"],
        index=0,
    )

    hedge_grid = np.linspace(0.0, 1.0, 11)

    # -------------------------
    # Run simulation
    # -------------------------
    if st.button("Run simulation"):
        st.write("▶️ Running Monte Carlo simulation...")

        N_int = int(N)
        seed_int = int(seed)

        # Forward rate via CIP
        F = forward_rate(S0, R_brl, R_usd, T)

        # --- Simulate according to chosen price mode ---
        if price_mode == "Terminal price P(T)":
            P_T, S_T = simulate_paths_terminal(
                P0=P0,
                S0=S0,
                sigma_p=sigma_p,
                sigma_s=sigma_s,
                T=T,
                N=N_int,
                seed=seed_int,
            )
            P_sale = P_T  # sale price uses terminal P_T
        else:
            P_T, P_avg, S_T = simulate_paths_with_avg(
                P0=P0,
                S0=S0,
                sigma_p=sigma_p,
                sigma_s=sigma_s,
                T=T,
                days=days,
                N=N_int,
                seed=seed_int,
            )
            P_sale = P_avg  # sale price uses average price

        # Cost in BRL (same across paths and hedge ratios), financed at BRL rate
        cost = P0 * S0 * (1.0 + R_brl * T)

        # Unhedged profits: revenue uses P_sale and S_T
        profits_unhedged = (1.0 + m) * P_sale * S_T - cost

        # User-selected hedge profits
        profits_hedged = (1.0 + m) * P_sale * (h_user * F + (1.0 - h_user) * S_T) - cost

        def summarize(profits):
            mean = np.mean(profits)
            std = np.std(profits, ddof=1)
            prob_loss = np.mean(profits < 0.0)
            return mean, std, prob_loss

        mean_u, std_u, ploss_u = summarize(profits_unhedged)
        mean_h, std_h, ploss_h = summarize(profits_hedged)

        # -------------------------
        # Summary statistics
        # -------------------------
        st.subheader("Summary statistics")

        pricing_label = (
            "terminal price P(T)"
            if price_mode == "Terminal price P(T)"
            else "average price over the period"
        )
        st.markdown(f"*Sale price indexed to **{pricing_label}***")

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

        # -------------------------
        # Histogram of profits
        # -------------------------
        st.subheader("Distribution of profits")

        fig1, ax1 = plt.subplots()
        ax1.hist(
            profits_unhedged,
            bins=50,
            alpha=0.6,
            density=True,
            label="Unhedged",
        )
        ax1.hist(
            profits_hedged,
            bins=50,
            alpha=0.6,
            density=True,
            label=f"Hedged (h={h_user:.2f})",
        )
        ax1.set_xlabel("Profit (BRL)")
        ax1.set_ylabel("Density")
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

        # -------------------------
        # Mean–volatility trade-off across hedge ratios
        # -------------------------
        st.subheader("Mean–volatility trade-off across hedge ratios")

        H = hedge_grid[:, None]  # shape (11, 1)
        revenue_grid = (1.0 + m) * P_sale[None, :] * (
            H * F + (1.0 - H) * S_T[None, :]
        )
        profits_grid = revenue_grid - cost

        means = profits_grid.mean(axis=1)
        vols = profits_grid.std(axis=1, ddof=1)

        fig2, ax2 = plt.subplots()
        ax2.plot(vols, means, marker="o")

        for h, x, y in zip(hedge_grid, vols, means):
            ax2.annotate(
                f"h={h:.1f}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax2.set_xlabel("Profit volatility (std dev, BRL)")
        ax2.set_ylabel("Mean profit (BRL)")
        st.pyplot(fig2)
        plt.close(fig2)

        st.write("✅ Simulation finished.")

    # -------------------------
    # Disclaimer
    # -------------------------
    st.markdown("---")
    st.markdown("### This app is for educational purposes only.")


if __name__ == "__main__":
    main()
