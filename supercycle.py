# supercycle.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------
# Helper functions
# --------------------------------

def ar1_path(T, rho, mu=1.0, sigma=0.05, seed=0, positive_bump_periods=5, bump_size=1.0):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    if positive_bump_periods > 0:
        eps[:positive_bump_periods] = bump_size

    alpha = np.empty(T)
    alpha[0] = mu
    for t in range(1, T):
        alpha[t] = (1 - rho) * mu + rho * alpha[t - 1] + sigma * eps[t]
    # first step uses eps[0]
    alpha[0] = (1 - rho) * mu + rho * mu + sigma * eps[0]
    return alpha


def equilibrium_price_path(T, k, alpha_c, alpha_i, theta_c, theta_i, p_init=None):
    if p_init is None:
        p_init = np.ones(k)
    assert len(p_init) == k, "p_init must have length k"

    p = np.empty(T)
    p[:k] = p_init
    for t in range(k, T):
        denom = alpha_i[t - k] * (p[t - k] ** theta_i)
        denom = max(denom, 1e-10)
        p_t = (alpha_c[t] / denom) ** (1.0 / theta_c)
        p[t] = max(p_t, 1e-10)
    return p


def step_assets(a_t, r, p_t, delivered, i_t):
    """a_{t+1} = a_t*(1+r) + p_t*delivered - i_t"""
    return a_t * (1.0 + r) + p_t * delivered - i_t


# --------------------------------
# Page
# --------------------------------

st.set_page_config(page_title="Investment-Lag Model (Game Mode)", layout="wide")
st.title("Investment-Lag Commodity Model")

with st.sidebar:
    # Core horizon and lag
    T = st.number_input("Horizon T", min_value=10, max_value=1000, value=200, step=10)
    k = st.number_input("Lag k (periods)", min_value=1, max_value=24, value=4, step=1)

    # Elasticities
    theta_c = st.number_input("Demand elasticity theta_c", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    theta_i = st.number_input("Investment elasticity theta_i", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # AR(1) params
    rho_c = st.slider("rho_c (demand persistence)", 0.0, 0.99, 0.8, 0.01)
    rho_i = st.slider("rho_i (investment persistence)", 0.0, 0.99, 0.8, 0.01)
    mu_c = st.number_input("mu_c (mean demand)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    mu_i = st.number_input("mu_i (mean investment)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    sigma_c = st.number_input("sigma_c (demand shocks)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
    sigma_i = st.number_input("sigma_i (investment shocks)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)

    # Positive shock seed/bump
    bump_periods = st.slider("Number of early positive shock periods", 0, 50, 10, 1)
    bump_size = st.number_input("Bump size (std dev units)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=10000, value=1234, step=1)

    # Initial conditions
    a0 = st.number_input("Initial assets a0", min_value=-1_000_000.0, max_value=1_000_000.0, value=0.0, step=100.0, format="%.2f")
    p_init_val = st.number_input("Initial pre-sample price (repeated k times)", min_value=1e-6, max_value=1e6, value=1.0, step=0.1, format="%.6f")
    i_hist_val = st.number_input("Inherited pre-sample investment (repeated k times)", min_value=0.0, max_value=1e6, value=0.0, step=10.0, format="%.2f")

    # Interest rate
    r = st.number_input("Interest rate r (per period)", min_value=-0.99, max_value=10.0, value=0.01, step=0.01)

    # Controls
    reset_clicked = st.button("Reset / Start new simulation", type="primary")


# --------------------------------
# Initialize / Reset simulation state
# --------------------------------

def init_sim():
    alpha_c = ar1_path(
        T=int(T),
        rho=rho_c, mu=mu_c, sigma=sigma_c,
        seed=int(seed) + 1,
        positive_bump_periods=int(bump_periods),
        bump_size=bump_size,
    )
    alpha_i = ar1_path(
        T=int(T),
        rho=rho_i, mu=mu_i, sigma=sigma_i,
        seed=int(seed) + 2,
        positive_bump_periods=int(bump_periods),
        bump_size=bump_size,
    )

    p = equilibrium_price_path(
        T=int(T),
        k=int(k),
        alpha_c=alpha_c,
        alpha_i=alpha_i,
        theta_c=theta_c,
        theta_i=theta_i,
        p_init=np.full(int(k), float(p_init_val), dtype=float),
    )

    st.session_state.T = int(T)
    st.session_state.k = int(k)
    st.session_state.r = float(r)

    st.session_state.alpha_c = alpha_c
    st.session_state.alpha_i = alpha_i
    st.session_state.p = p

    st.session_state.i_hist = np.full(int(k), float(i_hist_val), dtype=float)

    st.session_state.i = np.full(int(T), np.nan, dtype=float)
    st.session_state.a = np.empty(int(T) + 1, dtype=float)
    st.session_state.a[0] = float(a0)

    st.session_state.t = 0
    st.session_state.initialized = True


if reset_clicked or ("initialized" not in st.session_state):
    init_sim()


# --------------------------------
# Period-by-period interaction
# --------------------------------

T = st.session_state.T
k = st.session_state.k
r = st.session_state.r
p = st.session_state.p
i_hist = st.session_state.i_hist
i = st.session_state.i
a = st.session_state.a
t = st.session_state.t

left, right = st.columns([1, 2])

with left:
    if t < T:
        st.subheader(f"Period t = {t}")
        st.metric("Equilibrium price p_t", f"{p[t]:.4f}")
        st.metric("Assets a_t", f"{a[t]:.4f}")

        # Choose i_t
        default_i = 0.0 if np.isnan(i[t]) else float(i[t])
        i_t_input = st.number_input(
            "Choose investment i_t",
            min_value=0.0,
            max_value=1e9,
            value=default_i,
            step=10.0,
            format="%.4f",
            key=f"i_input_{t}",
        )

        # Commit decision and advance one period
        if st.button("Commit i_t and advance →", use_container_width=True):
            # Delivered units (decided k periods ago)
            if t - k >= 0 and not np.isnan(i[t - k]):
                delivered = float(i[t - k])
            else:
                delivered = float(i_hist[t - k]) if t - k < 0 else 0.0

            # Update
            i[t] = float(i_t_input)
            a[t + 1] = step_assets(a[t], r, p[t], delivered, i[t])

            # Advance time
            st.session_state.t = t + 1
            st.rerun()
    else:
        st.subheader("Simulation complete")
        st.metric("Final assets a_T", f"{a[T]:.4f}")
        st.write("You can reset to start a new run.")


# --------------------------------
# Plots (show realized history so far)
# --------------------------------

history_t = st.session_state.t  # number of completed decisions
plot_t_index = np.arange(history_t if history_t > 0 else 1)

with right:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Price")
        fig1, ax1 = plt.subplots()
        # Show the full price path for context
        ax1.plot(np.arange(T), p, linewidth=2)
        ax1.set_xlabel("t")
        ax1.set_ylabel("p_t")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, clear_figure=True)

    with col2:
        st.subheader("Investment")
        fig2, ax2 = plt.subplots()
        # Aggregate investment is not shown; only user decisions
        if history_t > 0:
            ax2.plot(plot_t_index, i[:history_t], linewidth=2, label="i_t")
            ax2.legend()
        ax2.set_xlabel("t")
        ax2.set_ylabel("Investment")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)

    with col3:
        st.subheader("Assets")
        fig3, ax3 = plt.subplots()
        # Plot assets up to current t
        ax3.plot(np.arange(history_t + 1), a[:history_t + 1], linewidth=2)
        ax3.set_xlabel("t")
        ax3.set_ylabel("a_t")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3, clear_figure=True)


# --------------------------------
# Data table + download (realized so far)
# --------------------------------

if t > 0:
    df = pd.DataFrame({
        "t": np.arange(t),
        "p_t": p[:t],
        "i_t": i[:t],
        "a_t_end": a[1:t+1],  # asset at end of period t
    })
else:
    df = pd.DataFrame({
        "t": [0],
        "p_t": [p[0]],
        "i_t": [np.nan],
        "a_t_end": [np.nan],
    })

with st.expander("Show data"):
    st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("Download data", csv, file_name="simulation.csv", mime="text/csv")
