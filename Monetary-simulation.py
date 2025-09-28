# Monetary-simulation.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Central Bank Game")

# --- Parameters (constant for the demo) ---
beta   = 0.99
sigma  = 1.0
kappa  = 0.1
phi_pi = 1.5
phi_x  = 0.5
lam    = 0.5  # welfare weight on output gap

st.sidebar.header("Shock settings")
seed   = st.sidebar.number_input("Random seed", value=0, step=1)
rho_r  = st.sidebar.slider("Persistence of demand shock", 0.0, 0.99, 0.8)
rho_u  = st.sidebar.slider("Persistence of cost shock",   0.0, 0.99, 0.5)
sd_r   = st.sidebar.slider("Std dev demand shock", 0.0, 1.0, 0.5)
sd_u   = st.sidebar.slider("Std dev cost shock",   0.0, 1.0, 0.5)

np.random.seed(seed)

T = 12

# Persistent AR(1) shocks
r_shock = np.zeros(T)  # demand / natural-rate shock (think r^n_t)
u_shock = np.zeros(T)  # cost-push shock
for t in range(1, T):
    r_shock[t] = rho_r * r_shock[t-1] + sd_r * np.random.randn()
    u_shock[t] = rho_u * u_shock[t-1] + sd_u * np.random.randn()

# --- Session state / game history ---
if "month" not in st.session_state:
    st.session_state.month = 0
    st.session_state.i_path = []   # chosen policy rates
    st.session_state.x_path = []   # output gap
    st.session_state.pi_path = []  # inflation

month = st.session_state.month

st.write("Students choose a monthly policy rate. "
         "For pedagogy, we take $E_t[x_{t+1}] = E_t[\\pi_{t+1}] = 0$.")

if month < T:
    st.subheader(f"Month {month+1} of {T}")
    policy = st.slider("Set policy interest rate i_t (%)",
                       -5.0, 10.0, 0.0, step=0.25)

    # Simplified NK IS and Phillips with zero expectations:
    # x_t ≈ -(i_t - r^n_t)/sigma
    # π_t ≈ κ x_t + u_t
    x_t  = -(policy - r_shock[month]) / sigma
    pi_t = kappa * x_t + u_shock[month]

    col1, col2, col3 = st.columns(3)
    col1.metric("Chosen i_t (%)", f"{policy:.2f}")
    col2.metric("Output gap x_t", f"{x_t:.3f}")
    col3.metric("Inflation π_t", f"{pi_t:.3f}")

    if st.button("Confirm this month’s decision"):
        st.session_state.i_path.append(policy)
        st.session_state.x_path.append(x_t)
        st.session_state.pi_path.append(pi_t)
        st.session_state.month += 1
        st.experimental_rerun()

else:
    st.subheader("Game complete")
    i = np.array(st.session_state.i_path)
    x = np.array(st.session_state.x_path)
    pi = np.array(st.session_state.pi_path)

    # Plot results
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, T+1), i, marker="o")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Policy rate i_t (%)")
    ax1.set_title("Policy path")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, T+1), x, marker="o")
    ax2.set_xlabel("Month"); ax2.set_ylabel("x_t")
    ax2.set_title("Output gap")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, T+1), pi, marker="o")
    ax3.set_xlabel("Month"); ax3.set_ylabel("π_t")
    ax3.set_title("Inflation")
    st.pyplot(fig3)

    if st.button("Reset"):
        for k in ["month", "i_path", "x_path", "pi_path"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()
