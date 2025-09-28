requirements.txt
streamlit
numpy
quantecon
matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Running a Central Bank")

# --- Parameters ---
beta   = 0.99
sigma  = 1.0
kappa  = 0.1
phi_pi = 1.5
phi_x  = 0.5
lam    = 0.5          # welfare weight on output gap

st.sidebar.header("Shock settings")
seed   = st.sidebar.number_input("Random seed", value=0, step=1)
rho_r  = st.sidebar.slider("Persistence of demand shock",0.0,0.99,0.8)
rho_u  = st.sidebar.slider("Persistence of cost shock",  0.0,0.99,0.5)
sd_r   = st.sidebar.slider("Std dev demand shock",0.0,1.0,0.5)
sd_u   = st.sidebar.slider("Std dev cost shock",  0.0,1.0,0.5)

np.random.seed(seed)

T = 12
r_shock = np.zeros(T)
u_shock = np.zeros(T)
for t in range(1,T):
    r_shock[t] = rho_r * r_shock[t-1] + sd_r*np.random.randn()
    u_shock[t] = rho_u * u_shock[t-1] + sd_u*np.random.randn()

# --- Solve once for rule-from-now-on ---
A = np.array([[1, -1/sigma],
              [0, 1]])
B = np.array([[1 + phi_x/sigma, phi_pi/sigma],
              [-kappa, 1]])
P,F,eigs = klein(A,B)  # P is 2x2, F gives shock response

# Compute expectations of x_{t+1}, pi_{t+1} under rule
# For teaching simplicity assume E_t[x_{t+1}] = 0, E_t[pi_{t+1}] = 0 when shocks mean 0
# (Or precompute using AR(1) path if desired)

# --- Game State ---
if "month" not in st.session_state:
    st.session_state.month = 0
    st.session_state.history = []

month = st.session_state.month

if month < T:
    st.subheader(f"Month {month+1}")
    policy = st.slider("Set policy interest rate i_t (%)", -5.0, 10.0, 0.0, step=0.25)

    # Expectations under future Taylor rule (simplified zero for demo)
    E_x1 = 0
    E_pi1 = 0

    rn = r_shock[month]
    u  = u_shock[month]

    # Solve current x_t, pi_t
    x_t = E_x1 - (policy - E_pi1 - rn)/sigma
    pi_t = beta*E_pi1 + kappa*x_t + u  # add cost shock

    st.write(f"Output gap x_t: {x_t:.2f}")
    st.write(f"Inflation π_t: {pi_t:.2f}")

    st.session_state.history.append((month, policy, x_t, pi_t))
    if st.button("Next month"):
        st.session_state.month += 1

else:
    st.success("Game Over!")
    hist = np.array(st.session_state.history)
    loss = np.sum(hist[:,3]**2 + lam*hist[:,2]**2)
    st.write(f"Total welfare loss: {loss:.2f}")

    fig, ax = plt.subplots()
    ax.plot(hist[:,0], hist[:,2], label="Output gap")
    ax.plot(hist[:,0], hist[:,3], label="Inflation")
    ax.legend()
    st.pyplot(fig)
