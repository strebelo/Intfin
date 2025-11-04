# supercycle.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
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
    return a_t * (1.0 + r) + p_t * delivered - i_t

# -----------------------------
# Page / sidebar
# -----------------------------
st.set_page_config(page_title="Investment-Lag Model (Game Mode)", layout="wide")
st.title("Investment-Lag Commodity Model")

with st.sidebar:
    T = st.number_input("Horizon T", min_value=10, max_value=1000, value=200, step=10)
    k = st.number_input("Lag k (periods)", min_value=1, max_value=24, value=4, step=1)

    theta_c = st.number_input("Demand elasticity theta_c", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    theta_i = st.number_input("Investment elasticity theta_i", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    rho_c = st.slider("rho_c (demand persistence)", 0.0, 0.99, 0.8, 0.01)
    rho_i = st.slider("rho_i (investment persistence)", 0.0, 0.99, 0.8, 0.01)
    mu_c = st.number_input("mu_c (mean demand)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    mu_i = st.number_input("mu_i (mean investment)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    sigma_c = st.number_input("sigma_c (demand shocks)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
    sigma_i = st.number_input("sigma_i (investment shocks)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)

    bump_periods = st.slider("Positive shock periods", 0, 50, 10, 1)
    bump_size = st.number_input("Bump size", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=10000, value=1234, step=1)

    a0 = st.number_input("Initial assets a0", min_value=-1_000_000.0, max_value=1_000_000.0, value=0.0, step=100.0, format="%.2f")
    p_init_val = st.number_input("Initial pre-sample price (repeated k times)", min_value=1e-6, max_value=1e6.0, value=1.0, step=0.1, format="%.6f")
    i_hist_val = st.number_input("Inherited pre-sample investment (repeated k times)", min_value=0.0, max_value=1e6.0, value=0.0, step=10.0, format="%.2f")

    r = st.number_input("Interest rate r (per period)", min_value=-0.99, max_value=10.0, value=0.01, step=0.01)

    reset_clicked = st.button("Reset / Start", type="primary")

# -----------------------------
# Init / reset
# -----------------------------
def init_sim():
    alpha_c = ar1_path(int(T), rho_c, mu_c, sigma_c, int(seed) + 1, int(bump_periods), bump_size)
    alpha_i = ar1_path(int(T), rho_i, mu_i, sigma_i, int(seed) + 2, int(bump_periods), bump_size)
    p = equilibrium_price_path(int(T), int(k), alpha_c, alpha_i, theta_c, theta_i, np.full(int(k), float(p_init_val)))

    st.session_state.T = int(T)
    st.session_state.k = int(k)
    st.session_state.r = float(r)
    st.session_state.p = p
    st.session_state.i_hist = np.full(int(k), float(i_hist_val))
    st.session_state.i = np.full(int(T), np.nan)
    st.session_state.a = np.empty(int(T) + 1)
    st.session_state.a[0] = float(a0)
    st.session_state.t = 0
    st.session_state.initialized = True

if reset_clicked or ("initialized" not in st.session_state):
    init_sim()

# -----------------------------
# State
# -----------------------------
T = st.session_state.T
k = st.session_state.k
r = st.session_state.r
p = st.session_state.p
i_hist = st.session_state.i_hist
i = st.session_state.i
a = st.session_state.a
t = st.session_state.t

# -----------------------------
# Top: price chart only
# -----------------------------
fig_price, ax = plt.subplots()
ax.plot(np.arange(T), p, linewidth=2)
ax.set_xlabel("t")
ax.set_ylabel("price p_t")
ax.grid(True, alpha=0.3)
st.pyplot(fig_price, clear_figure=True)

# -----------------------------
# Delivered/output & revenue for current period
# -----------------------------
def delivered_at(j):
    if (j - k) >= 0 and not np.isnan(i[j - k]):
        return float(i[j - k])
    return float(i_hist[j - k]) if (j - k) < 0 else 0.0

# -----------------------------
# Table: price, output, revenue, assets, investment
# rows = completed periods only (0..t-1)
# -----------------------------
if t > 0:
    out_hist = [delivered_at(j) for j in range(t)]
    rev_hist = [p[j] * out_hist[j] for j in range(t)]
    data = {
        "price": p[:t],
        "output": out_hist,
        "revenue": rev_hist,
        "assets": a[1:t+1],      # end-of-period assets
        "investment": i[:t],
    }
    df = pd.DataFrame(data, index=np.arange(t))
else:
    df = pd.DataFrame({"price": [p[0]], "output": [np.nan], "revenue": [np.nan], "assets": [np.nan], "investment": [np.nan]})

st.dataframe(df, use_container_width=True)

# -----------------------------
# Decision input and advance
# -----------------------------
if t < T:
    delivered = delivered_at(t)
    default_i = 0.0 if np.isnan(i[t]) else float(i[t])
    i_t_input = st.number_input("Choose investment i_t", min_value=0.0, max_value=1e12, value=default_i, step=10.0, format="%.4f", key=f"i_input_{t}")
    if st.button("Commit and advance"):
        i[t] = float(i_t_input)
        a[t + 1] = step_assets(a[t], r, p[t], delivered, i[t])
        st.session_state.t = t + 1
        st.rerun()
else:
    st.write(f"Final assets: {a[T]:.4f}")

# -----------------------------
# Download
# -----------------------------
csv = df.to_csv(index=False).encode()
st.download_button("Download table (csv)", csv, file_name="simulation.csv", mime="text/csv")
