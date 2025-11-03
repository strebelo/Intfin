# supercycle.py
import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import deque
from io import StringIO

st.set_page_config(page_title="Cattle Cycle Game", page_icon="🐄", layout="wide")

# ---------- Utility helpers ----------
def init_game_state(k, r, epsilon, use_shocks, rho, sigma, seed, a0):
    """
    Initialize session state variables and queues.
    We'll start near steady state: Q*=1 → need (1+r)*I* = 1 ⇒ I* = 1/(1+r)
    Pre-fill k pending deliveries so early prices are well-defined.
    """
    st.session_state.t = 0
    st.session_state.k = k
    st.session_state.r = r
    st.session_state.epsilon = epsilon

    # Shock process A_t
    st.session_state.use_shocks = use_shocks
    st.session_state.rho = rho
    st.session_state.sigma = sigma
    st.session_state.rng = np.random.default_rng(seed if seed is not None else 0)
    st.session_state.A_hist = []

    # Initialize A_0
    if use_shocks:
        st.session_state.a_t = a0  # log A
    else:
        st.session_state.a_t = 0.0  # log(1)=0

    # Investment queues (for deliveries after k periods)
    st.session_state.queue_h = deque([0.0]*k, maxlen=k)
    st.session_state.queue_b = deque([0.0]*k, maxlen=k)

    # Pre-fill with steady-state deliveries so first k periods have Q≈1
    I_star = 1.0 / (1.0 + r)
    for _ in range(k):
        st.session_state.queue_h.appendleft(0.5 * I_star)  # split between human/bot for symmetry
        st.session_state.queue_b.appendleft(0.5 * I_star)

    # Histories
    st.session_state.price_hist = []   # p_t
    st.session_state.Q_hist = []       # Q_t
    st.session_state.Ih_hist = []      # human I_t (decision at time t)
    st.session_state.Ib_hist = []      # bot I_t (decision at time t)
    st.session_state.pay_h_hist = []   # realized profits (per t) for human
    st.session_state.pay_b_hist = []   # realized profits (per t) for bot

    # Maturing investment logs (to compute payoffs when they deliver)
    st.session_state.maturing_h = deque([0.5 * I_star]*(k), maxlen=k)
    st.session_state.maturing_b = deque([0.5 * I_star]*(k), maxlen=k)

    # Current price p_t will be computed in first advance_step() call
    st.session_state.p_t = None
    st.session_state.Q_t = None

def compute_price(Q_t, A_t, epsilon):
    # Guard against zero output
    Q_safe = max(Q_t, 1e-8)
    return A_t * (Q_safe ** (-1.0 / epsilon))

def advance_shock(use_shocks, rho, sigma, rng, a_t):
    """Return next log-shock a_{t+1} and level A_{t+1}."""
    if not use_shocks:
        return 0.0, 1.0
    a_next = rho * a_t + sigma * rng.normal()
    return a_next, math.exp(a_next)

def bot_investment(p_t, theta, gamma, I_cap):
    """Simple 'buy when high' rule."""
    I = gamma * max(0.0, p_t - theta)
    if I_cap is not None:
        I = min(I, I_cap)
    return max(I, 0.0)

def realize_profits(p_price, r, I_matured_h, I_matured_b, unit_cost):
    rev_h = p_price * (1.0 + r) * I_matured_h
    rev_b = p_price * (1.0 + r) * I_matured_b
    pay_h = rev_h - unit_cost * I_matured_h
    pay_b = rev_b - unit_cost * I_matured_b
    return pay_h, pay_b

def build_dataframe():
    T = len(st.session_state.price_hist)
    df = pd.DataFrame({
        "t": np.arange(T),
        "A_t": st.session_state.A_hist[:T],
        "Q_t": st.session_state.Q_hist[:T],
        "p_t": st.session_state.price_hist[:T],
        "I_h(t)": st.session_state.Ih_hist[:T],
        "I_b(t)": st.session_state.Ib_hist[:T],
        "π_h(t)": st.session_state.pay_h_hist[:T],
        "π_b(t)": st.session_state.pay_b_hist[:T],
    })
    return df

# ---------- Page sidebar (parameters) ----------
with st.sidebar:
    st.header("⚙️ Game Parameters")

    col_a, col_b = st.columns(2)
    with col_a:
        k = st.number_input("Gestation lag k", min_value=1, max_value=20, value=3, step=1)
        epsilon = st.number_input("Demand elasticity ε", min_value=0.2, max_value=10.0, value=2.0, step=0.1, format="%.1f")
        r = st.number_input("Return r", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
    with col_b:
        unit_cost = st.number_input("Unit cost c", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=123, step=1)
        rounds_to_autoplay = st.number_input("Auto advance rounds", min_value=0, max_value=100, value=0, step=1,
                                             help="Advance this many rounds when you click 'Submit & Advance'.")

    st.markdown("---")
    st.subheader("Shocks Aₜ")
    use_shocks = st.checkbox("Use AR(1) shocks to demand level (A_t)", value=False)
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        rho = st.number_input("ρ (AR coef)", min_value=0.0, max_value=0.99, value=0.60, step=0.05)
    with col_s2:
        sigma = st.number_input("σ (shock s.d.)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f")
    with col_s3:
        a0 = st.number_input("log A₀", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)

    st.markdown("---")
    st.subheader("Bot (Invests When Price Is High)")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        theta = st.number_input("Threshold θ", min_value=0.0, max_value=5.0, value=1.00, step=0.05)
    with col_b2:
        gamma = st.number_input("Aggressiveness γ", min_value=0.0, max_value=10.0, value=0.80, step=0.05,
                                help="Slope of bot's response: I_b = γ·max(0, p_t-θ).")
    with col_b3:
        I_cap = st.number_input("Bot I cap (0=none)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        I_cap = None if I_cap == 0.0 else I_cap

    st.markdown("---")
    if st.button("🔄 Reset Game", use_container_width=True):
        init_game_state(k, r, epsilon, use_shocks, rho, sigma, seed, a0)
        st.experimental_rerun()

# ---------- First-load initialization ----------
if "t" not in st.session_state:
    init_game_state(k, r, epsilon, use_shocks, rho, sigma, seed, a0)

# If sidebar params changed after in
