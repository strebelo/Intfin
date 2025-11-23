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

    For FX we only need S_T, so we simulate it as before.
    """
    rng = np.random.default_rng(seed)

    # --- price path for P_t ---
    n_steps = int(max(1, days))   # at least 1 step
    dt = T / n_steps

    # shocks for log-returns
    eps_p = rng.normal(0.0, 1.0, size=(N, n_steps))

    # log-price path
    logP0 = np.log(P0)
    logP_increments = sigma_p * np.sqrt(dt) * eps_p
    logP_path = logP0 + np.cumsum(logP_increments, axis=1)

    P_path = np.exp(logP_path)       # shape (N, n_steps)
    P_T = P_path[:, -1]
    P_avg = P_path.mean(axis=1)

    # --- FX terminal value ---
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

    P0 = st.sidebar.number_input("Initial fertilizer price P₀ (USD)",
                                 value=100.0, min_value=0.0)
    S0 = st.sidebar.numbe
