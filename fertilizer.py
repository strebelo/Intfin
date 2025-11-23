import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (edit as you like)
# -----------------------------
P0 = 100.0        # current fertilizer price in USD, P_t
S0 = 5.0          # current spot FX rate, BRL per USD, S_t
m = 0.20          # gross margin (e.g., 0.20 = 20%)
R_brl = 0.10      # BRL annual interest rate R (domestic)
R_usd = 0.05      # USD annual interest rate R* (foreign)
sigma_p = 0.30    # annual volatility of P_t
sigma_s = 0.20    # annual volatility of S_t
T = 60.0 / 360.0  # time to maturity in years (60 days)
N = 5000          # number of simulations
seed = 123        # random seed for reproducibility

# Hedge ratios we want to analyze for the mean–volatility plot
hedge_grid = np.linspace(0.0, 1.0, 11)  # 0, 0.1, ..., 1.0


# ----------------------------------
# Helper: compute forward via CIP
# ----------------------------------
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

    Parameters
    ----------
    h : float
        Hedge ratio (fraction of USD revenue hedged at the forward rate).
    P0 : float
        Initial fertilizer price in USD.
    S0 : float
        Initial spot FX rate (BRL per USD).
    m : float
        Gross margin (e.g., 0.2 = 20%).
    R_usd : float
        Annual interest rate in USD (R*).
    sigma_p : float
        Annual volatility of P_t.
    sigma_s : float
        Annual volatility of S_t.
    T : float
        Time horizon in years (e.g., 60/360).
    R_brl : float or None
        Annual interest rate in BRL (R). Used only to compute F via CIP.
        If None, uses F = S0 (no interest differential).
    N : int
        Number of Monte Carlo simulations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    profits : ndarray
        Array of simulated profits in BRL of length N.
    """
    rng = np.random.default_rng(seed)

    # Forward rate: if R_brl is provided, use CIP. Otherwise, set F = S0.
    if R_brl is not None:
        F = forward_rate(S0, R_brl, R_usd, T)
    else:
        F = S0

    # Draw shocks for P and S (independent standard normals)
    eps_p = rng.normal(0.0, 1.0, size=N)
    eps_s = r_
