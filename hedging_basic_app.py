# ==========================
# basic_app.py
# ==========================
# Streamlit app — Constant h_t (basic)
# - LaTeX-rendered math explanations
# - Bid-ask spread scales per year of tenor
# - Overlayed PV-profit distributions (Hedge-all-at-0 vs Rolling 1-Year)
# - H-sweep frontier: σ(PV Profit) vs E[PV Profit] for both strategies
# ==========================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Math helpers & pricing
# ------------------------------
def make_discount_factors_constant(r: float, T: int):
    DF = np.ones(T + 1, dtype=float)
    base = 1.0 + float(r)
    if base <= 0:
        base = 1e-9
    for t in range(1, T + 1):
        DF[t] = 1.0 / (base ** t)
    return DF

def forward_dc_per_fc_constant_rate(S_t_dc_fc, r_d, r_f, t, m):
    """ CIP-style forward (mid) with constant rates rd, rf. """
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    horiz = m - t
    num = (1.0 + float(r_d)) ** horiz
    den = (1.0 + float(r_f)) ** horiz
    if den <= 0:
        den = 1e-12
    return S_t_dc_fc * (num / den)

def bid_ask_from_mid_tenor_scaled(mid, spread_bps_per_year, years):
    """ Scale spread by tenor: s_total = (bps_per_year * years)/10000. """
    s_total = max(0.0, float(spread_bps_per_year)) * float(years) / 10000.0
    bid = mid * (1.0 - s_total / 2.0)
    ask = mid * (1.0 + s_total / 2.0)
    return bid, ask

def simulate_spot_paths_dc_fc_with_infl_drift(S0_dc_fc, sigma, infl_diff, n_sims, T, seed=123):
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    if (1.0 + infl_diff) <= 0.0:
        raise ValueError("Inflation differential too negative (1 + pi_Delta must be > 0).")

    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, T+1), dtype=float)
    paths[:, 0] = float(S0_dc_fc)
    # ensure E[ growth ] = 1 + infl_diff
    mu_adj = np.log1p(infl_diff) - 0.5 * (sigma ** 2)

    for t in range(1, T+1):
        eps = rng.standard_normal(n_sims)
        growth = np.exp(mu_adj + sigma * eps)
        paths[:, t] = np.maximum(1e-12, paths[:, t-1] * growth)

    return paths

# ------------------------------
# Engines (return per-year DC revenue too)
# ------------------------------
def results_constant_h(
    S_paths, S0, DF_d,
    r_d, r_f,
    costs_dc, revenue_fc,
    spread_bps_per_year,
    h, strategy # 'all_at_t0' or 'roll_one_year'
):
    n_sims, T_plus_1 = S_paths.shape
    T = T_plus_1 - 1

    dc_rev_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t = np.zeros((n_sims, T), dtype=float)

    h = float(np.clip(h, 0.0, 1.0))

    if strategy == "all_at_t0":
        # For each delivery year t (1..T) we hedge at t=0 using tenor = t years
        for t in range(1, T+1):
            F_mid_0t = forward_dc_per_fc_constant_rate(S0, r_d, r_f, 0, t)
            bid_0t, _ = bid_ask_from_mid_tenor_scaled(F_mid_0t, spread_bps_per_year, years=t)

            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            dc_forward   = hedged_fc * bid_0t
            spot_t       = np.maximum(S_paths[:, t], 1e-12)
            dc_unhedged  = unhedged_fc * spot_t

            dc_rev_t[:, t-1] = dc_forward + dc_unhedged

    elif strategy == "roll_one_year":
        # Each year hedge next year's flow with a 1-year tenor forward (tenor=1 → spread scale=1)
        ratio = (1.0 + float(r_d)) / max(1e-12, (1.0 + float(r_f)))
        for t in range(1, T+1):
            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            S_prev = S_paths[:, t-1]
            F_mid_prev_t = S_prev * ratio
            bid_prev_t, _ = bid_ask_from_mid_tenor_scaled(F_mid_prev_t, spread_bps_per_year, years=1.0)

            dc_forward   = hedged_fc * bid_prev_t
            spot_t       = np.maximum(S_paths[:, t], 1e-12)
            dc_unhedged  = unhedged_fc * spot_t

            dc_rev_t[:, t-1] = dc_forward + dc_unhedged
    else:
        raise ValueError("Unknown strategy.")

    for t in range(1, T+1):
        dc_costs_t[:, t-1] = costs_dc[t-1]

    pv_rev = np.sum(dc_rev_t * DF_d[1:][None, :], axis=1)
    pv_cost = np.sum(dc_costs_t * DF_d[1:][None, :], axis=1)
    pv_profit = pv_rev - pv_cost

    def _nanstd(x):
        x = x[np.isfinite(x)]
        if x.size <= 1: return 0.0
        return float(np.std(x, ddof=1))
    def _fracneg(x):
        x = x[np.isfinite(x)]
        if x.size == 0: return float("nan")
        return float(np.mean(x < 0.0))

    out = {
        "pv_revenue_per_sim": np.where(np.isfinite(pv_rev), pv_rev, np.nan),
        "pv_cost_per_sim": np.where(np.isfinite(pv_cost), pv_cost, np.nan),
        "pv_profit_per_sim": np.where(np.isfinite(pv_profit), pv_profit, np.nan),
        "avg_pv_revenue": float(np.nanmean(pv_rev)) if np.isfinite(pv_rev).any() else float("nan"),
        "avg_pv_cost": float(np.nanmean(pv_cost)) if np.isfinite(pv_cost).any() else float("nan"),
        "avg_pv_profit": float(np.nanmean(pv_profit)) if np.isfinite(pv_profit).any() else float("nan"),
        "std_pv_profit": _nanstd(pv_profit),
        "frac_neg_profit": _fracneg(pv_profit),
        # kept for potential future use; no plots rely on this now
        "mean_dc_rev_by_year": np.nanmean(dc_rev_t, axis=0),  # shape (T,)
    }
    return out

# ------------------------------
# Streamlit UI (basic)
# ------------------------------
st.set_page_config(page_title="Currency Hedging — Basic (constant h_t)", layout="wide")
st.title("💱 Currency Risk Hedging Laboratory")

with st.expander("Show math / notation"):
    st.latex(r"F_{t,m} = S_t \cdot \frac{(1+r_d)^{m-t}}{(1+r_f)^{m-t}}")
    st.latex(r"\mu = \ln(1+\pi_{\Delta}) - \tfrac{1}{2}\sigma^2,\quad \mathbb{E}\!\left[\frac{S_t}{S_{t-1}}\right]=1+\pi_{\Delta}")
    st.latex(r"\text
