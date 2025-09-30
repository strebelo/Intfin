# ------------------------------
# Currency Risk Hedging Simulator (Streamlit) — Constant Hedge Fraction
# with Unhedged Baseline (h = 0)
# ------------------------------
# What changed vs. your previous version?
# - Adds an explicit "Unhedged (h=0)" baseline alongside:
#     (A) Hedge-all-at-0
#     (B) Roll 1-Year
# - Summary table and histograms include the unhedged distribution.
#
# Modeling assumptions:
# - Students choose a SINGLE hedge fraction h (0–100%) that applies to every year.
# - No term structure: constant domestic and foreign annual rates r_d, r_f.
# - Forwards via covered interest parity with constant rates:
#     F_{t->m} (FC/DC) = S_t * [ (1 + r_d)^(m - t) / (1 + r_f)^(m - t) ]
# - FC/DC notation: spot/forward are "foreign currency per 1 unit of domestic".
#   To convert foreign amount (FC) to domestic (DC) at a rate X (FC/DC): DC = FC / X.
#
# Discount factors with constant r:
#   DF_r[0] = 1
#   DF_r[t] = 1 / (1 + r)^t, for t = 1..T
# ------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Helper functions
# ------------------------------

def make_discount_factors_constant(r: float, T: int = 10):
    """
    Build discount factors DF[0..T] from a single annual rate r (constant across maturities).
    Annual compounding:
        DF[0] = 1
        DF[t] = 1 / (1 + r)^t
    Notes:
        - If 1 + r <= 0, discounting breaks. We guard and set a tiny positive denom.
    """
    DF = np.ones(T + 1, dtype=float)
    denom_base = 1.0 + float(r)
    if denom_base <= 0:
        denom_base = 1e-9  # guard
    for t in range(1, T + 1):
        DF[t] = 1.0 / (denom_base ** t)
    return DF

def forward_fc_per_dc_constant_rate(S_t, r_d, r_f, t, m):
    """
    Synthetic forward (FC/DC) from t to m using covered interest parity with constant rates.
        F_{t->m} = S_t * ((1 + r_d)^(m - t) / (1 + r_f)^(m - t))
    Requires m > t.
    Accepts scalar or vector S_t.
    """
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    horiz = m - t
    num = (1.0 + float(r_d)) ** horiz
    den = (1.0 + float(r_f)) ** horiz
    if den <= 0:
        den = 1e-12
    return S_t * (num / den)

def dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_mid, spread_bps):
    """
    Convert an FC/DC forward mid into DC/FC bid-ask with a symmetric spread in basis points.
    - mid_dc_fc = 1 / mid_fc_dc
    - bid = mid_dc_fc * (1 - s/2), ask = mid_dc_fc * (1 + s/2), where s = spread_bps / 10,000
    """
    mid_dc_fc = 1.0 / F_fc_dc_mid
    s = max(0.0, float(spread_bps)) / 10000.0
    bid = mid_dc_fc * (1.0 - s / 2.0)
    ask = mid_dc_fc * (1.0 + s / 2.0)
    return bid, ask

def simulate_spot_paths(S0, sigma, n_sims, T=10, seed=123):
    """
    Lognormal spot simulation with zero drift (risk-neutral aside from rate parity built into forwards).
    S_{t} = S_{t-1} * exp(sigma * epsilon_t), epsilon_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, T+1), dtype=float)
    paths[:, 0] = S0
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    for t in range(1, T+1):
        eps = rng.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(sigma * eps)
    return paths

def compute_strategy_results_constant_hedge(
    S_paths,
    S0,
    DF_d_0, DF_f_0,
    r_d, r_f,
    costs_dc, revenue_fc,
    spread_bps,
    hedge_frac,
    strategy="all_at_t0"
):
    """
    Compute PV results under a hedging strategy with a CONSTANT hedge fraction across years.

    Inputs
    ------
    S_paths : (n_sims, T+1) simulated spot paths (FC/DC)
    S0      : scalar initial spot S_0 (FC/DC)
    DF_d_0  : domestic discount factors at time 0, length T+1
    DF_f_0  : foreign discount factors at time 0, length T+1 (kept for extensibility)
    r_d, r_f: single annual domestic/foreign rates (constants, used for forwards)
    costs_dc: length T, DOM cash costs each year (already in DC)
    revenue_fc: length T, FC revenues each year
    spread_bps: forward bid-ask spread in basis points
    hedge_frac: scalar in [0,1], fraction of each year's revenue to hedge
    strategy : "all_at_t0" or "roll_one_year"

    Returns
    -------
    Dict with per-simulation PVs and summary stats.
    """
    n_sims = S_paths.shape[0]
    T = 10
    DF_d = DF_d_0  # alias

    # Containers for yearly DC revenue/cost across simulations
    dc_revenue_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)

    h = float(hedge_frac)
    h = min(max(h, 0.0), 1.0)  # clamp to [0,1]

    if strategy == "all_at_t0":
        # At t=0, lock the hedged portion of each year's revenue at F_{0->t}
        for t in range(1, T+1):
            # Forward FC/DC based on S0 (scalar)
            F_fc_dc_0t = forward_fc_per_dc_constant_rate(S0, r_d, r_f, 0, t)
            bid_dc_fc, _ = dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_0t, spread_bps)

            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            # Hedged portion converts at forward's DC/FC bid (same for all sims because S0 is scalar)
            dc_from_forward = hedged_fc * bid_dc_fc

            # Unhedged portion converts at spot S_t (varies by sim)
            S_t = S_paths[:, t]
            S_t_safe = np.maximum(S_t, 1e-12)
            dc_from_unhedged = unhedged_fc / S_t_safe

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    elif strategy == "roll_one_year":
        # Each year t-1, lock the hedged portion for year t via a 1y forward
        num = (1.0 + float(r_d))
        den = (1.0 + float(r_f))
        if den <= 0:
            den = 1e-12
        for t in range(1, T+1):
            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            S_prev = S_paths[:, t-1]
            # 1y forward FC/DC for each path: F = S_{t-1} * (1+r_d)/(1+r_f)
            F_fc_dc_prev_t = S_prev * (num / den)
            mid_dc_fc = 1.0 / F_fc_dc_prev_t
            s = max(0.0, float(spread_bps)) / 10000.0
            bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)

            dc_from_forward = hedged_fc * bid_dc_fc

            S_t = S_paths[:, t]
            S_t_safe = np.maximum(S_t, 1e-12)
            dc_from_unhedged = unhedged_fc / S_t_safe

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    else:
        raise ValueError("Unknown strategy option. Use 'all_at_t0' or 'roll_one_year'.")

    # Deterministic DC costs each year (same across sims)
    for t in range(1, T+1):
        dc_costs_t[:, t-1] = costs_dc[t-1]

    # Present values (discount in domestic currency using DF_d)
    pv_revenue_per_sim = np.sum(dc_revenue_t * DF_d[1:][None, :], axis=1)
    pv_cost_per_sim    = np.sum(dc_costs_t   * DF_d[1:][None, :], axis=1)
    pv_profit_per_sim  = pv_revenue_per_sim - pv_cost_per_sim

    # Clean non-finites
    pv_revenue_per_sim = np.where(np.isfinite(pv_revenue_per_sim), pv_revenue_per_sim, np.nan)
    pv_cost_per_sim    = np.where(np.isfinite(pv_cost_per_sim),    pv_cost_per_sim,    np.nan)
    pv_profit_per_sim  = np.where(np.isfinite(pv_profit_per_sim),  pv_profit_per_sim,  np.nan)

    def _nanstd(x):
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return 0.0
        return float(np.std(x, ddof=1))

    return {
        "pv_revenue_per_sim": pv_revenue_per_sim,
        "pv_cost_per_sim": pv_cost_per_sim,
        "pv_profit_per_sim": pv_profit_per_sim,
        "avg_pv_revenue": float(np.nanmean(pv_revenue_per_sim)) if np.isfinite(pv_revenue_per_sim).any() else float("nan"),
        "avg_pv_cost": float(np.nanmean(pv_cost_per_sim)) if np.isfinite(pv_cost_per_sim).any() else float("nan"),
        "avg_pv_profit": float(np.nanmean(pv_profit_per_sim)) if np.isfinite(pv_profit_per_sim).any() else float("nan"),
        "std_pv_profit": _nanstd(pv_profit_per_sim),
    }

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Currency Risk Hedging Simulator", layout="wide")
st.title("💱 Currency Risk Hedging Simulator")

st.write(
    "Rates and spots are shown as **FOREIGN per 1 DOMESTIC (FC/DC)**. "
    "To convert foreign amount (FC) to domestic (DC), divide by the rate."
)

# Sidebar controls
st.sidebar.header("Simulation Controls")
S0 = st.sidebar.number_input("Current spot S₀ (FOREIGN per DOMESTIC)", min_value=1e-9, value=0.95, step=0.01, format="%.6f")
sigma_input = st.sidebar.number_input("Annual volatility σ (percent, log spot)", min_value=0.0, value=10.0, step=0.5)
sigma = sigma_input / 100.0
spread_bps = st.sidebar.number_input("Forward bid-ask spread (basis points)", min_value=0.0, value=25.0, step=1.0)
n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100))
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))

st.sidebar.markdown("---")
st.sidebar.header("Inputs")
r_d_pct = st.sidebar.number_input("Domestic interest rate r_d (% per year)", value=5.0, step=0.25, format="%.4f")
r_f_pct = st.sidebar.number_input("Foreign interest rate r_f  (% per year)", value=3.0, step=0.25, format="%.4f")
hedge_frac_pct = st.sidebar.number_input("Hedge fraction of revenue h (% of each year)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, format="%.1f")

# Convert to decimals
r_d = r_d_pct / 100.0
r_f = r_f_pct / 100.0
hedge_frac = hedge_frac_pct / 100.0

# Validate rates
if (1.0 + r_d) <= 0.0 or (1.0 + r_f) <= 0.0:
    st.error("Rates must be greater than -100%. Please adjust r_d and r_f.")
    st.stop()

# Build discount factors from CONSTANT rates (length 11 for t=0..10)
DF_d_0 = make_discount_factors_constant(r_d, T=10)
DF_f_0 = make_discount_factors_constant(r_f, T=10)  # kept for completeness/extensibility

# Cash flows
st.subheader("Cash Flows")
st.caption("Costs in DOM, Revenues in FOR (foreign currency). Provide amounts for years 1–10.")
years = list(range(1, 11))
costs_df = st.data_editor(pd.DataFrame({"Year": years, "Cost (DOM)": [0.0]*10}), num_rows="fixed", use_container_width=True)
revs_df  = st.data_editor(pd.DataFrame({"Year": years, "Revenue (FOR)": [0.0]*10}), num_rows="fixed", use_container_width=True)

costs_dc   = costs_df["Cost (DOM)"].to_numpy(dtype=float)
revenue_fc = revs_df["Revenue (FOR)"].to_numpy(dtype=float)

# Tabs
tabs = st.tabs(["Compare Constant-Fraction Strategies"])

with tabs[0]:
    st.markdown("### Constant Hedge Fraction (h) — Strategy Comparison")
    st.caption(
        "- **Unhedged (h=0)**: 100% converts at spot S_t. (Forward spreads/strategy irrelevant.)\n"
        "- **Hedge-all-at-0**: for each year t, hedge `h × revenue_t` at t=0 using F₀,ₜ = S₀ × ((1+r_d)^t / (1+r_f)^t); "
        "the remaining (1−h) converts at spot S_t.\n"
        "- **Roll 1-Year**: each year t−1, hedge `h × revenue_t` for year t using F_{t-1,t} = S_{t-1} × (1+r_d)/(1+r_f); "
        "the remaining (1−h) converts at spot S_t."
    )

    if st.button("Simulate"):
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

        # --- Strategy A: Hedge-all-at-0 (constant h) ---
        res_A = compute_strategy_results_constant_hedge(
            S_paths=S_paths, S0=S0,
            DF_d_0=DF_d_0, DF_f_0=DF_f_0,
            r_d=r_d, r_f=r_f,
            costs_dc=costs_dc, revenue_fc=revenue_fc,
            spread_bps=spread_bps,
            hedge_frac=hedge_frac,
            strategy="all_at_t0"
        )

        # --- Strategy B: Roll 1-Year (constant h) ---
        res_B = compute_strategy_results_constant_hedge(
            S_paths=S_paths, S0=S0,
            DF_d_0=DF_d_0, DF_f_0=DF_f_0,
            r_d=r_d, r_f=r_f,
            costs_dc=costs_dc, revenue_fc=revenue_fc,
            spread_bps=spread_bps,
            hedge_frac=hedge_frac,
            strategy="roll_one_year"
        )

        # --- Unhedged baseline (h = 0) ---
        res_U = compute_strategy_results_constant_hedge(
            S_paths=S_paths, S0=S0,
            DF_d_0=DF_d_0, DF_f_0=DF_f_0,
            r_d=r_d, r_f=r_f,
            costs_dc=costs_dc, revenue_fc=revenue_fc,
            spread_bps=spread_bps,
            hedge_frac=0.0,                 # key line (h=0)
            strategy="all_at_t0"            # strategy irrelevant when h=0
        )

        # Summary table
        summary = pd.DataFrame({
            "Strategy": ["Unhedged (h=0)", "Hedge-all-at-0", "Roll 1-Year"],
            "Hedge Fraction h": [0.0, hedge_frac, hedge_frac],
            "Avg PV Revenue (DOM)": [res_U["avg_pv_revenue"], res_A["avg_pv_revenue"], res_B["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_U["avg_pv_cost"],    res_A["avg_pv_cost"],    res_B["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_U["avg_pv_profit"],  res_A["avg_pv_profit"],  res_B["avg_pv_profit"]],
            "StdDev PV Profit":     [res_U["std_pv_profit"],  res_A["std_pv_profit"],  res_B["std_pv_profit"]],
        })
        st.dataframe(summary, use_container_width=True)

        # Histograms
        st.markdown("#### PV Profit Distribution")
        plot_list = [
            ("Unhedged (h=0): PV Profit (DOM)", res_U["pv_profit_per_sim"]),
            ("Hedge-all-at-0: PV Profit (DOM)", res_A["pv_profit_per_sim"]),
            ("Roll 1-Year: PV Profit (DOM)",    res_B["pv_profit_per_sim"]),
        ]
        for title, arr in plot_list:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                st.warning(f"No finite values to plot for '{title}'. Check inputs (rates > -100%, etc.).")
            else:
                fig = plt.figure()
                plt.hist(finite, bins=30)
                plt.title(title)
                st.pyplot(fig)

# End of file
