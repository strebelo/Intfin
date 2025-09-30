# ------------------------------
# Currency Risk Hedging Simulator (Streamlit) — Constant Rates Version
# ------------------------------
# What changed vs. your previous version?
# - Removed yield curves and interpolation entirely (no term structure).
# - Students now report ONE domestic rate r_d and ONE foreign rate r_f (annual, constant).
# - The app builds discount factors from these constants and computes synthetic forwards
#   via covered interest parity for any t->m horizon.
# - All strategies, custom matrix logic, simulations, and charts remain, now driven by r_d/r_f.
#
# Notation:
#   FC/DC = "foreign currency per 1 unit of domestic currency"
#   To convert foreign amount (FC) to domestic (DC) at rate X = (FC/DC): DC = FC / X
#
# Forward formula (covered interest parity, with constant rates):
#   F_{t->m} (FC/DC) = S_t * [ (1 + r_d)^(m - t) / (1 + r_f)^(m - t) ]
#
# Discount factors with constant r:
#   DF_r[0] = 1
#   DF_r[t] = 1 / (1 + r)^t, for t = 1..T
#
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
        # Guard: invalid rate (<= -100%). Use tiny positive to avoid division by zero.
        denom_base = 1e-9
    for t in range(1, T + 1):
        DF[t] = 1.0 / (denom_base ** t)
    return DF

def forward_fc_per_dc_constant_rate(S_t, r_d, r_f, t, m):
    """
    Synthetic forward (FC/DC) from t to m using covered interest parity with constant rates.
        F_{t->m} = S_t * ((1 + r_d)^(m - t) / (1 + r_f)^(m - t))
    Requires m > t.
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

def _finite(a):
    return a[np.isfinite(a)]

def compute_strategy_results(S_paths,
                             DF_d_0, DF_f_0,
                             r_d, r_f,
                             costs_dc, revenue_fc,
                             spread_bps, strategy="all_at_t0",
                             custom_sells=None, custom_buys=None):
    """
    Compute PV results under a hedging strategy.

    Inputs
    ------
    S_paths : (n_sims, T+1) simulated spot paths (FC/DC)
    DF_d_0  : domestic discount factors at time 0, length T+1
    DF_f_0  : foreign  discount factors at time 0, length T+1
              (kept for extensibility even though built from a single r_f)
    r_d, r_f: single annual domestic/foreign rates (constants, used for forwards)
    costs_dc: length T, DOM cash costs each year (already in DC)
    revenue_fc: length T, FC revenues each year
    spread_bps: forward bid-ask spread in basis points
    strategy : "all_at_t0", "roll_one_year", or "custom"
    custom_sells, custom_buys: 10x10 matrices of FC notionals (only used for strategy="custom")

    Returns
    -------
    Dict with per-simulation PVs and summary stats.
    """
    n_sims = S_paths.shape[0]
    T = 10
    DF_d = DF_d_0  # alias for readability

    dc_revenue_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)

    if strategy == "all_at_t0":
        # Hedge each year's FC revenue at t=0 using F_{0->t}.
        for t in range(1, T+1):
            F_fc_dc_0t = forward_fc_per_dc_constant_rate(S_paths[0, 0], r_d, r_f, 0, t)
            bid_dc_fc, _ = dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_0t, spread_bps)
            hedged_fc = revenue_fc[t-1]
            dc_from_forward = hedged_fc * bid_dc_fc
            dc_revenue_t[:, t-1] = dc_from_forward  # fully hedged baseline

    elif strategy == "roll_one_year":
        # Each year t-1, hedge that year's revenue using 1-year forward from t-1 to t.
        for t in range(1, T+1):
            S_prev = S_paths[:, t-1]
            # F_{t-1->t} with constant rates:
            #   F = S_{t-1} * (1+r_d)/(1+r_f)
            num = (1.0 + float(r_d))
            den = (1.0 + float(r_f))
            if den <= 0:
                den = 1e-12
            F_fc_dc_prev_t = S_prev * (num / den)
            mid_dc_fc = 1.0 / F_fc_dc_prev_t
            s = max(0.0, float(spread_bps)) / 10000.0
            bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)
            hedged_fc = revenue_fc[t-1]
            dc_from_forward = hedged_fc * bid_dc_fc
            dc_revenue_t[:, t-1] = dc_from_forward

    elif strategy == "custom":
        # User supplies a matrix of FC sells/buys at t=k for maturity m (k rows, m columns).
        if custom_sells is None or custom_buys is None:
            raise ValueError("Custom strategy requires both sells and buys matrices.")
        sells = np.array(custom_sells, dtype=float)
        buys  = np.array(custom_buys,  dtype=float)
        sells = np.nan_to_num(sells, nan=0.0)
        buys  = np.nan_to_num(buys,  nan=0.0)

        net_hedge_fc = np.sum(sells, axis=0) - np.sum(buys, axis=0)  # length 10: net FC hedged into each delivery year

        for t in range(1, T+1):
            dc_from_forwards = np.zeros(n_sims, dtype=float)

            # For all trades placed at k < t that deliver at t
            for k in range(0, t):
                amount_fc_sell = sells[k, t-1]
                amount_fc_buy  = buys[k,  t-1]

                if amount_fc_sell != 0.0:
                    S_k = S_paths[:, k]
                    F_fc_dc_k_t = forward_fc_per_dc_constant_rate(S_k, r_d, r_f, k, t)  # vectorized w.r.t. S_k
                    mid_dc_fc = 1.0 / F_fc_dc_k_t
                    s = max(0.0, float(spread_bps)) / 10000.0
                    bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)
                    dc_from_forwards += amount_fc_sell * bid_dc_fc

                if amount_fc_buy != 0.0:
                    S_k = S_paths[:, k]
                    F_fc_dc_k_t = forward_fc_per_dc_constant_rate(S_k, r_d, r_f, k, t)
                    mid_dc_fc = 1.0 / F_fc_dc_k_t
                    s = max(0.0, float(spread_bps)) / 10000.0
                    ask_dc_fc = mid_dc_fc * (1.0 + s / 2.0)
                    dc_from_forwards -= amount_fc_buy * ask_dc_fc

            # Any unhedged FC revenue for year t is converted at spot S_t (FC/DC):
            S_t = S_paths[:, t]
            unhedged_fc = revenue_fc[t-1] - net_hedge_fc[t-1]
            S_t_safe = np.maximum(S_t, 1e-12)
            dc_from_unhedged = unhedged_fc / S_t_safe

            dc_revenue_t[:, t-1] = dc_from_forwards + dc_from_unhedged

    else:
        raise ValueError("Unknown strategy option.")

    # Deterministic DC costs each year
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
st.title("💱 Currency Risk Hedging Simulator (Constant Rates)")

st.write(
    "Spot & forwards are **FOREIGN per 1 DOMESTIC (FC/DC)**. "
    "To convert FC to DC, divide by the rate."
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
st.sidebar.header("Student Inputs: Interest Rates (Constant)")
r_d_pct = st.sidebar.number_input("Domestic interest rate r_d (% per year)", value=5.0, step=0.25, format="%.4f")
r_f_pct = st.sidebar.number_input("Foreign interest rate r_f  (% per year)", value=3.0, step=0.25, format="%.4f")

# Convert to decimals
r_d = r_d_pct / 100.0
r_f = r_f_pct / 100.0

# Validate rates: forbid r <= -100% which breaks discounting/forwards
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
tabs = st.tabs(["Compare Built-In Strategies", "Custom Forward Matrix"])

with tabs[0]:
    st.markdown("### Hedge-all-at-0 vs Roll 1-Year")
    st.caption(
        "- **Hedge-all-at-0**: lock each year’s FC revenue at t=0 using F₀,ₜ = S₀ × ((1+r_d)^(t) / (1+r_f)^(t)).\n"
        "- **Roll 1-Year**: each year t-1, lock year-t revenue using the 1-year forward F_{t-1,t} = S_{t-1} × (1+r_d)/(1+r_f)."
    )
    if st.button("Simulate (Built-In Strategies)"):
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

        res_A = compute_strategy_results(S_paths, DF_d_0, DF_f_0, r_d, r_f, costs_dc, revenue_fc, spread_bps, strategy="all_at_t0")
        res_B = compute_strategy_results(S_paths, DF_d_0, DF_f_0, r_d, r_f, costs_dc, revenue_fc, spread_bps, strategy="roll_one_year")

        summary = pd.DataFrame({
            "Strategy": ["Hedge-all-at-0", "Roll 1-Year"],
            "Avg PV Revenue (DOM)": [res_A["avg_pv_revenue"], res_B["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_A["avg_pv_cost"],    res_B["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_A["avg_pv_profit"],  res_B["avg_pv_profit"]],
            "StdDev PV Profit":     [res_A["std_pv_profit"],  res_B["std_pv_profit"]],
        })
        st.dataframe(summary, use_container_width=True)

        st.markdown("#### PV Profit Distribution")
        for title, arr in [("Hedge-all-at-0: PV Profit (DOM)", res_A["pv_profit_per_sim"]),
                           ("Roll 1-Year: PV Profit (DOM)",    res_B["pv_profit_per_sim"])]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                st.warning(f"No finite values to plot for '{title}'. Check inputs (rates > -100%, etc.).")
            else:
                fig = plt.figure()
                plt.hist(finite, bins=30)
                plt.title(title)
                st.pyplot(fig)

with tabs[1]:
    st.markdown("### Custom Forward Matrix (Foreign-Currency Notionals)")
    st.write("Rows = trade year k=0..9; columns = maturity m=1..10. Only fill cells where m > k.")

    idx_trade = [f"t={k}" for k in range(0, 10)]
    cols_mat  = [f"m={m}" for m in range(1, 11)]
    zero_mat = pd.DataFrame(np.zeros((10, 10)), index=idx_trade, columns=cols_mat)

    if "sells_df" not in st.session_state:
        st.session_state["sells_df"] = zero_mat.copy()
    if "buys_df" not in st.session_state:
        st.session_state["buys_df"] = zero_mat.copy()

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Autofill: Hedge-all-at-0"):
            df = zero_mat.copy()
            for m in range(1, 11):
                df.iloc[0, m-1] = revenue_fc[m-1]
            st.session_state["sells_df"] = df
            st.session_state["buys_df"]  = zero_mat.copy()
    with colB:
        if st.button("Autofill: Roll 1-Year"):
            df = zero_mat.copy()
            for t in range(1, 11):
                df.iloc[t-1, t-1] = revenue_fc[t-1]
            st.session_state["sells_df"] = df
            st.session_state["buys_df"]  = zero_mat.copy()
    with colC:
        if st.button("Reset to Zeros"):
            st.session_state["sells_df"] = zero_mat.copy()
            st.session_state["buys_df"]  = zero_mat.copy()

    st.markdown("**SELL Foreign Currency (FC):**")
    sells_df = st.data_editor(st.session_state["sells_df"], use_container_width=True, key="custom_sells")
    st.markdown("**BUY Foreign Currency (FC):**")
    buys_df  = st.data_editor(st.session_state["buys_df"],  use_container_width=True, key="custom_buys")

    if st.button("Simulate (Custom Strategy)"):
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

        res_C = compute_strategy_results(
            S_paths, DF_d_0, DF_f_0, r_d, r_f, costs_dc, revenue_fc, spread_bps,
            strategy="custom",
            custom_sells=sells_df.to_numpy(dtype=float),
            custom_buys=buys_df.to_numpy(dtype=float)
        )

        summary_C = pd.DataFrame({
            "Strategy": ["Custom"],
            "Avg PV Revenue (DOM)": [res_C["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_C["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_C["avg_pv_profit"]],
            "StdDev PV Profit":     [res_C["std_pv_profit"]],
        })
        st.dataframe(summary_C, use_container_width=True)

        finite = res_C["pv_profit_per_sim"][np.isfinite(res_C["pv_profit_per_sim"])]
        if finite.size == 0:
            st.warning("No finite values to plot for Custom strategy. Check inputs (e.g., rates > -100%).")
        else:
            fig = plt.figure()
            plt.hist(finite, bins=30)
            plt.title("Custom Strategy: PV Profit (DOM)")
            st.pyplot(fig)

st.markdown("---")
st.markdown(
    "- **Student inputs**: one **domestic rate r_d** and one **foreign rate r_f** (annual, constant). "
    "These generate discount factors and **synthetic forwards** via covered interest parity.\n"
    "- **Validity checks**: rates must be greater than **-100%**; spot must be positive; spread non-negative.\n"
    "- **If a chart doesn't render**: the app warns instead of crashing (caused by non-finite results)."
)
