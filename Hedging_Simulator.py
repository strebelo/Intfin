
# ------------------------------
# Currency Hedging Simulator (Streamlit)
# ------------------------------
# This app lets students simulate currency risk and compare hedging strategies.
# You can:
#   1) Enter a current spot exchange rate S0 quoted as (FOREIGN / DOMESTIC). Example: EUR/USD quoted as EUR per 1 USD.
#   2) Provide annual volatility (stdev) for log spot changes.
#   3) Enter a bid-ask spread for forward contracts (in basis points, i.e., 25 = 0.25% total spread).
#   4) Provide domestic and foreign zero-coupon yield curves for maturities 1..10 years (in %). Leave blanks to interpolate.
#   5) Provide annual cost (in domestic currency) and annual revenue (in foreign currency) for years 1..10.
#   6) Compare two teaching strategies:
#        A) "Hedge-all-at-time-0": sell the entire year's revenue forward at t=0 for each maturity t=1..10
#        B) "Roll one-year hedges": at each year t-1 sell next year's revenue forward (maturity t)
#      and/or define a CUSTOM hedging matrix with forward trades at times 0..9 for maturities up to year 10.
#
# Assumptions & Conventions (IMPORTANT):
#   - Quote convention: S_t is FOREIGN per 1 DOMESTIC (FC/DC). So to convert 1 unit of FOREIGN into DOMESTIC, use (DC = FC / S).
#   - Covered Interest Parity (CIP): Forward (quoted FC/DC) between years t and m (t<m)
#       F_{t,m}^{FC/DC} = S_t * [ DF_f(0,m) * DF_d(0,t) ] / [ DF_f(0,t) * DF_d(0,m) ]
#     where DF_x(0,k) = 1 / (1 + r_x(k))^k are zero-coupon discount factors using the input yield curves.
#   - Forward execution costs (bid/ask):
#       We compute the mid forward in FC/DC and then invert to DC/FC.  We apply the bid-ask spread in DC/FC terms:
#           mid_dc_per_fc = 1 / mid_fc_per_dc
#           bid_dc_per_fc = mid_dc_per_fc * (1 - spread/2)
#           ask_dc_per_fc = mid_dc_per_fc * (1 + spread/2)
#       If you SELL foreign currency forward, you RECEIVE DC at the BID (DC per 1 FC).
#       If you BUY foreign currency forward, you PAY DC at the ASK (DC per 1 FC).
#   - Discounting to present value uses DOMESTIC zero-coupon yields: PV = sum_t Cash_t * DF_d(0,t).
#   - Revenue is in FOREIGN currency; Costs are in DOMESTIC currency.
#   - Random walk in logs for spot: log S_t = log S_{t-1} + sigma * epsilon_t, epsilon ~ N(0,1). Annual steps (t=0..10).
#
# Outputs:
#   - For each strategy (All at t0, Roll 1y, Custom), we compute across simulations:
#       * Average PV(Revenue), PV(Cost), PV(Profit)
#       * Std. Dev. of PV(Profit)
#   - We also show histograms of PV(Profit) where applicable (All at t0 is typically degenerate).
#
# How to run locally:
#   1) Install packages:  pip install streamlit numpy pandas matplotlib
#   2) Run:               streamlit run streamlit_fx_hedge_app.py
#
# Author notes:
#   - The code is intentionally verbose with comments for teaching clarity.
#   - The math is kept simple and explicit to match the classroom description.
# ------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Helper functions
# ------------------------------

def interpolate_curve(curve_vals):
    '''
    Interpolate a length-10 list/array of yields (in decimal), where some entries may be None/np.nan.
    Linear interpolation is used for interior gaps.
    Endpoints are filled by nearest available value.
    If all are NaN, return zeros.
    '''
    arr = np.array(curve_vals, dtype=float)  # shape (10,)
    if np.all(np.isnan(arr)):
        return np.zeros(10, dtype=float)
    # Indices where values are known
    idx = np.arange(10)
    known = ~np.isnan(arr)
    # Forward/backward fill for ends
    # Forward fill
    if not known[0]:
        first_idx = np.argmax(known)
        arr[:first_idx] = arr[first_idx]
        known[:first_idx] = True
    # Backward fill
    if not known[-1]:
        last_idx = len(known) - 1 - np.argmax(known[::-1])
        arr[last_idx+1:] = arr[last_idx]
        known[last_idx+1:] = True
    # Now linearly interpolate remaining interior NaNs
    nans = np.isnan(arr)
    if np.any(nans):
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def make_discount_factors(yields_decimal):
    '''
    Convert zero-coupon yields r(1..10) in DECIMAL into discount factors DF(0,t) for t=0..10:
      DF(0,0)=1, DF(0,t)=1/(1+r(t))^t for t>=1.
    Returns an array of length 11 indexing t=0..10.
    '''
    DF = np.ones(11, dtype=float)
    for t in range(1, 11):
        r = yields_decimal[t-1]  # yield for maturity t
        DF[t] = (1.0 / ((1.0 + r) ** t))
    return DF


def forward_fc_per_dc(S_t, DF_d_0, DF_f_0, t, m):
    '''
    Compute forward (FC/DC) for (t -> m) using time-0 discount factors and spot S_t.
    Formula: F^{FC/DC}_{t,m} = S_t * [ DF_f(0,m) * DF_d(0,t) ] / [ DF_f(0,t) * DF_d(0,m) ]
    '''
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    num = DF_f_0[m] * DF_d_0[t]
    den = DF_f_0[t] * DF_d_0[m]
    return S_t * (num / den)


def dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_mid, spread_bps):
    '''
    Given a forward mid in FC/DC, produce BID/ASK in DC/FC with a symmetric percentage spread.
    Steps:
      - Convert mid to DC/FC: mid_dc_fc = 1 / mid_fc_dc
      - Compute spread fraction: s = spread_bps / 10,000
      - Bid = mid * (1 - s/2), Ask = mid * (1 + s/2)
    Returns: (bid_dc_per_fc, ask_dc_per_fc)
    '''
    mid_dc_fc = 1.0 / F_fc_dc_mid
    s = float(spread_bps) / 10000.0
    bid = mid_dc_fc * (1.0 - s / 2.0)
    ask = mid_dc_fc * (1.0 + s / 2.0)
    return bid, ask


def simulate_spot_paths(S0, sigma, n_sims, T=10, seed=123):
    '''
    Simulate S_t (FC/DC) for t=0..T using a log random walk.
      log S_t = log S_{t-1} + sigma * eps_t, eps ~ N(0,1).
    Returns an array of shape (n_sims, T+1). Column 0 is S0.
    '''
    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, T+1), dtype=float)
    paths[:, 0] = S0
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    for t in range(1, T+1):
        eps = rng.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(sigma * eps)
    return paths


def compute_strategy_results(S_paths, DF_d_0, DF_f_0, r_domestic, costs_dc, revenue_fc,
                             spread_bps, strategy="all_at_t0", custom_sells=None, custom_buys=None):
    '''
    Compute PV results for a given strategy.
    Inputs:
      - S_paths: (n_sims, 11) spot in FC/DC from t=0..10
      - DF_d_0, DF_f_0: discount factors arrays length 11
      - r_domestic: yields (decimal) length 10 (unused here except DF already reflects them)
      - costs_dc: array length 10, costs in DC for t=1..10
      - revenue_fc: array length 10, revenues in FC for t=1..10
      - spread_bps: bid-ask spread in basis points
      - strategy: "all_at_t0", "roll_one_year", or "custom"
      - custom_sells, custom_buys: (10x10) matrices of FC notionals:
            row = trade year k (0..9), col = maturity year m (1..10). Only cells with m>k are meaningful.
    Returns:
      dict with keys:
        "pv_revenue_per_sim", "pv_cost_per_sim", "pv_profit_per_sim",
        "avg_pv_revenue", "avg_pv_cost", "avg_pv_profit", "std_pv_profit"
    '''
    n_sims = S_paths.shape[0]
    T = 10
    # Pre-calc: PV factors DF_d(0,t) (t=1..10). We already have DF_d_0 array with t=0..10.
    DF_d = DF_d_0  # alias

    # Prepare arrays to collect per-simulation DOMESTIC cash flows by period t=1..10
    dc_revenue_t = np.zeros((n_sims, T), dtype=float)  # DOMESTIC revenue (from hedged + unhedged components)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)  # DOMESTIC costs (given)

    # Strategy A: Hedge all revenue at time 0 for each maturity (t=1..10)
    if strategy == "all_at_t0":
        # For each maturity t, compute time-0 forward (FC/DC), then DC/FC bid.
        for t in range(1, T+1):
            F_fc_dc_0t = forward_fc_per_dc(S_paths[0, 0], DF_d, DF_f_0, 0, t)  # same for all sims (uses S0)
            bid_dc_fc, _ask_dc_fc = dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_0t, spread_bps)
            # Sell exactly revenue_fc[t-1] forward at t=0 for maturity t:
            hedged_fc = revenue_fc[t-1]
            dc_from_forward = hedged_fc * bid_dc_fc  # DC received at maturity t
            # Unhedged portion is zero because we hedged full revenue:
            dc_from_unhedged_spot = 0.0
            # Total DC revenue in period t is constant across simulations
            total_dc = dc_from_forward + dc_from_unhedged_spot
            dc_revenue_t[:, t-1] = total_dc

    # Strategy B: Roll one-year hedges: at each t-1, sell revenue for period t
    elif strategy == "roll_one_year":
        for t in range(1, T+1):
            # The forward for maturity t is struck at time t-1 based on S_{t-1} of each simulation path
            # Compute forward mid (FC/DC) per simulation, then DC/FC bid per simulation
            S_prev = S_paths[:, t-1]  # (n_sims,)
            F_fc_dc_prev_t = S_prev * (DF_f_0[t] * DF_d[t-1]) / (DF_f_0[t-1] * DF_d[t])
            # Convert to DC/FC bid with spread
            mid_dc_fc = 1.0 / F_fc_dc_prev_t
            s = float(spread_bps) / 10000.0
            bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)  # (n_sims,)
            # Sell exactly revenue_fc[t-1] forward at t-1 for maturity t
            hedged_fc = revenue_fc[t-1]
            dc_from_forward = hedged_fc * bid_dc_fc  # vector per sim
            # Unhedged revenue is zero (fully hedged that year's revenue)
            dc_from_unhedged_spot = 0.0
            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged_spot

    # Strategy C: Custom matrices
    elif strategy == "custom":
        # Validate matrices exist
        if custom_sells is None or custom_buys is None:
            raise ValueError("Custom strategy requires both sells and buys matrices.")
        sells = np.array(custom_sells, dtype=float)  # shape (10,10) k=0..9, m=1..10
        buys  = np.array(custom_buys, dtype=float)

        # Pre-compute net FC hedged per maturity: net_hedge_fc[m-1] = sum_k (sells[k,m] - buys[k,m])
        net_hedge_fc = np.nansum(np.where(np.isnan(sells), 0.0, sells), axis=0) -                        np.nansum(np.where(np.isnan(buys),  0.0, buys),  axis=0)

        # Loop over maturities t=1..10 and accumulate DC from forwards and unhedged revenue per simulation
        for t in range(1, T+1):
            # DC from forwards maturing at t (summing over trade years k < t)
            dc_from_forwards = np.zeros(n_sims, dtype=float)

            # SELL legs at various trade years k
            for k in range(0, t):
                amount_fc = sells[k, t-1]
                if np.isnan(amount_fc) or amount_fc == 0.0:
                    continue
                # Compute forward mid FC/DC using S_k per sim (except k=0 uses S0 which is constant within sim row)
                S_k = S_paths[:, k]
                F_fc_dc_k_t = S_k * (DF_f_0[t] * DF_d[k]) / (DF_f_0[k] * DF_d[t])
                # DC/FC bid
                mid_dc_fc = 1.0 / F_fc_dc_k_t
                s = float(spread_bps) / 10000.0
                bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)
                # Receive DC for selling FC forward
                dc_from_forwards += amount_fc * bid_dc_fc

            # BUY legs at various trade years k
            for k in range(0, t):
                amount_fc = buys[k, t-1]
                if np.isnan(amount_fc) or amount_fc == 0.0:
                    continue
                S_k = S_paths[:, k]
                F_fc_dc_k_t = S_k * (DF_f_0[t] * DF_d[k]) / (DF_f_0[k] * DF_d[t])
                # DC/FC ask
                mid_dc_fc = 1.0 / F_fc_dc_k_t
                s = float(spread_bps) / 10000.0
                ask_dc_fc = mid_dc_fc * (1.0 + s / 2.0)
                # Pay DC when buying FC forward (negative DC inflow)
                dc_from_forwards -= amount_fc * ask_dc_fc

            # Unhedged portion of REVENUE for period t converted at SPOT S_t
            # Unhedged FC = revenue_fc[t-1] - net_hedge_fc[t-1]
            # (If over-hedged, this becomes negative and the spot conversion will subtract DC accordingly.)
            S_t = S_paths[:, t]
            unhedged_fc = revenue_fc[t-1] - net_hedge_fc[t-1]
            dc_from_unhedged = unhedged_fc / S_t  # vector per sim (FC/DC -> DC by dividing)

            # Total DC revenue at period t
            dc_revenue_t[:, t-1] = dc_from_forwards + dc_from_unhedged

    else:
        raise ValueError("Unknown strategy option.")

    # Costs (in DC) are deterministic per year; copy across simulations
    for t in range(1, T+1):
        dc_costs_t[:, t-1] = costs_dc[t-1]

    # Present value per simulation: sum over t=1..10 of Cash_t * DF_d(0,t)
    pv_revenue_per_sim = np.sum(dc_revenue_t * DF_d[1:][None, :], axis=1)
    pv_cost_per_sim    = np.sum(dc_costs_t   * DF_d[1:][None, :], axis=1)
    pv_profit_per_sim  = pv_revenue_per_sim - pv_cost_per_sim

    return {
        "pv_revenue_per_sim": pv_revenue_per_sim,
        "pv_cost_per_sim": pv_cost_per_sim,
        "pv_profit_per_sim": pv_profit_per_sim,
        "avg_pv_revenue": float(np.mean(pv_revenue_per_sim)),
        "avg_pv_cost": float(np.mean(pv_cost_per_sim)),
        "avg_pv_profit": float(np.mean(pv_profit_per_sim)),
        "std_pv_profit": float(np.std(pv_profit_per_sim, ddof=1)) if len(pv_profit_per_sim) > 1 else 0.0
    }


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Currency Risk Hedging Simulator", layout="wide")

st.title("💱 Currency Risk Hedging Simulator")
st.write("""
This app simulates FX revenue hedging with forward contracts.  
**Quote convention:** Spot and forwards are entered as **FOREIGN per 1 DOMESTIC** (FC/DC).  
**Example:** If DOMESTIC=USD and FOREIGN=EUR, 0.95 means €0.95 per $1.
""")

# ---- Sidebar inputs ----
st.sidebar.header("Simulation Controls")
S0 = st.sidebar.number_input("Current spot S₀ (FOREIGN per DOMESTIC)", min_value=1e-8, value=0.95, step=0.01, format="%.6f")
sigma_input = st.sidebar.number_input("Annual volatility (σ) for log spot (in %)", min_value=0.0, value=10.0, step=0.5)
sigma = sigma_input / 100.0
spread_bps = st.sidebar.number_input("Forward bid-ask spread (basis points)", min_value=0.0, value=25.0, step=1.0)
n_sims = st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100)
seed = st.sidebar.number_input("Random seed (int)", min_value=0, value=42, step=1)

# ---- Yield curves (domestic & foreign) ----
st.subheader("Yield Curves (Zero-Coupon, Annual Compounding)")
st.caption("Enter yields in **percent** for maturities 1..10 years. Leave blanks to interpolate.")

years = [f"Year {i}" for i in range(1, 11)]
default_dc = pd.DataFrame({"Maturity (years)": list(range(1, 11)), "Yield (%)": [np.nan]*10})
default_fc = pd.DataFrame({"Maturity (years)": list(range(1, 11)), "Yield (%)": [np.nan]*10})

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Domestic Yield Curve**")
    dc_curve_df = st.data_editor(default_dc, num_rows="fixed", use_container_width=True, key="dc_yc")
with col2:
    st.markdown("**Foreign Yield Curve**")
    fc_curve_df = st.data_editor(default_fc, num_rows="fixed", use_container_width=True, key="fc_yc")

# Convert to decimals with interpolation
r_d_input = dc_curve_df["Yield (%)"].to_numpy(dtype=float) / 100.0
r_f_input = fc_curve_df["Yield (%)"].to_numpy(dtype=float) / 100.0
r_d = interpolate_curve(r_d_input)
r_f = interpolate_curve(r_f_input)

# Discount factors from time 0
DF_d_0 = make_discount_factors(r_d)
DF_f_0 = make_discount_factors(r_f)

# ---- Cash flows ----
st.subheader("Cash Flows")
st.caption("Enter **Costs** in DOMESTIC currency and **Revenue** in FOREIGN currency for years 1..10.")

default_costs = pd.DataFrame({"Year": list(range(1, 11)), "Cost (DOM)": [0.0]*10})
default_revs  = pd.DataFrame({"Year": list(range(1, 11)), "Revenue (FOR)": [0.0]*10})

c1, c2 = st.columns(2)
with c1:
    costs_df = st.data_editor(default_costs, num_rows="fixed", use_container_width=True, key="costs")
with c2:
    revs_df  = st.data_editor(default_revs,  num_rows="fixed", use_container_width=True, key="revs")

costs_dc   = costs_df["Cost (DOM)"].to_numpy(dtype=float)   # length 10
revenue_fc = revs_df["Revenue (FOR)"].to_numpy(dtype=float) # length 10

# ---- Hedging strategy selection ----
st.subheader("Hedging Strategies")
st.markdown("""
You can compare built-in strategies and/or define a **Custom Forward Matrix**.  
- **Hedge-all-at-time-0:** For each t=1..10, sell the full revenue R_t forward at time 0 (maturities 1..10).  
- **Roll one-year hedges:** Each year t-1, sell R_t forward for maturity t (1-year rolling hedge).  
- **Custom:** Specify amount of foreign currency to SELL/BUY forward at time k (rows) for maturity m (columns).
""")

tabs = st.tabs(["Compare Built-In Strategies", "Custom Forward Matrix"])

# ---- Tab 1: Built-in strategies ----
with tabs[0]:
    st.markdown("### Compare: Hedge-all-at-0 vs. Roll 1-Year")
    if st.button("Simulate (Built-In Strategies)"):
        # Simulate spot paths
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=int(n_sims), T=10, seed=int(seed))

        # Strategy A
        res_A = compute_strategy_results(S_paths, DF_d_0, DF_f_0, r_d, costs_dc, revenue_fc,
                                         spread_bps, strategy="all_at_t0")

        # Strategy B
        res_B = compute_strategy_results(S_paths, DF_d_0, DF_f_0, r_d, costs_dc, revenue_fc,
                                         spread_bps, strategy="roll_one_year")

        # Show summary table
        summary = pd.DataFrame({
            "Strategy": ["Hedge-all-at-0", "Roll 1-Year"],
            "Avg PV Revenue (DOM)": [res_A["avg_pv_revenue"], res_B["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_A["avg_pv_cost"],    res_B["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_A["avg_pv_profit"],  res_B["avg_pv_profit"]],
            "StdDev PV Profit":     [res_A["std_pv_profit"],  res_B["std_pv_profit"]],
        })
        st.dataframe(summary, use_container_width=True)

        # Histogram(s) of PV Profit (All-at-0 usually nearly degenerate)
        st.markdown("#### PV Profit Distribution")
        fig1 = plt.figure()
        plt.hist(res_A["pv_profit_per_sim"], bins=30)
        plt.title("Hedge-all-at-0: PV Profit (DOM)")
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.hist(res_B["pv_profit_per_sim"], bins=30)
        plt.title("Roll 1-Year: PV Profit (DOM)")
        st.pyplot(fig2)

# ---- Tab 2: Custom Matrix ----
with tabs[1]:
    st.markdown("### Custom Forward Matrix")
    st.write("""
Fill in NOTIONAL amounts of **foreign currency** you plan to **SELL** or **BUY** forward.  
**Rows = Trade year (0..9), Columns = Maturity year (1..10).** Only fill cells where maturity > trade year.  
Leave other cells as 0 (or blank). Amounts are in **FOREIGN currency units**.
""")

    # Build default triangle matrices (10x10)
    years_trade = [f"t={k}" for k in range(0, 10)]
    years_mat   = [f"m={m}" for m in range(1, 11)]
    sells_default = pd.DataFrame(np.zeros((10, 10)), index=years_trade, columns=years_mat)
    buys_default  = pd.DataFrame(np.zeros((10, 10)),  index=years_trade, columns=years_mat)

    # Convenience buttons to auto-fill matrices from the revenue schedule
    colA, colB, colC = st.columns(3)
    with colA:
        autofill_all0 = st.button("Autofill: Hedge-all-at-0")
    with colB:
        autofill_roll = st.button("Autofill: Roll 1-Year")
    with colC:
        reset_custom = st.button("Reset to Zeros")

    # Hold current matrices in session state so buttons can set them
    if "sells_df" not in st.session_state:
        st.session_state["sells_df"] = sells_default.copy()
    if "buys_df" not in st.session_state:
        st.session_state["buys_df"] = buys_default.copy()

    # Apply button actions
    if reset_custom:
        st.session_state["sells_df"] = sells_default.copy()
        st.session_state["buys_df"]  = buys_default.copy()

    if autofill_all0:
        # Set row t=0 to sell revenue for all maturities; zero elsewhere
        df = sells_default.copy()
        for m in range(1, 11):
            df.iloc[0, m-1] = revenue_fc[m-1]
        st.session_state["sells_df"] = df
        st.session_state["buys_df"]  = buys_default.copy()

    if autofill_roll:
        # Set sells at (k=t-1, m=t) equal to revenue[t]
        df = sells_default.copy()
        for t in range(1, 11):
            df.iloc[t-1, t-1] = revenue_fc[t-1]
        st.session_state["sells_df"] = df
        st.session_state["buys_df"]  = buys_default.copy()

    st.markdown("**SELL Foreign Currency (notional in FC):**")
    sells_df = st.data_editor(st.session_state["sells_df"], use_container_width=True, key="custom_sells")

    st.markdown("**BUY Foreign Currency (notional in FC):**")
    buys_df  = st.data_editor(st.session_state["buys_df"],  use_container_width=True, key="custom_buys")

    # Simulate custom
    if st.button("Simulate (Custom Strategy)"):
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=int(n_sims), T=10, seed=int(seed))

        res_C = compute_strategy_results(
            S_paths, DF_d_0, DF_f_0, r_d, costs_dc, revenue_fc, spread_bps,
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

        # Histogram of PV Profit for custom
        fig3 = plt.figure()
        plt.hist(res_C["pv_profit_per_sim"], bins=30)
        plt.title("Custom Strategy: PV Profit (DOM)")
        st.pyplot(fig3)

st.markdown("---")
st.markdown("**Notes:**")
st.markdown("""
- **Units:** Spot/forwards are in FOREIGN per DOMESTIC (FC/DC). To convert FC to DC, the app divides by the rate.  
- **CIP & Forwards:** For t>0, forwards are recomputed each year using the simulated spot S_t and the original (time-0) yield curves.  
- **Spread:** The bid-ask spread (in basis points) is applied **in DC per FC terms** after inverting the FC/DC mid forward.  
- **Discounting:** Present values use the domestic discount factors from the domestic yield curve.  
- **Interpolation:** Missing yields are filled with linear interpolation; endpoints use the nearest available value.  
- **Performance tip:** If the app feels slow, reduce the number of simulations.
""")
