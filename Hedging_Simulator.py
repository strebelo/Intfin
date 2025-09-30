
# ------------------------------
# Currency Risk Hedging Simulator (Streamlit) — Safe Version
# ------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Helper functions
# ------------------------------

def interpolate_curve(curve_vals):
    arr = np.array(curve_vals, dtype=float)  # shape (10,)
    if np.all(np.isnan(arr)):
        return np.zeros(10, dtype=float)
    idx = np.arange(10)
    known = ~np.isnan(arr)
    if not known[0]:
        first_idx = np.argmax(known)
        arr[:first_idx] = arr[first_idx]
        known[:first_idx] = True
    if not known[-1]:
        last_idx = len(known) - 1 - np.argmax(known[::-1])
        arr[last_idx+1:] = arr[last_idx]
        known[last_idx+1:] = True
    nans = np.isnan(arr)
    if np.any(nans):
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr

def make_discount_factors(yields_decimal):
    DF = np.ones(11, dtype=float)
    for t in range(1, 11):
        r = yields_decimal[t-1]
        denom = (1.0 + r)
        if denom <= 0:
            # Invalid yield (<= -100%); use a tiny positive to avoid division by zero and flag later
            denom = 1e-9
        DF[t] = 1.0 / (denom ** t)
    return DF

def forward_fc_per_dc(S_t, DF_d_0, DF_f_0, t, m):
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    num = DF_f_0[m] * DF_d_0[t]
    den = DF_f_0[t] * DF_d_0[m]
    if den == 0:
        den = 1e-12
    return S_t * (num / den)

def dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_mid, spread_bps):
    mid_dc_fc = 1.0 / F_fc_dc_mid
    s = max(0.0, float(spread_bps)) / 10000.0
    bid = mid_dc_fc * (1.0 - s / 2.0)
    ask = mid_dc_fc * (1.0 + s / 2.0)
    return bid, ask

def simulate_spot_paths(S0, sigma, n_sims, T=10, seed=123):
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

def compute_strategy_results(S_paths, DF_d_0, DF_f_0, costs_dc, revenue_fc,
                             spread_bps, strategy="all_at_t0", custom_sells=None, custom_buys=None):
    n_sims = S_paths.shape[0]
    T = 10
    DF_d = DF_d_0  # alias
    dc_revenue_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)

    if strategy == "all_at_t0":
        for t in range(1, T+1):
            F_fc_dc_0t = forward_fc_per_dc(S_paths[0, 0], DF_d, DF_f_0, 0, t)
            bid_dc_fc, _ = dc_per_fc_bid_ask_from_fc_per_dc(F_fc_dc_0t, spread_bps)
            hedged_fc = revenue_fc[t-1]
            dc_from_forward = hedged_fc * bid_dc_fc
            total_dc = dc_from_forward  # no unhedged in this baseline
            dc_revenue_t[:, t-1] = total_dc

    elif strategy == "roll_one_year":
        for t in range(1, T+1):
            S_prev = S_paths[:, t-1]
            F_fc_dc_prev_t = S_prev * (DF_f_0[t] * DF_d[t-1]) / max(DF_f_0[t-1] * DF_d[t], 1e-12)
            mid_dc_fc = 1.0 / F_fc_dc_prev_t
            s = max(0.0, float(spread_bps)) / 10000.0
            bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)
            hedged_fc = revenue_fc[t-1]
            dc_from_forward = hedged_fc * bid_dc_fc
            dc_revenue_t[:, t-1] = dc_from_forward

    elif strategy == "custom":
        if custom_sells is None or custom_buys is None:
            raise ValueError("Custom strategy requires both sells and buys matrices.")
        sells = np.array(custom_sells, dtype=float)
        buys  = np.array(custom_buys, dtype=float)
        sells = np.nan_to_num(sells, nan=0.0)
        buys  = np.nan_to_num(buys,  nan=0.0)

        net_hedge_fc = np.sum(sells, axis=0) - np.sum(buys, axis=0)  # length 10

        for t in range(1, T+1):
            dc_from_forwards = np.zeros(n_sims, dtype=float)

            for k in range(0, t):
                amount_fc = sells[k, t-1]
                if amount_fc != 0.0:
                    S_k = S_paths[:, k]
                    denom = max(DF_f_0[k] * DF_d[t], 1e-12)
                    F_fc_dc_k_t = S_k * (DF_f_0[t] * DF_d[k]) / denom
                    mid_dc_fc = 1.0 / F_fc_dc_k_t
                    s = max(0.0, float(spread_bps)) / 10000.0
                    bid_dc_fc = mid_dc_fc * (1.0 - s / 2.0)
                    dc_from_forwards += amount_fc * bid_dc_fc

            for k in range(0, t):
                amount_fc = buys[k, t-1]
                if amount_fc != 0.0:
                    S_k = S_paths[:, k]
                    denom = max(DF_f_0[k] * DF_d[t], 1e-12)
                    F_fc_dc_k_t = S_k * (DF_f_0[t] * DF_d[k]) / denom
                    mid_dc_fc = 1.0 / F_fc_dc_k_t
                    s = max(0.0, float(spread_bps)) / 10000.0
                    ask_dc_fc = mid_dc_fc * (1.0 + s / 2.0)
                    dc_from_forwards -= amount_fc * ask_dc_fc

            S_t = S_paths[:, t]
            unhedged_fc = revenue_fc[t-1] - net_hedge_fc[t-1]
            # Lognormal spot ensures positive; still guard extremely small values
            S_t_safe = np.maximum(S_t, 1e-12)
            dc_from_unhedged = unhedged_fc / S_t_safe
            dc_revenue_t[:, t-1] = dc_from_forwards + dc_from_unhedged

    else:
        raise ValueError("Unknown strategy option.")

    for t in range(1, T+1):
        dc_costs_t[:, t-1] = costs_dc[t-1]

    pv_revenue_per_sim = np.sum(dc_revenue_t * DF_d[1:][None, :], axis=1)
    pv_cost_per_sim    = np.sum(dc_costs_t   * DF_d[1:][None, :], axis=1)
    pv_profit_per_sim  = pv_revenue_per_sim - pv_cost_per_sim

    # Replace non-finite with NaN; caller will handle plotting/summary robustly
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
    "Spot & forwards are **FOREIGN per 1 DOMESTIC** (FC/DC). "
    "To convert FC to DC, divide by the rate."
)

# Sidebar controls
st.sidebar.header("Simulation Controls")
S0 = st.sidebar.number_input("Current spot S₀ (FOREIGN per DOMESTIC)", min_value=1e-9, value=0.95, step=0.01, format="%.6f")
sigma_input = st.sidebar.number_input("Annual volatility σ (percent, log spot)", min_value=0.0, value=10.0, step=0.5)
sigma = sigma_input / 100.0
spread_bps = st.sidebar.number_input("Forward bid‑ask spread (basis points)", min_value=0.0, value=25.0, step=1.0)
n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100))
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))

# Yield curves
st.subheader("Yield Curves (Zero‑Coupon, Annual Compounding)")
st.caption("Enter yields in percent for maturities 1–10. Leave blanks to interpolate; endpoints are filled.")

years = list(range(1, 11))
dc_curve_df = st.data_editor(pd.DataFrame({"Maturity": years, "Domestic Yield (%)": [np.nan]*10}), num_rows="fixed", use_container_width=True)
fc_curve_df = st.data_editor(pd.DataFrame({"Maturity": years, "Foreign Yield (%)":  [np.nan]*10}), num_rows="fixed", use_container_width=True)

r_d_input = dc_curve_df["Domestic Yield (%)"].to_numpy(dtype=float) / 100.0
r_f_input = fc_curve_df["Foreign Yield (%)"].to_numpy(dtype=float)  / 100.0
r_d = interpolate_curve(r_d_input)
r_f = interpolate_curve(r_f_input)

# Validate yields: forbid r <= -99% which breaks discounting
if np.any(1.0 + r_d <= 0.0) or np.any(1.0 + r_f <= 0.0):
    st.error("Yields must be greater than -100%. Please adjust the yield curves.")
    st.stop()

DF_d_0 = make_discount_factors(r_d)
DF_f_0 = make_discount_factors(r_f)

# Cash flows
st.subheader("Cash Flows")
st.caption("Costs in DOM, Revenues in FOR.")

costs_df = st.data_editor(pd.DataFrame({"Year": years, "Cost (DOM)": [0.0]*10}), num_rows="fixed", use_container_width=True)
revs_df  = st.data_editor(pd.DataFrame({"Year": years, "Revenue (FOR)": [0.0]*10}), num_rows="fixed", use_container_width=True)

costs_dc   = costs_df["Cost (DOM)"].to_numpy(dtype=float)
revenue_fc = revs_df["Revenue (FOR)"].to_numpy(dtype=float)

# Tabs
tabs = st.tabs(["Compare Built‑In Strategies", "Custom Forward Matrix"])

with tabs[0]:
    st.markdown("### Hedge‑all‑at‑0 vs Roll 1‑Year")
    if st.button("Simulate (Built‑In Strategies)"):
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

        res_A = compute_strategy_results(S_paths, DF_d_0, DF_f_0, costs_dc, revenue_fc, spread_bps, strategy="all_at_t0")
        res_B = compute_strategy_results(S_paths, DF_d_0, DF_f_0, costs_dc, revenue_fc, spread_bps, strategy="roll_one_year")

        summary = pd.DataFrame({
            "Strategy": ["Hedge‑all‑at‑0", "Roll 1‑Year"],
            "Avg PV Revenue (DOM)": [res_A["avg_pv_revenue"], res_B["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_A["avg_pv_cost"],    res_B["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_A["avg_pv_profit"],  res_B["avg_pv_profit"]],
            "StdDev PV Profit":     [res_A["std_pv_profit"],  res_B["std_pv_profit"]],
        })
        st.dataframe(summary, use_container_width=True)

        st.markdown("#### PV Profit Distribution")
        # Plot only finite values; show a message if none are finite
        for title, arr in [("Hedge‑all‑at‑0: PV Profit (DOM)", res_A["pv_profit_per_sim"]),
                           ("Roll 1‑Year: PV Profit (DOM)",    res_B["pv_profit_per_sim"])]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                st.warning(f"No finite values to plot for '{title}'. Check inputs (yields > -100%, etc.).")
            else:
                fig = plt.figure()
                plt.hist(finite, bins=30)
                plt.title(title)
                st.pyplot(fig)

with tabs[1]:
    st.markdown("### Custom Forward Matrix (Foreign‑Currency Notionals)")
    st.write("Rows = trade year 0..9; columns = maturity 1..10. Only fill cells where maturity > trade year.")

    idx_trade = [f"t={k}" for k in range(0, 10)]
    cols_mat  = [f"m={m}" for m in range(1, 11)]
    zero_mat = pd.DataFrame(np.zeros((10, 10)), index=idx_trade, columns=cols_mat)

    if "sells_df" not in st.session_state:
        st.session_state["sells_df"] = zero_mat.copy()
    if "buys_df" not in st.session_state:
        st.session_state["buys_df"] = zero_mat.copy()

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Autofill: Hedge‑all‑at‑0"):
            df = zero_mat.copy()
            for m in range(1, 11):
                df.iloc[0, m-1] = revenue_fc[m-1]
            st.session_state["sells_df"] = df
            st.session_state["buys_df"] = zero_mat.copy()
    with colB:
        if st.button("Autofill: Roll 1‑Year"):
            df = zero_mat.copy()
            for t in range(1, 11):
                df.iloc[t-1, t-1] = revenue_fc[t-1]
            st.session_state["sells_df"] = df
            st.session_state["buys_df"] = zero_mat.copy()
    with colC:
        if st.button("Reset to Zeros"):
            st.session_state["sells_df"] = zero_mat.copy()
            st.session_state["buys_df"] = zero_mat.copy()

    st.markdown("**SELL Foreign Currency (FC):**")
    sells_df = st.data_editor(st.session_state["sells_df"], use_container_width=True, key="custom_sells")
    st.markdown("**BUY Foreign Currency (FC):**")
    buys_df  = st.data_editor(st.session_state["buys_df"],  use_container_width=True, key="custom_buys")

    if st.button("Simulate (Custom Strategy)"):
        S_paths = simulate_spot_paths(S0=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

        res_C = compute_strategy_results(
            S_paths, DF_d_0, DF_f_0, costs_dc, revenue_fc, spread_bps,
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
            st.warning("No finite values to plot for Custom strategy. Check inputs (e.g., yields > -100%).")
        else:
            fig = plt.figure()
            plt.hist(finite, bins=30)
            plt.title("Custom Strategy: PV Profit (DOM)")
            st.pyplot(fig)

st.markdown("---")
st.markdown(
    "- **Validity checks**: yields must be greater than **-100%**; spot must be positive; spread is non‑negative.\n"
    "- **If a chart doesn't render**: the app now shows a warning instead of crashing (caused by non‑finite results)."
)
