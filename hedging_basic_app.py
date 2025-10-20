# ==========================
# basic_app.py
# ==========================
# Streamlit app — Constant h_t (basic)
# - LaTeX-rendered math explanations
# - Bid-ask spread scales per year of tenor
# - Overlayed PV-profit distributions (Hedge-all-at-0 vs Rolling 1-Year)
# - Cumulative hedge-converted cash flows line chart (both strategies)
# - NEW: Population expected DC cash flow per year (Unhedged vs Hedge-all-at-0 vs Rolling 1-Year)
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
        # per-year DC revenues (mean across sims) for cumulative chart
        "mean_dc_rev_by_year": np.nanmean(dc_rev_t, axis=0),  # shape (T,)
    }
    return out

# ------------------------------
# NEW: Closed-form population expectations (no simulation)
# ------------------------------
def expected_S_t_population(S0: float, infl_diff: float, t: int) -> float:
    """
    Under the simulation design (mu = ln(1+pi_Delta) - 0.5*sigma^2),
    E[S_t] = S0 * (1 + pi_Delta)^t regardless of sigma.
    """
    return float(S0) * (1.0 + float(infl_diff))**int(t)

def expected_dc_revenue_paths_population(
    S0: float,
    infl_diff: float,
    r_d: float,
    r_f: float,
    spread_bps_per_year: float,
    revenue_fc: np.ndarray,
    h: float,
) -> dict:
    """
    Compute the PER-YEAR population expected DC revenue for:
    - Unhedged
    - Hedge-all-at-0 (bid forward with tenor=t)
    - Rolling 1-Year (bid forward with tenor=1 each year based on S_{t-1})
    """
    T = len(revenue_fc)
    h = float(np.clip(h, 0.0, 1.0))
    e_cf_unhedged = np.zeros(T, dtype=float)
    e_cf_all0     = np.zeros(T, dtype=float)
    e_cf_roll     = np.zeros(T, dtype=float)

    # rolling: 1-year ratio and 1-year bid spread
    ratio_1y = (1.0 + float(r_d)) / max(1e-12, (1.0 + float(r_f)))
    s1 = max(0.0, float(spread_bps_per_year)) / 10000.0  # per-year total bps → fraction
    bid_factor_1y = (1.0 - s1/2.0)

    for t in range(1, T+1):
        rev_fc = float(revenue_fc[t-1])

        # E[S_t]
        E_S_t = expected_S_t_population(S0, infl_diff, t)

        # Unhedged: E[rev_fc * S_t]
        e_unh = rev_fc * E_S_t
        e_cf_unhedged[t-1] = e_unh

        # Hedge-all-at-0:
        #  hedged leg uses bid F_{0->t} mid with tenor t, then bid spread scaled by years=t
        F_mid_0t = forward_dc_per_fc_constant_rate(S0, r_d, r_f, 0, t)
        bid_0t, _ = bid_ask_from_mid_tenor_scaled(F_mid_0t, spread_bps_per_year, years=t)
        hedged_fc   = h * rev_fc
        unhedged_fc = (1.0 - h) * rev_fc
        e_all0 = hedged_fc * bid_0t + unhedged_fc * E_S_t
        e_cf_all0[t-1] = e_all0

        # Rolling 1-Year:
        #  At t-1, forward mid = S_{t-1} * ratio_1y. Bid = mid * (1 - s1/2).
        #  E[bid_prev_t] = bid_factor_1y * ratio_1y * E[S_{t-1}]
        E_S_tm1 = expected_S_t_population(S0, infl_diff, t-1)
        E_bid_prev_t = bid_factor_1y * ratio_1y * E_S_tm1
        e_roll = hedged_fc * E_bid_prev_t + unhedged_fc * E_S_t
        e_cf_roll[t-1] = e_roll

    return {
        "unhedged": e_cf_unhedged,
        "all_at_0": e_cf_all0,
        "roll_1y": e_cf_roll,
    }

# ------------------------------
# Streamlit UI (basic)
# ------------------------------
st.set_page_config(page_title="Currency Hedging — Basic (constant h_t)", layout="wide")
st.title("💱 Currency Risk Hedging — Basic (constant $h_t$)")

with st.expander("Show math / notation"):
    st.latex(r"F_{t,m} = S_t \cdot \frac{(1+r_d)^{m-t}}{(1+r_f)^{m-t}}")
    st.latex(r"\mu = \ln(1+\pi_{\Delta}) - \tfrac{1}{2}\sigma^2,\quad \mathbb{E}\!\left[\frac{S_t}{S_{t-1}}\right]=1+\pi_{\Delta}")
    st.latex(r"\text{h is the fraction of foreign currency revenue hedged}")
    st.latex(r"\text{PV(profit)} = \sum_{t=1}^T \left(\text{revenue}_t^{(DC)}-\text{cost}_t^{(DC)}\right)\cdot DF_d(t)")
  

st.sidebar.header("Inputs")
S0 = st.sidebar.number_input("Current spot S0 (DC/FC)", min_value=1e-9, value=1.05, step=0.01, format="%.6f")
T = int(st.sidebar.number_input("Time horizon T (years)", min_value=1, max_value=50, value=3, step=1))
sigma = st.sidebar.number_input("Volatility σ (%/yr)", min_value=0.0, value=10.0, step=0.5)/100.0
infl_diff = st.sidebar.number_input("Inflation diff (DOM−FOR, %/yr)", value=0.0, step=0.25, format="%.4f")/100.0
r_d = st.sidebar.number_input("Domestic rate r_d (%/yr)", value=3.0, step=0.25, format="%.4f")/100.0
r_f = st.sidebar.number_input("Foreign rate r_f (%/yr)", value=5.0, step=0.25, format="%.4f")/100.0
spread_bps_per_year = st.sidebar.number_input("Forward bid–ask spread (bps **per year** of tenor)", min_value=0.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
h_pct = st.sidebar.number_input("Hedge fraction h (% of each year)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, format="%.1f")
h = h_pct/100.0
n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100))
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))

if (1.0+r_d)<=0 or (1.0+r_f)<=0:
    st.error("Rates must be > -100%.")
    st.stop()
if (1.0+infl_diff)<=0:
    st.error("Inflation differential too negative: require 1 + (DOM−FOR) > 0.")
    st.stop()

DF_d = make_discount_factors_constant(r_d, T)
# Cash flows
st.subheader("Cash Flows")
st.caption(f"Costs (DOM) and Revenues (FOR) for years **1–{T}**.")
cash_df = pd.DataFrame({"Cost (DOM)": [0.0]*T, "Revenue (FOR)": [0.0]*T})
cash_df.index = pd.Index(range(1, T+1), name=f"Year (1–{T})")
cash_df = st.data_editor(cash_df, num_rows="fixed", use_container_width=True)
costs_dc = cash_df["Cost (DOM)"].to_numpy(float)
revenue_fc = cash_df["Revenue (FOR)"].to_numpy(float)

col1, col2 = st.columns([1,1])
with col1:
    simulate_btn = st.button("Run Simulation")
with col2:
    hsweep_btn = st.button("Plot σ vs Mean (H-sweep, constant h)")

def run_paths():
    return simulate_spot_paths_dc_fc_with_infl_drift(
        S0_dc_fc=S0, sigma=sigma, infl_diff=infl_diff,
        n_sims=n_sims, T=T, seed=seed
    )

if simulate_btn:
    S_paths = run_paths()
    # Unhedged baseline (for table only)
    unhedged = results_constant_h(
        S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc,
        spread_bps_per_year, h=0.0, strategy="all_at_t0"
    )
    # Hedge-all-at-0 and Rolling
    all0 = results_constant_h(
        S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc,
        spread_bps_per_year, h=h, strategy="all_at_t0"
    )
    roll = results_constant_h(
        S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc,
        spread_bps_per_year, h=h, strategy="roll_one_year"
    )

    # Summary table
    summary = pd.DataFrame({
        "Strategy": ["Unhedged", "Hedge-all-at-0", "Rolling 1-Year"],
        "Hedge h": [f"{0:.1f}% (constant)", f"{h*100:.1f}% (constant)", f"{h*100:.1f}% (constant)"],
        "Avg PV Revenue (DOM)": [unhedged["avg_pv_revenue"], all0["avg_pv_revenue"], roll["avg_pv_revenue"]],
        "Avg PV Cost (DOM)":    [unhedged["avg_pv_cost"],    all0["avg_pv_cost"],    roll["avg_pv_cost"]],
        "Avg PV Profit (DOM)":  [unhedged["avg_pv_profit"],  all0["avg_pv_profit"],  roll["avg_pv_profit"]],
        "StdDev PV Profit":     [unhedged["std_pv_profit"],  all0["std_pv_profit"],  roll["std_pv_profit"]],
        "Frac(PV Profit < 0)":  [unhedged["frac_neg_profit"],all0["frac_neg_profit"],roll["frac_neg_profit"]],
    })
    fmt = summary.copy()
    for col in ["Avg PV Revenue (DOM)", "Avg PV Cost (DOM)", "Avg PV Profit (DOM)", "StdDev PV Profit"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}")
    fmt["Frac(PV Profit < 0)"] = (fmt["Frac(PV Profit < 0)"]*100.0).map(lambda x: f"{x:.1f}%")
    st.dataframe(fmt, use_container_width=True)

    # Overlayed PV-profit hist (two hedging strategies)
    st.markdown("#### PV Profit — Overlayed Distributions (Hedge-all-at-0 vs Rolling 1-Year)")
    A = all0["pv_profit_per_sim"]; A = A[np.isfinite(A)]
    B = roll["pv_profit_per_sim"]; B = B[np.isfinite(B)]
    if A.size and B.size:
        fig = plt.figure()
        plt.hist(A, bins=40, alpha=0.5, label="Hedge-all-at-0")
        plt.hist(B, bins=40, alpha=0.5, label="Rolling 1-Year")
        plt.xlabel("PV Profit (DOM)"); plt.ylabel("Frequency")
        plt.legend(); plt.title("PV Profit: Overlayed Distributions")
        st.pyplot(fig)
    else:
        st.info("Not enough finite values to plot.")

    # Cumulative hedge-converted DC cash flows (means)
    st.markdown("#### Cumulative Hedge-Converted Cash Flows (mean across sims)")
    tgrid = np.arange(1, T+1)
    cum_all0 = np.cumsum(all0["mean_dc_rev_by_year"])
    cum_roll = np.cumsum(roll["mean_dc_rev_by_year"])
    fig = plt.figure()
    plt.plot(tgrid, cum_all0, marker="o", label="Hedge-all-at-0")
    plt.plot(tgrid, cum_roll, marker="o", label="Rolling 1-Year")
    plt.xlabel("Year t"); plt.ylabel("Cumulative DC revenue (mean)")
    plt.title("Cumulative Hedge-Converted Cash Flows (Mean)")
    plt.legend()
    st.pyplot(fig)

    # ------------------------------
    # NEW: Population expected DC cash flow per year (no simulation)
    # ------------------------------
    st.markdown("#### Population Expected DC Cash Flow per Year (No Simulation)")
    e_cf = expected_dc_revenue_paths_population(
        S0=S0,
        infl_diff=infl_diff,
        r_d=r_d,
        r_f=r_f,
        spread_bps_per_year=spread_bps_per_year,
        revenue_fc=revenue_fc,
        h=h,
    )
    fig = plt.figure()
    plt.plot(tgrid, np.cumsum(e_cf["unhedged"]), marker="o", label="Unhedged (E[CF])")
    plt.plot(tgrid, np.cumsum(e_cf["all_at_0"]), marker="o", label="Hedge-all-at-0 (E[CF])")
    plt.plot(tgrid, np.cumsum(e_cf["roll_1y"]), marker="o", label="Rolling 1-Year (E[CF])")
    plt.xlabel("Year t"); plt.ylabel("Cumulative expected DC cash flow")
    plt.title("Cumulative Expected Cash Flow — Population (Unhedged vs Hedged)")
    plt.legend()
    st.pyplot(fig)

if hsweep_btn:
    S_paths = run_paths()
    hs = np.linspace(0.0, 1.0, 11)
    rows_all0, rows_roll = [], []
    for hv in hs:
        rA = results_constant_h(S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year, h=hv, strategy="all_at_t0")
        rB = results_constant_h(S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year, h=hv, strategy="roll_one_year")
        rows_all0.append({"h": hv, "mean": rA["avg_pv_profit"], "std": rA["std_pv_profit"]})
        rows_roll.append({"h": hv, "mean": rB["avg_pv_profit"], "std": rB["std_pv_profit"]})
    fa = pd.DataFrame(rows_all0); fb = pd.DataFrame(rows_roll)
    fig = plt.figure()
    plt.scatter(fa["std"], fa["mean"], label="Hedge-all-at-0")
    for _, r in fa.iterrows():
        plt.annotate(f"{int(r['h']*100)}%", (r["std"], r["mean"]), textcoords="offset points", xytext=(5,3), fontsize=8)
    plt.scatter(fb["std"], fb["mean"], label="Rolling 1-Year")
    for _, r in fb.iterrows():
        plt.annotate(f"{int(r['h']*100)}%", (r["std"], r["mean"]), textcoords="offset points", xytext=(5,3), fontsize=8)
    plt.xlabel(r"$\sigma(\text{PV Profit})$")
    plt.ylabel(r"$\mathbb{E}[\text{PV Profit}]$")
    plt.title("Frontier over $h \in [0,1]$ (both strategies)")
    plt.legend()
    st.pyplot(fig)
