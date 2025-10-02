# ------------------------------
# Currency Risk Hedging Simulator (Streamlit) — DC/FC Quoting
# Constant Hedge Fraction with Unhedged Baseline (h = 0)
# + User-selectable time horizon T
# + σ vs mean (H-sweep) plot
# + Inflation differential-driven drift in spot simulation
# + Unified "Inputs" section (no separate "Simulation Controls")
# ------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Helper functions
# ------------------------------

def make_discount_factors_constant(r: float, T: int):
    """
    Build discount factors DF[0..T] from a single annual rate r (constant across maturities).
        DF[0] = 1
        DF[t] = 1 / (1 + r)^t
    """
    DF = np.ones(T + 1, dtype=float)
    base = 1.0 + float(r)
    if base <= 0:
        base = 1e-9
    for t in range(1, T + 1):
        DF[t] = 1.0 / (base ** t)
    return DF

def forward_dc_per_fc_constant_rate(S_t_dc_fc, r_d, r_f, t, m):
    """
    Synthetic forward (DC/FC) from t to m using covered interest parity with CONSTANT rates:
        F_{t->m}^{DC/FC} = S_t^{DC/FC} * ((1 + r_f)^(m - t) / (1 + r_d)^(m - t))
    Requires m > t. Accepts scalar or vector S_t^{DC/FC}.
    """
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    horiz = m - t
    num = (1.0 + float(r_f)) ** horiz
    den = (1.0 + float(r_d)) ** horiz
    if den <= 0:
        den = 1e-12
    return S_t_dc_fc * (num / den)

def dc_per_fc_bid_ask_from_mid(mid_dc_fc, spread_bps):
    """
    Given a DC/FC forward MID and a symmetric spread in basis points, return (bid, ask) in DC/FC.
      bid = mid * (1 - s/2),  ask = mid * (1 + s/2),  s = spread_bps / 10,000
    Use BID when converting FC→DC (you receive bid).
    """
    s = max(0.0, float(spread_bps)) / 10000.0
    bid = mid_dc_fc * (1.0 - s / 2.0)
    ask = mid_dc_fc * (1.0 + s / 2.0)
    return bid, ask

def simulate_spot_paths_dc_fc_with_infl_drift(S0_dc_fc, sigma, infl_diff, n_sims, T, seed=123):
    """
    Lognormal spot simulation with drift linked to the inflation differential (DOM − FOR):
        Let πΔ = infl_diff (decimal per year). We simulate
            S_t = S_{t-1} * exp( μ_adj + σ * ε_t ),  ε_t ~ N(0,1),
        where μ_adj = ln(1 + πΔ) - 0.5*σ^2 ensures E[S_t/S_{t-1}] = 1 + πΔ.

        Interpretation with DC/FC quoting:
        - If πΔ > 0 (DOM inflation > FOR inflation), then E[S_t] grows ⇒ DC depreciates
          on average versus FOR at rate πΔ, matching PPP intuition.
    Returns array (n_sims, T+1)
    """
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    if (1.0 + infl_diff) <= 0.0:
        raise ValueError("Inflation differential too negative (1 + πΔ must be > 0).")

    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, T+1), dtype=float)
    paths[:, 0] = float(S0_dc_fc)

    # Mean-matching drift so that average depreciation equals the inflation differential
    mu_adj = np.log1p(infl_diff) - 0.5 * (sigma ** 2)

    for t in range(1, T+1):
        eps = rng.standard_normal(n_sims)
        growth = np.exp(mu_adj + sigma * eps)
        paths[:, t] = np.maximum(1e-12, paths[:, t-1] * growth)

    return paths

def compute_strategy_results_constant_hedge(
    S_paths_dc_fc,
    S0_dc_fc,
    DF_d_0, DF_f_0,
    r_d, r_f,
    costs_dc, revenue_fc,
    spread_bps,
    hedge_frac,
    strategy="all_at_t0",
):
    """
    Compute PV results under a hedging strategy with a CONSTANT hedge fraction across years.
    All rates are DC/FC; FC revenue converts to DC via multiplication by a DC/FC rate.
    """
    n_sims, T_plus_1 = S_paths_dc_fc.shape
    T = T_plus_1 - 1
    DF_d = DF_d_0

    dc_revenue_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)

    h = float(hedge_frac)
    h = min(max(h, 0.0), 1.0)

    if strategy == "all_at_t0":
        # At t=0, lock the hedged portion of each year's revenue at F_{0->t}^{DC/FC} (same for all sims)
        for t in range(1, T+1):
            F_dc_fc_0t = forward_dc_per_fc_constant_rate(S0_dc_fc, r_d, r_f, 0, t)
            bid_dc_fc, _ = dc_per_fc_bid_ask_from_mid(F_dc_fc_0t, spread_bps)

            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            dc_from_forward  = hedged_fc * bid_dc_fc
            S_t_dc_fc        = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged = unhedged_fc * S_t_dc_fc

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    elif strategy == "roll_one_year":
        # Each year t−1, hedge year t using 1y forward: F_{t-1->t}^{DC/FC} = S_{t-1} * (1+r_f)/(1+r_d)
        ratio = (1.0 + float(r_f)) / max(1e-12, (1.0 + float(r_d)))
        for t in range(1, T+1):
            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            S_prev_dc_fc       = S_paths_dc_fc[:, t-1]
            F_dc_fc_prev_t_mid = S_prev_dc_fc * ratio
            bid_dc_fc, _       = dc_per_fc_bid_ask_from_mid(F_dc_fc_prev_t_mid, spread_bps)

            dc_from_forward  = hedged_fc * bid_dc_fc
            S_t_dc_fc        = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged = unhedged_fc * S_t_dc_fc

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    else:
        raise ValueError("Unknown strategy option. Use 'all_at_t0' or 'roll_one_year'.")

    # Deterministic DC costs
    for t in range(1, T+1):
        dc_costs_t[:, t-1] = costs_dc[t-1]

    # PVs in domestic currency
    pv_revenue_per_sim = np.sum(dc_revenue_t * DF_d[1:][None, :], axis=1)
    pv_cost_per_sim    = np.sum(dc_costs_t   * DF_d[1:][None, :], axis=1)
    pv_profit_per_sim  = pv_revenue_per_sim - pv_cost_per_sim

    # Clean non-finites
    pv_revenue_per_sim = np.where(np.isfinite(pv_revenue_per_sim), pv_revenue_per_sim, np.nan)
    pv_cost_per_sim    = np.where(np.isfinite(pv_cost_per_sim),    pv_cost_per_sim,    np.nan)
    pv_profit_per_sim  = np.where(np.isfinite(pv_profit_per_sim),  pv_profit_per_sim,  np.nan)

    def _nanstd(x):
        x = x[np.isfinite(x)]
        if x.size <= 1: return 0.0
        return float(np.std(x, ddof=1))

    def _frac_negative(x):
        x = x[np.isfinite(x)]
        if x.size == 0: return float("nan")
        return float(np.mean(x < 0.0))

    return {
        "pv_revenue_per_sim": pv_revenue_per_sim,
        "pv_cost_per_sim": pv_cost_per_sim,
        "pv_profit_per_sim": pv_profit_per_sim,
        "avg_pv_revenue": float(np.nanmean(pv_revenue_per_sim)) if np.isfinite(pv_revenue_per_sim).any() else float("nan"),
        "avg_pv_cost": float(np.nanmean(pv_cost_per_sim)) if np.isfinite(pv_cost_per_sim).any() else float("nan"),
        "avg_pv_profit": float(np.nanmean(pv_profit_per_sim)) if np.isfinite(pv_profit_per_sim).any() else float("nan"),
        "std_pv_profit": _nanstd(pv_profit_per_sim),
        "frac_neg_profit": _frac_negative(pv_profit_per_sim),
    }

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Currency Risk Hedging Simulator", layout="wide")
st.title("💱 Currency Risk Hedging Simulator")

# Unified Inputs (sidebar)
st.sidebar.header("Inputs")

# Core market & model inputs
S0 = st.sidebar.number_input("Current spot S₀ (Domestic currency/Foreign currency)", min_value=1e-9, value=1.05, step=0.01, format="%.6f")
T = int(st.sidebar.number_input("Time horizon T (years)", min_value=1, max_value=50, value=3, step=1))
sigma_input = st.sidebar.number_input("Annual volatility σ (% per year)", min_value=0.0, value=10.0, step=0.5)
sigma = sigma_input / 100.0

# NEW: Inflation differential input
infl_diff_pct = st.sidebar.number_input("Inflation differential (DOM − FOR, % per year)", value=0.0, step=0.25, format="%.4f")
infl_diff = infl_diff_pct / 100.0

# Rates, hedge, trading frictions
r_d_pct = st.sidebar.number_input("Domestic interest rate r_d (% per year)", value=3.0, step=0.25, format="%.4f")
r_f_pct = st.sidebar.number_input("Foreign interest rate r_f  (% per year)", value=5.0, step=0.25, format="%.4f")
spread_bps = st.sidebar.number_input("Forward bid-ask spread (basis points)", min_value=0.0, value=25.0, step=1.0)
hedge_frac_pct = st.sidebar.number_input("Hedge fraction of revenue h (% of each year)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, format="%.1f")

# Simulation controls
n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100))
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))

# Convert to decimals
r_d = r_d_pct / 100.0
r_f = r_f_pct / 100.0
hedge_frac = hedge_frac_pct / 100.0

# Validate rates & inflation differential
if (1.0 + r_d) <= 0.0 or (1.0 + r_f) <= 0.0:
    st.error("Rates must be greater than -100%. Please adjust r_d and r_f.")
    st.stop()
if (1.0 + infl_diff) <= 0.0:
    st.error("Inflation differential too negative. Please ensure 1 + (DOM − FOR) > 0.")
    st.stop()

# Discount factors (t=0..T)
DF_d_0 = make_discount_factors_constant(r_d, T=T)
DF_f_0 = make_discount_factors_constant(r_f, T=T)  # kept for completeness/extensibility

# ------------------------------
# Cash flows (years 1..T) — NO 'Year' column
# ------------------------------
st.subheader("Cash Flows")
st.caption(f"Costs in DOM, Revenues in FOR. Provide amounts for years **1–{T}** (row index shows the year).")

cash_df = pd.DataFrame({
    "Cost (DOM)": [0.0]*T,
    "Revenue (FOR)": [0.0]*T,
})
cash_df.index = pd.Index(range(1, T+1), name=f"Year (1–{T})")  # visual index only

cash_df = st.data_editor(
    cash_df,
    num_rows="fixed",
    use_container_width=True,
)

# Extract arrays
costs_dc   = cash_df["Cost (DOM)"].to_numpy(dtype=float)
revenue_fc = cash_df["Revenue (FOR)"].to_numpy(dtype=float)

# Tabs
tabs = st.tabs(["Compare Constant-Fraction Strategies"])

with tabs[0]:
    st.markdown("### Constant Hedge Fraction (h) — Strategy Comparison (DC/FC)")
    st.caption(
        "- **Spot path**: lognormal with drift tied to inflation differential πΔ = (DOM − FOR).  \n"
        "  Uses μ = ln(1+πΔ) − 0.5σ² so that E[S_t/S_{t-1}] = 1+πΔ (positive πΔ ⇒ DOM depreciation).  \n"
        "- **Unhedged (h=0)**: 100% converts at spot S_t (DC/FC).  \n"
        "- **Hedge-all-at-0**: for each year t, hedge `h × revenue_t` at t=0 using "
        "F₀→t = S₀ × ((1+r_f)^t / (1+r_d)^t); the remaining (1−h) converts at spot S_t.  \n"
        "- **Roll 1-Year**: each year t−1, hedge `h × revenue_t` for year t using "
        "F_{t-1→t} = S_{t-1} × (1+r_f)/(1+r_d); the remaining (1−h) converts at spot S_t."
    )

    colA, colB = st.columns([1,1])
    with colA:
        simulate_btn = st.button("Simulate")
    with colB:
        hsweep_btn = st.button("Plot σ vs Mean (H-sweep)")

    if simulate_btn:
        S_paths = simulate_spot_paths_dc_fc_with_infl_drift(
            S0_dc_fc=S0,
            sigma=sigma,
            infl_diff=infl_diff,
            n_sims=n_sims,
            T=T,
            seed=seed
        )

        # Strategy A: Hedge-all-at-0
        res_A = compute_strategy_results_constant_hedge(
            S_paths_dc_fc=S_paths, S0_dc_fc=S0,
            DF_d_0=DF_d_0, DF_f_0=DF_f_0,
            r_d=r_d, r_f=r_f,
            costs_dc=costs_dc, revenue_fc=revenue_fc,
            spread_bps=spread_bps,
            hedge_frac=hedge_frac,
            strategy="all_at_t0",
        )

        # Strategy B: Roll 1-Year
        res_B = compute_strategy_results_constant_hedge(
            S_paths_dc_fc=S_paths, S0_dc_fc=S0,
            DF_d_0=DF_d_0, DF_f_0=DF_f_0,
            r_d=r_d, r_f=r_f,
            costs_dc=costs_dc, revenue_fc=revenue_fc,
            spread_bps=spread_bps,
            hedge_frac=hedge_frac,
            strategy="roll_one_year",
        )

        # Unhedged baseline (h = 0)
        res_U = compute_strategy_results_constant_hedge(
            S_paths_dc_fc=S_paths, S0_dc_fc=S0,
            DF_d_0=DF_d_0, DF_f_0=DF_f_0,
            r_d=r_d, r_f=r_f,
            costs_dc=costs_dc, revenue_fc=revenue_fc,
            spread_bps=spread_bps,
            hedge_frac=0.0,
            strategy="all_at_t0",  # irrelevant when h=0
        )

        # Summary (includes loss fractions)
        summary = pd.DataFrame({
            "Strategy": ["Unhedged (h=0)", "Hedge-all-at-0", "Roll 1-Year"],
            "Hedge Fraction h": [0.0, hedge_frac, hedge_frac],
            "Avg PV Revenue (DOM)": [res_U["avg_pv_revenue"], res_A["avg_pv_revenue"], res_B["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_U["avg_pv_cost"],    res_A["avg_pv_cost"],    res_B["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_U["avg_pv_profit"],  res_A["avg_pv_profit"],  res_B["avg_pv_profit"]],
            "StdDev PV Profit":     [res_U["std_pv_profit"],  res_A["std_pv_profit"],  res_B["std_pv_profit"]],
            "Frac(PV Profit < 0)":  [res_U["frac_neg_profit"], res_A["frac_neg_profit"], res_B["frac_neg_profit"]],
        })

        fmt = summary.copy()
        fmt["Hedge Fraction h"]      = (fmt["Hedge Fraction h"]*100.0).map(lambda x: f"{x:.1f}%")
        for col in ["Avg PV Revenue (DOM)", "Avg PV Cost (DOM)", "Avg PV Profit (DOM)", "StdDev PV Profit"]:
            fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}")
        fmt["Frac(PV Profit < 0)"]   = (fmt["Frac(PV Profit < 0)"]*100.0).map(lambda x: f"{x:.1f}%")

        st.dataframe(fmt, use_container_width=True)

        # Histograms
        st.markdown("#### PV Profit Distribution (in domestic currency)")
        for title, arr in [
            ("Unhedged (h=0): PV Profit (DOM)", res_U["pv_profit_per_sim"]),
            ("Hedge-all-at-0: PV Profit (DOM)", res_A["pv_profit_per_sim"]),
            ("Rolling 1-Year Hedge: PV Profit (DOM)",    res_B["pv_profit_per_sim"]),
        ]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                st.warning(f"No finite values to plot for '{title}'. Check inputs (rates > -100%, etc.).")
            else:
                fig = plt.figure()
                plt.hist(finite, bins=30)
                plt.title(title)
                st.pyplot(fig)

    # --- σ vs Mean plot over h-grid (respects T) ---
    if hsweep_btn:
        S_paths = simulate_spot_paths_dc_fc_with_infl_drift(
            S0_dc_fc=S0,
            sigma=sigma,
            infl_diff=infl_diff,
            n_sims=n_sims,
            T=T,
            seed=seed
        )
        hs = np.linspace(0.0, 1.0, 11)  # 0, 0.1, ..., 1.0
        strategies = [
            ("Hedge-all-at-0", "all_at_t0"),
            ("Rolling 1-Year Hedge",    "roll_one_year"),
        ]

        for label, strat in strategies:
            rows = []
            for h in hs:
                res = compute_strategy_results_constant_hedge(
                    S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                    DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                    r_d=r_d, r_f=r_f,
                    costs_dc=costs_dc, revenue_fc=revenue_fc,
                    spread_bps=spread_bps,
                    hedge_frac=h,
                    strategy=strat,
                )
                rows.append({"h": h, "mean": res["avg_pv_profit"], "std": res["std_pv_profit"]})

            frontier = pd.DataFrame(rows)

            fig = plt.figure()
            plt.scatter(frontier["std"], frontier["mean"])
            for _, r in frontier.iterrows():
                plt.annotate(f"{int(r['h']*100)}%", (r["std"], r["mean"]), textcoords="offset points", xytext=(5,3))
            plt.xlabel("σ(PV Profit)")
            plt.ylabel("Mean PV Profit")
            plt.title(f"{label}: Frontier (σ, mean) over h = 0…100% (T = {T})")
            st.pyplot(fig)

# End of file
