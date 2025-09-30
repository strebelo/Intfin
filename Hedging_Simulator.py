# ------------------------------
# Currency Risk Hedging Simulator (Streamlit) — DC/FC Quoting
# Constant Hedge Fraction with H-Sweep Frontiers and Tail Risk
# ------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Helper functions
# ------------------------------

def make_discount_factors_constant(r: float, T: int = 10):
    DF = np.ones(T + 1, dtype=float)
    base = 1.0 + float(r)
    if base <= 0:
        base = 1e-9
    for t in range(1, T + 1):
        DF[t] = 1.0 / (base ** t)
    return DF

def forward_dc_per_fc_constant_rate(S_t_dc_fc, r_d, r_f, t, m):
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    horiz = m - t
    num = (1.0 + float(r_f)) ** horiz
    den = (1.0 + float(r_d)) ** horiz
    if den <= 0:
        den = 1e-12
    return S_t_dc_fc * (num / den)

def dc_per_fc_bid_ask_from_mid(mid_dc_fc, spread_bps):
    s = max(0.0, float(spread_bps)) / 10000.0
    bid = mid_dc_fc * (1.0 - s / 2.0)
    ask = mid_dc_fc * (1.0 + s / 2.0)
    return bid, ask

def simulate_spot_paths_dc_fc(S0_dc_fc, sigma, n_sims, T=10, seed=123):
    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, T+1), dtype=float)
    paths[:, 0] = S0_dc_fc
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    for t in range(1, T+1):
        eps = rng.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(sigma * eps)
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
    n_sims = S_paths_dc_fc.shape[0]
    T = 10
    DF_d = DF_d_0

    dc_revenue_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)

    h = float(hedge_frac)
    h = min(max(h, 0.0), 1.0)

    if strategy == "all_at_t0":
        for t in range(1, T+1):
            F_dc_fc_0t = forward_dc_per_fc_constant_rate(S0_dc_fc, r_d, r_f, 0, t)
            bid_dc_fc, _ = dc_per_fc_bid_ask_from_mid(F_dc_fc_0t, spread_bps)

            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            dc_from_forward   = hedged_fc * bid_dc_fc
            S_t_dc_fc         = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged  = unhedged_fc * S_t_dc_fc

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    elif strategy == "roll_one_year":
        ratio = (1.0 + float(r_f)) / max(1e-12, (1.0 + float(r_d)))
        for t in range(1, T+1):
            hedged_fc   = h * revenue_fc[t-1]
            unhedged_fc = (1.0 - h) * revenue_fc[t-1]

            S_prev_dc_fc       = S_paths_dc_fc[:, t-1]
            F_dc_fc_prev_t_mid = S_prev_dc_fc * ratio
            bid_dc_fc, _       = dc_per_fc_bid_ask_from_mid(F_dc_fc_prev_t_mid, spread_bps)

            dc_from_forward   = hedged_fc * bid_dc_fc
            S_t_dc_fc         = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged  = unhedged_fc * S_t_dc_fc

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    else:
        raise ValueError("Unknown strategy option. Use 'all_at_t0' or 'roll_one_year'.")

    for t in range(1, T+1):
        dc_costs_t[:, t-1] = costs_dc[t-1]

    pv_revenue_per_sim = np.sum(dc_revenue_t * DF_d[1:][None, :], axis=1)
    pv_cost_per_sim    = np.sum(dc_costs_t   * DF_d[1:][None, :], axis=1)
    pv_profit_per_sim  = pv_revenue_per_sim - pv_cost_per_sim

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

# --- Risk metrics for frontiers ---
def var_es(arr, alpha=0.95):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan, np.nan
    q = np.nanpercentile(finite, (1-alpha)*100.0)  # left-tail quantile (e.g., 5th pct)
    tail = finite[finite <= q]
    VaR = -q
    ES  = -np.nanmean(tail) if tail.size else np.nan
    return VaR, ES

def sharpe_like(mean, std):
    return mean / std if (std and np.isfinite(std) and std > 0) else np.nan

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Currency Risk Hedging Simulator (DC/FC)", layout="wide")
st.title("💱 Currency Risk Hedging Simulator (DC/FC)")

st.write(
    "All FX rates are **DOMESTIC per 1 FOREIGN (DC/FC)**. "
    "**To convert foreign amount (FC) to domestic (DC), multiply:** DC = FC × (DC/FC)."
)

# Sidebar controls
st.sidebar.header("Simulation Controls")
S0 = st.sidebar.number_input("Current spot S₀ (DC per 1 FC)", min_value=1e-9, value=1.05, step=0.01, format="%.6f")
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

# Discount factors (t=0..10)
DF_d_0 = make_discount_factors_constant(r_d, T=10)
DF_f_0 = make_discount_factors_constant(r_f, T=10)

# ------------------------------
# Cash flows (years 1–10 only)
# ------------------------------
st.subheader("Cash Flows")
st.caption("Costs in DOM, Revenues in FOR. Provide amounts for years **1–10**.")

years = list(range(1, 11))
cash_df = pd.DataFrame({
    "Year": years,
    "Cost (DOM)": [0.0]*10,
    "Revenue (FOR)": [0.0]*10,
})
cash_df = st.data_editor(cash_df, num_rows="fixed", use_container_width=True)

costs_dc   = cash_df["Cost (DOM)"].to_numpy(dtype=float)
revenue_fc = cash_df["Revenue (FOR)"].to_numpy(dtype=float)

# Tabs
tabs = st.tabs(["Compare Constant-Fraction Strategies", "H-sweep (frontiers)"])

with tabs[0]:
    st.markdown("### Constant Hedge Fraction (h) — Strategy Comparison (DC/FC)")
    st.caption(
        "- **Unhedged (h=0)**: 100% converts at spot S_t (DC/FC).  \n"
        "- **Hedge-all-at-0**: for each year t, hedge `h × revenue_t` at t=0 using "
        "F₀→t = S₀ × ((1+r_f)^t / (1+r_d)^t); the remaining (1−h) converts at spot S_t.  \n"
        "- **Roll 1-Year**: each year t−1, hedge `h × revenue_t` for year t using "
        "F_{t-1→t} = S_{t-1} × (1+r_f)/(1+r_d); the remaining (1−h) converts at spot S_t."
    )

    if st.button("Simulate"):
        S_paths = simulate_spot_paths_dc_fc(S0_dc_fc=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

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
            strategy="all_at_t0",
        )

        # Summary table with tail risk
        def frac_below(arr, thr=0.0):
            finite = arr[np.isfinite(arr)]
            return float(np.mean(finite < thr)) if finite.size else np.nan

        # Unhedged reference variance for Hedge Effectiveness
        var_U = np.nanvar(res_U["pv_profit_per_sim"], ddof=1)

        def hedge_effectiveness(arr):
            v = np.nanvar(arr, ddof=1)
            return 1 - (v/var_U) if var_U > 0 else np.nan

        def enrich(res):
            VaR95, ES95 = var_es(res["pv_profit_per_sim"], alpha=0.95)
            return {
                "Avg PV Revenue (DOM)": res["avg_pv_revenue"],
                "Avg PV Cost (DOM)":    res["avg_pv_cost"],
                "Avg PV Profit (DOM)":  res["avg_pv_profit"],
                "StdDev PV Profit":     res["std_pv_profit"],
                "Frac(PV Profit < 0)":  frac_below(res["pv_profit_per_sim"], 0.0),
                "VaR95 (loss)":         VaR95,
                "CVaR95 (loss)":        ES95,
                "Sharpe":               sharpe_like(res["avg_pv_profit"], res["std_pv_profit"]),
                "HE vs Unhedged":       hedge_effectiveness(res["pv_profit_per_sim"]),
            }

        summary = pd.DataFrame.from_dict({
            "Unhedged (h=0)":   enrich(res_U),
            "Hedge-all-at-0":   enrich(res_A),
            "Roll 1-Year":      enrich(res_B),
        }, orient="index").reset_index().rename(columns={"index":"Strategy"})

        fmt = summary.copy()
        for col in ["Avg PV Revenue (DOM)", "Avg PV Cost (DOM)", "Avg PV Profit (DOM)", "StdDev PV Profit", "VaR95 (loss)", "CVaR95 (loss)"]:
            fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}")
        fmt["Frac(PV Profit < 0)"] = (summary["Frac(PV Profit < 0)"]*100.0).map(lambda x: f"{x:.1f}%")
        fmt["Sharpe"]              = summary["Sharpe"].map(lambda x: f"{x:.2f}")
        fmt["HE vs Unhedged"]      = summary["HE vs Unhedged"].map(lambda x: f"{x:.1%}" if pd.notnull(x) else "—")

        st.dataframe(fmt, use_container_width=True)

        # Histograms
        st.markdown("#### PV Profit Distribution (DC)")
        for title, arr in [
            ("Unhedged (h=0): PV Profit (DOM)", res_U["pv_profit_per_sim"]),
            ("Hedge-all-at-0: PV Profit (DOM)", res_A["pv_profit_per_sim"]),
            ("Roll 1-Year: PV Profit (DOM)",    res_B["pv_profit_per_sim"]),
        ]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                st.warning(f"No finite values to plot for '{title}'. Check inputs (rates > -100%, etc.).")
            else:
                fig = plt.figure()
                plt.hist(finite, bins=30)
                plt.title(title)
                st.pyplot(fig)

with tabs[1]:
    st.markdown("### H-sweep Frontiers (DC/FC)")
    st.caption(
        "For **h ∈ {0, 10%, …, 100%}**, we plot:\n"
        "- **(σ, mean)** of PV profit, and\n"
        "- **(CVaR₉₅, mean)** of PV profit (CVaR shown as a positive loss—lower is better).\n"
        "For each strategy, we highlight **min-CVaR h** and **max-Sharpe h**."
    )

    if st.button("Run H-sweep"):
        S_paths = simulate_spot_paths_dc_fc(S0_dc_fc=S0, sigma=sigma, n_sims=n_sims, T=10, seed=seed)

        hs = np.linspace(0.0, 1.0, 11)  # 0, 0.1, ..., 1.0
        strategies = [
            ("Hedge-all-at-0", "all_at_t0"),
            ("Roll 1-Year",    "roll_one_year"),
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
                mean = res["avg_pv_profit"]
                std  = res["std_pv_profit"]
                _, ES95 = var_es(res["pv_profit_per_sim"], alpha=0.95)
                sh = sharpe_like(mean, std)
                rows.append({"h": h, "mean": mean, "std": std, "ES95_loss": ES95, "sharpe": sh})
            frontier = pd.DataFrame(rows)

            # Identify min-CVaR (ES95) and max-Sharpe points
            idx_min_es = int(frontier["ES95_loss"].idxmin()) if frontier["ES95_loss"].notna().any() else None
            idx_max_sh = int(frontier["sharpe"].idxmax())    if frontier["sharpe"].notna().any() else None

            # --- Plot (σ, mean) ---
            fig1 = plt.figure()
            plt.scatter(frontier["std"], frontier["mean"])
            if idx_max_sh is not None:
                plt.scatter(frontier.loc[idx_max_sh, "std"], frontier.loc[idx_max_sh, "mean"], s=120, marker="*", label="Max Sharpe h")
            if idx_min_es is not None:
                plt.scatter(frontier.loc[idx_min_es, "std"], frontier.loc[idx_min_es, "mean"], s=120, marker="^", label="Min CVaR h")
            for _, r in frontier.iterrows():
                plt.annotate(f"{int(r['h']*100)}%", (r["std"], r["mean"]), textcoords="offset points", xytext=(5,3))
            plt.xlabel("σ(PV Profit)")
            plt.ylabel("Mean PV Profit")
            plt.title(f"{label}: Frontier (σ, mean) over h")
            if idx_max_sh is not None or idx_min_es is not None:
                plt.legend()
            st.pyplot(fig1)

            # --- Plot (CVaR95, mean) ---
            fig2 = plt.figure()
            plt.scatter(frontier["ES95_loss"], frontier["mean"])
            if idx_max_sh is not None:
                plt.scatter(frontier.loc[idx_max_sh, "ES95_loss"], frontier.loc[idx_max_sh, "mean"], s=120, marker="*", label="Max Sharpe h")
            if idx_min_es is not None:
                plt.scatter(frontier.loc[idx_min_es, "ES95_loss"], frontier.loc[idx_min_es, "mean"], s=120, marker="^", label="Min CVaR h")
            for _, r in frontier.iterrows():
                plt.annotate(f"{int(r['h']*100)}%", (r["ES95_loss"], r["mean"]), textcoords="offset points", xytext=(5,3))
            plt.xlabel("CVaR95 (loss)")
            plt.ylabel("Mean PV Profit")
            plt.title(f"{label}: Frontier (CVaR95, mean) over h")
            if idx_max_sh is not None or idx_min_es is not None:
                plt.legend()
            st.pyplot(fig2)

            # Show the key h values in a small table
            summary_rows = []
            if idx_min_es is not None:
                summary_rows.append({
                    "Strategy": label,
                    "Metric": "Min CVaR95 h",
                    "h": f"{int(frontier.loc[idx_min_es,'h']*100)}%",
                    "Mean": f"{frontier.loc[idx_min_es,'mean']:,.2f}",
                    "σ": f"{frontier.loc[idx_min_es,'std']:,.2f}",
                    "CVaR95": f"{frontier.loc[idx_min_es,'ES95_loss']:,.2f}",
                    "Sharpe": f"{frontier.loc[idx_min_es,'sharpe']:.2f}" if np.isfinite(frontier.loc[idx_min_es,'sharpe']) else "—",
                })
            if idx_max_sh is not None:
                summary_rows.append({
                    "Strategy": label,
                    "Metric": "Max Sharpe h",
                    "h": f"{int(frontier.loc[idx_max_sh,'h']*100)}%",
                    "Mean": f"{frontier.loc[idx_max_sh,'mean']:,.2f}",
                    "σ": f"{frontier.loc[idx_max_sh,'std']:,.2f}",
                    "CVaR95": f"{frontier.loc[idx_max_sh,'ES95_loss']:,.2f}",
                    "Sharpe": f"{frontier.loc[idx_max_sh,'sharpe']:.2f}" if np.isfinite(frontier.loc[idx_max_sh,'sharpe']) else "—",
                })
            if summary_rows:
                st.markdown("**Highlighted h values**")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# End of file
