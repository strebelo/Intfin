# ------------------------------
# Currency Risk Hedging Simulator (Streamlit) — DC/FC Quoting
# Constant or Per-Period Hedge Fractions
# + User-selectable time horizon T
# + σ vs mean (H-sweep) plot (constant-h only) [combined figure]
# + Inflation differential-driven drift in spot simulation
# + Unified "Inputs" section (no separate "Simulation Controls")
# + Per-year hedge schedule UI with presets
# + LaTeX-rendered math for all formulas & symbols (via st.latex / markdown)
# + NEW: final combined chart comparing Hedge-all-at-0 vs Rolling 1-Year in one plot
# ------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Helper functions
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
    if m <= t:
        raise ValueError("Forward maturity m must be greater than t.")
    horiz = m - t
    num = (1.0 + float(r_d)) ** horiz
    den = (1.0 + float(r_f)) ** horiz
    if den <= 0:
        den = 1e-12
    return S_t_dc_fc * (num / den)

def dc_per_fc_bid_ask_from_mid(mid_dc_fc, spread_bps):
    s = max(0.0, float(spread_bps)) / 10000.0
    bid = mid_dc_fc * (1.0 - s / 2.0)
    ask = mid_dc_fc * (1.0 + s / 2.0)
    return bid, ask

def simulate_spot_paths_dc_fc_with_infl_drift(S0_dc_fc, sigma, infl_diff, n_sims, T, seed=123):
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    if (1.0 + infl_diff) <= 0.0:
        raise ValueError("Inflation differential too negative (1 + πΔ must be > 0).")

    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, T+1), dtype=float)
    paths[:, 0] = float(S0_dc_fc)

    # mu = ln(1 + pi_Delta) - 0.5 * sigma^2
    mu_adj = np.log1p(infl_diff) - 0.5 * (sigma ** 2)

    for t in range(1, T+1):
        eps = rng.standard_normal(n_sims)
        growth = np.exp(mu_adj + sigma * eps)
        paths[:, t] = np.maximum(1e-12, paths[:, t-1] * growth)

    return paths

# ---- Existing constant-h engine ----
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
    n_sims, T_plus_1 = S_paths_dc_fc.shape
    T = T_plus_1 - 1
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

            dc_from_forward  = hedged_fc * bid_dc_fc
            S_t_dc_fc        = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged = unhedged_fc * S_t_dc_fc

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    elif strategy == "roll_one_year":
        ratio = (1.0 + float(r_d)) / max(1e-12, (1.0 + float(r_f)))
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

# ---- Per-period hedge schedule engine ----
def compute_strategy_results_variable_hedge(
    S_paths_dc_fc,
    S0_dc_fc,
    DF_d_0, DF_f_0,
    r_d, r_f,
    costs_dc, revenue_fc,
    spread_bps,
    hedge_fracs_by_year,   # array-like length T in decimals [0..1]
    strategy="all_at_t0",
):
    n_sims, T_plus_1 = S_paths_dc_fc.shape
    T = T_plus_1 - 1
    DF_d = DF_d_0

    h_vec = np.asarray(hedge_fracs_by_year, dtype=float)
    if h_vec.shape[0] != T:
        raise ValueError("hedge_fracs_by_year must have length T.")
    h_vec = np.clip(h_vec, 0.0, 1.0)

    dc_revenue_t = np.zeros((n_sims, T), dtype=float)
    dc_costs_t   = np.zeros((n_sims, T), dtype=float)

    if strategy == "all_at_t0":
        for t in range(1, T+1):
            ht = h_vec[t-1]
            F_dc_fc_0t = forward_dc_per_fc_constant_rate(S0_dc_fc, r_d, r_f, 0, t)
            bid_dc_fc, _ = dc_per_fc_bid_ask_from_mid(F_dc_fc_0t, spread_bps)

            hedged_fc   = ht * revenue_fc[t-1]
            unhedged_fc = (1.0 - ht) * revenue_fc[t-1]

            dc_from_forward  = hedged_fc * bid_dc_fc
            S_t_dc_fc        = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged = unhedged_fc * S_t_dc_fc

            dc_revenue_t[:, t-1] = dc_from_forward + dc_from_unhedged

    elif strategy == "roll_one_year":
        ratio = (1.0 + float(r_d)) / max(1e-12, (1.0 + float(r_f)))
        for t in range(1, T+1):
            ht = h_vec[t-1]

            hedged_fc   = ht * revenue_fc[t-1]
            unhedged_fc = (1.0 - ht) * revenue_fc[t-1]

            S_prev_dc_fc       = S_paths_dc_fc[:, t-1]
            F_dc_fc_prev_t_mid = S_prev_dc_fc * ratio
            bid_dc_fc, _       = dc_per_fc_bid_ask_from_mid(F_dc_fc_prev_t_mid, spread_bps)

            dc_from_forward  = hedged_fc * bid_dc_fc
            S_t_dc_fc        = np.maximum(S_paths_dc_fc[:, t], 1e-12)
            dc_from_unhedged = unhedged_fc * S_t_dc_fc

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

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Currency Risk Hedging Simulator", layout="wide")
st.title("💱 Currency Risk Hedging Simulator")

# Math summary (LaTeX)
with st.expander("Show math / notation"):
    st.latex(r"S_t \ \text{(DC/FC)}")
    st.latex(r"F_{t\to m} \;=\; S_t \times \frac{(1+r_d)^{\,m-t}}{(1+r_f)^{\,m-t}}")
    st.latex(r"\mu \;=\; \ln(1+\pi_{\Delta}) \;-\; \tfrac{1}{2}\sigma^2 \quad\Rightarrow\quad \mathbb{E}\!\left[\frac{S_t}{S_{t-1}}\right] = 1+\pi_{\Delta}")
    st.latex(r"h_t \in [0,1], \quad \text{hedged FC flow at } t = h_t \times \text{revenue}_t")
    st.latex(r"\text{PV}(\text{profit}) \;=\; \sum_{t=1}^{T} \Big( \text{revenue}_t^{\text{(DC)}} - \text{cost}_t^{\text{(DC)}} \Big)\times \text{DF}_d(t)")

# Unified Inputs (sidebar)
st.sidebar.header("Inputs (symbols rendered below)")

# Core market & model inputs (widget labels avoid LaTeX; we render formulas separately)
S0 = st.sidebar.number_input("Current spot S0 (DC/FC)", min_value=1e-9, value=1.05, step=0.01, format="%.6f")
T = int(st.sidebar.number_input("Time horizon T (years)", min_value=1, max_value=50, value=3, step=1))
sigma_input = st.sidebar.number_input("Annual volatility sigma (%/yr)", min_value=0.0, value=10.0, step=0.5)
sigma = sigma_input / 100.0

# Inflation differential input (DOM − FOR)
infl_diff_pct = st.sidebar.number_input("Inflation diff (DOM − FOR, %/yr)", value=0.0, step=0.25, format="%.4f")
infl_diff = infl_diff_pct / 100.0

# Rates, hedge, trading frictions
r_d_pct = st.sidebar.number_input("Domestic rate rd (%/yr)", value=3.0, step=0.25, format="%.4f")
r_f_pct = st.sidebar.number_input("Foreign rate rf (%/yr)", value=5.0, step=0.25, format="%.4f")
spread_bps = st.sidebar.number_input("Forward bid-ask spread (bps)", min_value=0.0, value=25.0, step=1.0)

# Render the key math near inputs
st.markdown(
    r"""
**Key:**
- Spot quoted DC/FC: $S_0$
- Volatility: $\sigma$
- Rates: $r_d, r_f$
- Inflation differential (DOM−FOR): $\pi_\Delta$
- Hedge fraction (per year): $h_t$
- Forward: $F_{t\to m} = S_t \frac{(1+r_d)^{m-t}}{(1+r_f)^{m-t}}$
"""
)

# Hedge mode selector
st.sidebar.markdown("---")
hedge_mode = st.sidebar.radio(
    "Hedge mode",
    ["Constant fraction h", "Per-year schedule h_t"],
    index=0,
    help="Choose a single hedge fraction for all years, or specify a different fraction per year."
)

# Constant-h control
hedge_frac_pct = st.sidebar.number_input(
    "Hedge fraction of revenue h (% of each year, for constant-h mode)",
    min_value=0.0, max_value=100.0, value=50.0, step=1.0, format="%.1f"
)
hedge_frac = hedge_frac_pct / 100.0

# Simulation controls
n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100))
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))

# Convert to decimals
r_d = r_d_pct / 100.0
r_f = r_f_pct / 100.0

# Validate rates & inflation differential
if (1.0 + r_d) <= 0.0 or (1.0 + r_f) <= 0.0:
    st.error("Rates must be greater than -100%. Please adjust r_d and r_f.")
    st.stop()
if (1.0 + infl_diff) <= 0.0:
    st.error("Inflation differential too negative. Please ensure 1 + (Domestic − Foreign) > 0.")
    st.stop()

# Discount factors (t=0..T)
DF_d_0 = make_discount_factors_constant(r_d, T=T)
DF_f_0 = make_discount_factors_constant(r_f, T=T)  # for completeness/extensibility

# ------------------------------
# Diagnostics (CIP forward points; LaTeX-rendered)
# ------------------------------
with st.expander("Diagnostics: Forward points (DC/FC)"):
    try:
        F_01_mid = forward_dc_per_fc_constant_rate(S0, r_d, r_f, 0, 1)
        pts = F_01_mid - S0
        st.latex(rf"S_0 = {S0:.6f} \quad;\quad F_{{0\to 1}} = {F_01_mid:.6f} \quad;\quad (F - S) = {pts:.6f}")
        st.markdown("Expectation: if $r_f > r_d$, then typically $F < S$.")
    except Exception as e:
        st.write(f"Diagnostics error: {e}")

# ------------------------------
# Cash flows (years 1..T)
# ------------------------------
st.subheader("Cash Flows")
st.caption(f"Costs in domestic currency, Revenues in foreign currency. Provide amounts for years **1–{T}** (row index shows the year).")

cash_df = pd.DataFrame({
    "Cost (DOM)": [0.0]*T,
    "Revenue (FOR)": [0.0]*T,
})
cash_df.index = pd.Index(range(1, T+1), name=f"Year (1–{T})")

cash_df = st.data_editor(
    cash_df,
    num_rows="fixed",
    use_container_width=True,
)

# Extract arrays
costs_dc   = cash_df["Cost (DOM)"].to_numpy(dtype=float)
revenue_fc = cash_df["Revenue (FOR)"].to_numpy(dtype=float)

# ------------------------------
# Hedge Schedule UI
# ------------------------------
per_year_df = None
hedge_vec = None

if hedge_mode == "Per-year schedule h_t":
    st.subheader("Per-Year Hedge Schedule $h_t$")
    st.caption("Set $h_t$ (% of revenue) for each year. Presets can prefill the schedule; fine-tune afterwards.")

    # default: fill with current constant-h
    default_ht = [hedge_frac_pct]*T

    # Presets
    presets = {
        "Flat (use constant h)": lambda T, h: [h]*T,
        "Front-loaded (100%, 75%, 50%, …)": lambda T, h: [max(0, 100 - 25*(t-1)) for t in range(1, T+1)],
        "Back-loaded (…, 50%, 75%, 100%)": lambda T, h: [min(100, 25*(t)) if t>0 else 0 for t in range(T)],
        "Ladder up (0% → 100%)": lambda T, h: [round(100*t/(T-1)) if T>1 else 100 for t in range(T)],
        "Barbell (high at ends)": lambda T, h: [80 if (t in [0, T-1]) else 20 for t in range(T)],
        "Unhedged (all 0%)": lambda T, h: [0]*T,
        "Fully hedged (all 100%)": lambda T, h: [100]*T,
    }
    sel = st.selectbox("Preset", list(presets.keys()), index=0)
    apply_preset = st.button("Apply preset")
    if apply_preset:
        default_ht = presets[sel](T, hedge_frac_pct)

    per_year_df = pd.DataFrame({"Hedge h_t (%)": default_ht})
    per_year_df.index = pd.Index(range(1, T+1), name=f"Year (1–{T})")

    per_year_df = st.data_editor(
        per_year_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Hedge h_t (%)": st.column_config.NumberColumn(
                "Hedge h_t (%)",
                help="Fraction of that year's revenue hedged via forwards.",
                min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
            ),
        },
    )
    hedge_vec = (per_year_df["Hedge h_t (%)"].to_numpy(dtype=float) / 100.0)

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Compare Hedging Strategies"])

with tabs[0]:
    st.markdown("### Strategy Comparison (DC/FC)")

    st.markdown(
        r"""
- **Spot path**: lognormal with drift tied to inflation differential $\pi_{\Delta} = (\text{DOM} - \text{FOR})$, using  
  $\mu = \ln(1+\pi_{\Delta}) - \tfrac{1}{2}\sigma^2$ so that $\mathbb{E}\!\left[\tfrac{S_t}{S_{t-1}}\right] = 1 + \pi_{\Delta}$.  

- **Unhedged**: $100\%$ converts at spot $S_t$.  

- **Hedge-all-at-0**: for each year $t$, hedge $h_t \times \text{revenue}_t$ at $t=0$ using  
  $F_{0\rightarrow t} = S_0 \times \dfrac{(1+r_d)^t}{(1+r_f)^t}$; the remaining $(1-h_t)$ converts at $S_t$.  

- **Rolling 1-Year**: at year $t-1$, hedge $h_t \times \text{revenue}_t$ using  
  $F_{t-1\rightarrow t} = S_{t-1} \times \dfrac{1+r_d}{1+r_f}$; the remaining $(1-h_t)$ converts at $S_t$.
"""
    )

    colA, colB = st.columns([1,1])
    with colA:
        simulate_btn = st.button("Simulate")
    with colB:
        hsweep_btn = st.button("Plot σ vs Mean (H-sweep, constant h)")

    if simulate_btn:
        S_paths = simulate_spot_paths_dc_fc_with_infl_drift(
            S0_dc_fc=S0,
            sigma=sigma,
            infl_diff=infl_diff,
            n_sims=n_sims,
            T=T,
            seed=seed
        )

        if hedge_mode == "Per-year schedule h_t":
            # Variable h_t path
            res_A = compute_strategy_results_variable_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_fracs_by_year=hedge_vec,
                strategy="all_at_t0",
            )
            res_B = compute_strategy_results_variable_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_fracs_by_year=hedge_vec,
                strategy="roll_one_year",
            )
            # Unhedged baseline
            res_U = compute_strategy_results_variable_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_fracs_by_year=np.zeros(T, dtype=float),
                strategy="all_at_t0",
            )
            h_display = "variable"
        else:
            # Constant-h path
            res_A = compute_strategy_results_constant_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_frac=hedge_frac,
                strategy="all_at_t0",
            )
            res_B = compute_strategy_results_constant_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_frac=hedge_frac,
                strategy="roll_one_year",
            )
            res_U = compute_strategy_results_constant_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_frac=0.0,
                strategy="all_at_t0",
            )
            h_display = f"{hedge_frac*100:.1f}% (constant)"

        # Summary table
        summary = pd.DataFrame({
            "Strategy": ["Unhedged", "Hedge-all-at-0", "Rolling 1-Year Hedge"],
            "Hedge h": [h_display, h_display, h_display],
            "Avg PV Revenue (DOM)": [res_U["avg_pv_revenue"], res_A["avg_pv_revenue"], res_B["avg_pv_revenue"]],
            "Avg PV Cost (DOM)":    [res_U["avg_pv_cost"],    res_A["avg_pv_cost"],    res_B["avg_pv_cost"]],
            "Avg PV Profit (DOM)":  [res_U["avg_pv_profit"],  res_A["avg_pv_profit"],  res_B["avg_pv_profit"]],
            "StdDev PV Profit":     [res_U["std_pv_profit"],  res_A["std_pv_profit"],  res_B["std_pv_profit"]],
            "Frac(PV Profit < 0)":  [res_U["frac_neg_profit"], res_A["frac_neg_profit"], res_B["frac_neg_profit"]],
        })

        fmt = summary.copy()
        for col in ["Avg PV Revenue (DOM)", "Avg PV Cost (DOM)", "Avg PV Profit (DOM)", "StdDev PV Profit"]:
            fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}")
        fmt["Frac(PV Profit < 0)"] = (fmt["Frac(PV Profit < 0)"]*100.0).map(lambda x: f"{x:.1f}%")

        st.dataframe(fmt, use_container_width=True)

        # Histograms (separate)
        st.markdown("#### PV Profit Distribution (in domestic currency)")
        for title, arr in [
            ("Unhedged: PV Profit (DOM)", res_U["pv_profit_per_sim"]),
            ("Hedge-all-at-0: PV Profit (DOM)", res_A["pv_profit_per_sim"]),
            ("Rolling 1-Year Hedge: PV Profit (DOM)", res_B["pv_profit_per_sim"]),
        ]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                st.warning(f"No finite values to plot for '{title}'. Check inputs (rates > -100%, etc.).")
            else:
                fig = plt.figure()
                plt.hist(finite, bins=30)
                plt.title(title)
                st.pyplot(fig)

        # --- NEW: Final combined chart comparing Hedge-all-at-0 vs Rolling 1-Year ---
        st.markdown("#### Combined Comparison: Hedge-all-at-0 vs Rolling 1-Year (PV Profit)")
        finite_A = res_A["pv_profit_per_sim"][np.isfinite(res_A["pv_profit_per_sim"])]
        finite_B = res_B["pv_profit_per_sim"][np.isfinite(res_B["pv_profit_per_sim"])]
        if finite_A.size and finite_B.size:
            fig = plt.figure()
            plt.hist(finite_A, bins=40, alpha=0.5, label="Hedge-all-at-0")
            plt.hist(finite_B, bins=40, alpha=0.5, label="Rolling 1-Year")
            plt.xlabel("PV Profit (DOM)")
            plt.ylabel("Frequency")
            plt.title("PV Profit: Overlayed Distributions")
            plt.legend()
            st.pyplot(fig)
        else:
            st.info("Not enough finite values to build the combined comparison plot.")

    # --- σ vs Mean plot over h-grid (constant-h only) — COMBINED FIGURE ---
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

        rows_A, rows_B = [], []
        for h in hs:
            resA = compute_strategy_results_constant_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_frac=h,
                strategy="all_at_t0",
            )
            rows_A.append({"h": h, "mean": resA["avg_pv_profit"], "std": resA["std_pv_profit"]})

            resB = compute_strategy_results_constant_hedge(
                S_paths_dc_fc=S_paths, S0_dc_fc=S0,
                DF_d_0=DF_d_0, DF_f_0=DF_f_0,
                r_d=r_d, r_f=r_f,
                costs_dc=costs_dc, revenue_fc=revenue_fc,
                spread_bps=spread_bps,
                hedge_frac=h,
                strategy="roll_one_year",
            )
            rows_B.append({"h": h, "mean": resB["avg_pv_profit"], "std": resB["std_pv_profit"]})

        frontier_A = pd.DataFrame(rows_A)
        frontier_B = pd.DataFrame(rows_B)

        fig = plt.figure()
        plt.scatter(frontier_A["std"], frontier_A["mean"], label="Hedge-all-at-0")
        for _, r in frontier_A.iterrows():
            plt.annotate(f"{int(r['h']*100)}%", (r["std"], r["mean"]), textcoords="offset points", xytext=(5,3), fontsize=8)

        plt.scatter(frontier_B["std"], frontier_B["mean"], label="Rolling 1-Year")
        for _, r in frontier_B.iterrows():
            plt.annotate(f"{int(r['h']*100)}%", (r["std"], r["mean"]), textcoords="offset points", xytext=(5,3), fontsize=8)

        plt.xlabel(r"$\sigma(\text{PV Profit})$")
        plt.ylabel(r"$\mathbb{E}[\text{PV Profit}]$")
        plt.title("Frontier over $h \in [0,1]$ (both strategies)")
        plt.legend()
        st.pyplot(fig)

# End of file
