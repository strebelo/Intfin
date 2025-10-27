# ==========================
# hedging_schedule_app.py
# ==========================
# Streamlit app — Per-cashflow hedge fractions and booking times
# - Users choose, for each delivery year t, what fraction to hedge at each booking time b < t
# - Forward pricing at booking time b with constant rates; bid/ask spread scales with tenor (t-b)
# - Overlay PV-profit distributions: Unhedged vs Custom Schedule vs Rolling 1-Year (preset)
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
    """ CIP-style forward (mid) with constant rates rd, rf. Requires m > t. """
    if m <= t:
        raise ValueError("Forward maturity m must be greater than booking time t.")
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
    paths = np.empty((n_sims, T + 1), dtype=float)
    paths[:, 0] = float(S0_dc_fc)
    mu_adj = np.log1p(infl_diff) - 0.5 * (sigma ** 2)

    for t in range(1, T + 1):
        eps = rng.standard_normal(n_sims)
        growth = np.exp(mu_adj + sigma * eps)
        paths[:, t] = np.maximum(1e-12, paths[:, t - 1] * growth)

    return paths


# ------------------------------
# Engines
# ------------------------------
def results_unhedged(S_paths, DF_d, costs_dc, revenue_fc):
    """All FC revenue at spot on delivery."""
    n_sims, T_plus_1 = S_paths.shape
    T = T_plus_1 - 1
    dc_rev_t = np.zeros((n_sims, T), dtype=float)
    for t in range(1, T + 1):
        spot_t = np.maximum(S_paths[:, t], 1e-12)
        dc_rev_t[:, t - 1] = revenue_fc[t - 1] * spot_t
    dc_costs_t = np.tile(costs_dc[None, :], (n_sims, 1))
    pv_rev = np.sum(dc_rev_t * DF_d[1:][None, :], axis=1)
    pv_cost = np.sum(dc_costs_t * DF_d[1:][None, :], axis=1)
    pv_profit = pv_rev - pv_cost
    return _summarize(pv_rev, pv_cost, pv_profit)


def results_schedule(
    S_paths, S0, DF_d,
    r_d, r_f,
    costs_dc, revenue_fc,
    spread_bps_per_year,
    H  # triangular matrix, shape (T, T); H[b, t-1] is fraction of cashflow at year t hedged at booking time b
):
    """
    Price a custom hedge schedule:
      - For each delivery year t=1..T:
          For each booking time b in {0..t-1}, fraction H[b, t-1] of that year's FC revenue is sold forward
          at booking time b with tenor (t-b), using the **bid**.
          Remainder (1 - sum_b H[b,t-1]) is left unhedged and converted at spot S_t on delivery.
    """
    n_sims, T_plus_1 = S_paths.shape
    T = T_plus_1 - 1

    H = np.asarray(H, dtype=float)
    H = np.where(np.isfinite(H), H, 0.0)
    H = np.clip(H, 0.0, 1.0)

    dc_rev_t = np.zeros((n_sims, T), dtype=float)

    ratio = (1.0 + float(r_d)) / max(1e-12, (1.0 + float(r_f)))

    for t in range(1, T + 1):
        # Hedged tranches
        col = H[:t, t - 1]  # booking times b = 0..t-1
        col_sum = float(col.sum())
        col = np.clip(col, 0.0, 1.0)

        for b, frac in enumerate(col):
            if frac <= 0.0:
                continue
            hedged_fc = frac * revenue_fc[t - 1]

            if b == 0:
                # Book all the way from t=0 to delivery t using S0
                F_mid = forward_dc_per_fc_constant_rate(S0, r_d, r_f, 0, t)
            else:
                # Book at time b using simulated S_b pathwise; tenor = t-b
                S_b = np.maximum(S_paths[:, b], 1e-12)
                F_mid = S_b * (ratio ** (t - b))

            years = float(t - b)
            bid_b_t, _ = bid_ask_from_mid_tenor_scaled(F_mid, spread_bps_per_year, years=years)
            dc_rev_t[:, t - 1] += hedged_fc * bid_b_t  # selling FC uses bid

        # Unhedged remainder at spot on delivery
        rem_frac = max(0.0, 1.0 - col_sum)
        if rem_frac > 0.0:
            unhedged_fc = rem_frac * revenue_fc[t - 1]
            spot_t = np.maximum(S_paths[:, t], 1e-12)
            dc_rev_t[:, t - 1] += unhedged_fc * spot_t

    # Costs
    n_sims = S_paths.shape[0]
    dc_costs_t = np.tile(costs_dc[None, :], (n_sims, 1))

    pv_rev = np.sum(dc_rev_t * DF_d[1:][None, :], axis=1)
    pv_cost = np.sum(dc_costs_t * DF_d[1:][None, :], axis=1)
    pv_profit = pv_rev - pv_cost
    return _summarize(pv_rev, pv_cost, pv_profit)


def _nanstd(x):
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _fracneg(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x < 0.0))


def _summarize(pv_rev, pv_cost, pv_profit):
    out = {
        "pv_revenue_per_sim": np.where(np.isfinite(pv_rev), pv_rev, np.nan),
        "pv_cost_per_sim": np.where(np.isfinite(pv_cost), pv_cost, np.nan),
        "pv_profit_per_sim": np.where(np.isfinite(pv_profit), pv_profit, np.nan),
        "avg_pv_revenue": float(np.nanmean(pv_rev)) if np.isfinite(pv_rev).any() else float("nan"),
        "avg_pv_cost": float(np.nanmean(pv_cost)) if np.isfinite(pv_cost).any() else float("nan"),
        "avg_pv_profit": float(np.nanmean(pv_profit)) if np.isfinite(pv_profit).any() else float("nan"),
        "std_pv_profit": _nanstd(pv_profit),
        "frac_neg_profit": _fracneg(pv_profit),
    }
    return out


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Currency Hedging — Schedule by Cash Flow", layout="wide")
st.title("💱 Currency Hedging Laboratory — Custom Schedule")

with st.expander("Show math / notation"):
    st.latex(r"F_{b,t} = S_b \cdot \frac{(1+r_d)^{\,t-b}}{(1+r_f)^{\,t-b}}")
    st.latex(r"\text{For revenue (sell FC), use the bid: } \text{bid} = F_{b,t}\!\left(1 - \tfrac{\text{spread}(t-b)}{2}\right)")
    st.latex(r"\mu = \ln(1+\pi_{\Delta}) - \tfrac{1}{2}\sigma^2,\quad \mathbb{E}\!\left[\frac{S_t}{S_{t-1}}\right]=1+\pi_{\Delta}")

st.sidebar.header("Inputs")
S0 = st.sidebar.number_input("Current spot S0 (Dom./For. currency)", min_value=1e-9, value=1.17, step=0.01, format="%.6f")
T = int(st.sidebar.number_input("Time horizon T (years)", min_value=1, max_value=50, value=4, step=1))
sigma = st.sidebar.number_input("Annual FX volatility (%)", min_value=0.0, value=10.0, step=0.5)/100.0
infl_diff = st.sidebar.number_input("Expected annual change in spot exchange rate, (%/yr)", value=0.0, step=0.25, format="%.4f")/100.0
r_d = st.sidebar.number_input("Domestic interest rate (%/yr)", value=4.0, step=0.25, format="%.4f")/100.0
r_f = st.sidebar.number_input("Foreign interest rate (%/yr)", value=3.0, step=0.25, format="%.4f")/100.0
spread_bps_per_year = st.sidebar.number_input("Forward bid–ask spread (bps per year of maturity)", min_value=0.0, value=0.0, step=0.5)

st.sidebar.markdown("---")
n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1, value=5000, step=100))
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
auto_normalize = st.sidebar.checkbox("Auto-normalize each column to ≤ 100%", value=True)

if (1.0+r_d)<=0 or (1.0+r_f)<=0:
    st.error("Rates must be > -100%.")
    st.stop()
if (1.0+infl_diff)<=0:
    st.error("Inflation differential too negative: require 1 + (DOM−FOR) > 0.")
    st.stop()

DF_d = make_discount_factors_constant(r_d, T)

# ------------------------------
# Cash Flows
# ------------------------------
st.subheader("Cash Flows")
default_cols = {
    "Cost cash flow in domestic currency": [0.0] * T,
    "Cash flow in foreign currency": [0.0] * T,
}
cash_df = pd.DataFrame(default_cols, index=pd.Index(range(1, T+1), name=f"Year (1–{T})"))
cash_df = st.data_editor(cash_df, num_rows="fixed", use_container_width=True)

costs_dc = pd.to_numeric(cash_df.iloc[:, 0], errors="coerce").fillna(0.0).to_numpy(dtype=float)
revenue_fc = pd.to_numeric(cash_df.iloc[:, 1], errors="coerce").fillna(0.0).to_numpy(dtype=float)

# ------------------------------
# Hedge Schedule Matrix (triangular)
# ------------------------------
st.subheader("Hedge Schedule (fractions by booking time and delivery year)")
st.caption("Enter fractions (0–1). For each column (delivery year), entries correspond to fractions booked at that earlier time. Any remainder is unhedged at spot on delivery.")

# Create triangular DataFrame with rows b=0..T-1, cols t=1..T
rows = [f"Book at t={b}" for b in range(T)]
cols = [f"Year {t}" for t in range(1, T+1)]
H_df = pd.DataFrame(0.0, index=rows, columns=cols)

# Mask invalid cells where b >= t => not allowed
mask_invalid = np.zeros((T, T), dtype=bool)
for b in range(T):
    for t in range(1, T+1):
        if b >= t:
            mask_invalid[b, t-1] = True
            H_df.iloc[b, t-1] = np.nan

# Presets
c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1:
    if st.button("Preset: All at t=0"):
        for t in range(1, T+1):
            H_df.loc["Book at t=0", f"Year {t}"] = 1.0
            for b in range(1, t):
                H_df.iloc[b, t-1] = 0.0
with c2:
    if st.button("Preset: Rolling 1-Year"):
        # 100% booked at t=j-1 for Year j
        for t in range(1, T+1):
            for b in range(0, t-1):
                H_df.iloc[b, t-1] = 0.0
            H_df.iloc[t-1-1 if t-1>=0 else 0, t-1] = 1.0 if t-1>=0 else 1.0  # safe for t>=1
with c3:
    if st.button("Preset: 50/50 Ladder (t=0 & t=j-1)"):
        for t in range(1, T+1):
            H_df.loc["Book at t=0", f"Year {t}"] = 0.5
            for b in range(1, t-1):
                H_df.iloc[b, t-1] = 0.0
            if t-1 >= 0:
                H_df.iloc[t-1-1 if t-1>=1 else 0, t-1] = 0.5 if t>=2 else 0.5  # if t=1, this equals t=0
with c4:
    st.write("")

# Editable grid
H_user = st.data_editor(H_df, num_rows="fixed", use_container_width=True, key="H_editor")

# Convert to matrix H[b, t-1]
H = np.zeros((T, T), dtype=float)
over_cols = []
coverage_lines = []
for t in range(1, T+1):
    col_vals = []
    for b in range(T):
        v = H_user.iloc[b, t-1]
        if mask_invalid[b, t-1]:
            v = 0.0
        v = 0.0 if pd.isna(v) else float(v)
        v = max(0.0, v)
        if b < t:
            col_vals.append(v)
            H[b, t-1] = v
        else:
            H[b, t-1] = 0.0
    col_sum = float(np.sum(col_vals))
    if auto_normalize and col_sum > 1.0:
        H[:t, t-1] = H[:t, t-1] / col_sum  # normalize to exactly 1.0
        col_sum = 1.0
    if col_sum > 1.0 + 1e-9:
        over_cols.append(t)
    cov_pct = min(100.0, 100.0 * col_sum)
    coverage_lines.append(f"Year {t}: Hedged {cov_pct:.1f}% | Unhedged {max(0.0, 100.0 - cov_pct):.1f}%")

if over_cols:
    st.warning(f"Some years exceed 100% hedge even after clipping: {over_cols}. Values above 1.0 will be clipped for computation.")

st.markdown("**Coverage by year:**  \n" + " · ".join(coverage_lines))

# ------------------------------
# Buttons
# ------------------------------
c1, c2 = st.columns([1,1])
with c1:
    simulate_btn = st.button("Run Simulation")
with c2:
    overlay_btn = st.button("Plot PV Profit Distributions")

def run_paths():
    return simulate_spot_paths_dc_fc_with_infl_drift(S0, sigma, infl_diff, n_sims, T, seed)

# ------------------------------
# Actions
# ------------------------------
if simulate_btn or overlay_btn:
    S_paths = run_paths()

    # Baselines
    unhedged = results_unhedged(S_paths, DF_d, costs_dc, revenue_fc)

    # Custom schedule
    custom = results_schedule(S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year, H)

    # Rolling 1-Year preset for comparison
    H_roll = np.zeros((T, T), dtype=float)
    for t in range(1, T+1):
        # book 100% at t=j-1 (if j>=1)
        b = max(0, t-1)
        H_roll[b, t-1] = 1.0
    roll = results_schedule(S_paths, S0, DF_d, r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year, H_roll)

    summary = pd.DataFrame({
        "Strategy": ["Unhedged", "Custom Schedule", "Rolling 1-Year"],
        "Avg PV Revenue (DOM)": [unhedged["avg_pv_revenue"], custom["avg_pv_revenue"], roll["avg_pv_revenue"]],
        "Avg PV Cost (DOM)": [unhedged["avg_pv_cost"], custom["avg_pv_cost"], roll["avg_pv_cost"]],
        "Avg PV Profit (DOM)": [unhedged["avg_pv_profit"], custom["avg_pv_profit"], roll["avg_pv_profit"]],
        "StdDev PV Profit": [unhedged["std_pv_profit"], custom["std_pv_profit"], roll["std_pv_profit"]],
        "Frac(PV Profit < 0)": [unhedged["frac_neg_profit"], custom["frac_neg_profit"], roll["frac_neg_profit"]],
    })

    fmt = summary.copy()
    for col in ["Avg PV Revenue (DOM)", "Avg PV Cost (DOM)", "Avg PV Profit (DOM)", "StdDev PV Profit"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}")
    fmt["Frac(PV Profit < 0)"] = (fmt["Frac(PV Profit < 0)"]*100.0).map(lambda x: f"{x:.1f}%")

    st.markdown("### Results Summary")
    st.dataframe(fmt, use_container_width=True)

    if overlay_btn:
        st.markdown("#### PV Profit — Overlayed Distributions (Unhedged vs Custom vs Rolling 1-Year)")
        A = np.asarray(custom["pv_profit_per_sim"]); A = A[np.isfinite(A)]
        B = np.asarray(roll["pv_profit_per_sim"]);   B = B[np.isfinite(B)]
        C = np.asarray(unhedged["pv_profit_per_sim"]); C = C[np.isfinite(C)]
        if A.size and B.size and C.size:
            fig = plt.figure()
            plt.hist(C, bins=40, alpha=0.5, label="Unhedged")
            plt.hist(B, bins=40, alpha=0.5, label="Rolling 1-Year")
            plt.hist(A, bins=40, alpha=0.5, label="Custom Schedule")
            plt.xlabel("PV Profit (DOM)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.title("PV Profit: Overlayed Distributions")
            st.pyplot(fig)
        else:
            st.info("Not enough finite values to plot.")
