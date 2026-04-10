import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Leverage Cycle Housing Simulator", layout="wide")

# ============================================================
# Helper functions
# ============================================================

def make_base_valuations(n: int, v_high: float, v_low: float) -> np.ndarray:
    """
    Equally spaced valuations from high to low.
    Household 1 has the highest valuation.
    """
    return np.linspace(v_high, v_low, n)


def build_valuation_path(base_vals: np.ndarray, shock_pcts: np.ndarray, cumulative: bool = True) -> np.ndarray:
    """
    Construct period-by-period valuation vectors.

    If cumulative=True:
        V_t = base_vals * Π_{s<=t}(1 + shock_s)

    If cumulative=False:
        V_t = base_vals * (1 + shock_t)

    shock_pcts must be in decimal form, e.g. 0.10 for +10%.
    """
    T = len(shock_pcts)
    n = len(base_vals)
    V = np.zeros((T, n))

    if cumulative:
        scale = 1.0
        for t in range(T):
            scale *= (1.0 + shock_pcts[t])
            V[t, :] = base_vals * scale
    else:
        for t in range(T):
            V[t, :] = base_vals * (1.0 + shock_pcts[t])

    return V


def solve_market_period(vals_t: np.ndarray, equity_t: np.ndarray, H_supply: float, kappa_t: float):
    """
    Solve one period of the housing market.

    Demand rule:
        if V_i > P, household i buys as much as possible:
            h_i = kappa_t * E_i / P

    The market can be in one of two regimes:

    1) Scarcity / marginal-buyer regime:
       Some households do not buy because their valuations are below the price.
       The price is pinned down by the valuation of the marginal buyer.

    2) All-buyers / liquidity regime:
       Even the lowest-valuation household is willing to buy at the clearing price.
       Then everyone buys and price satisfies:
           P = kappa_t * sum(E_i) / H

    Returns:
        price
        housing vector h
        new debt vector d
        marginal index
        regime string
    """
    n = len(vals_t)
    h = np.zeros(n)
    d = np.zeros(n)

    total_equity = equity_t.sum()

    if H_supply <= 0:
        return 0.0, h, d, None, "invalid_supply"

    if total_equity <= 0 or kappa_t <= 0:
        return 0.0, h, d, None, "no_buyers"

    # --------------------------------------------------------
    # First check the "all buyers" / liquidity regime
    # --------------------------------------------------------
    price_all = kappa_t * total_equity / H_supply

    # If the clearing price with all buyers is below the lowest valuation,
    # then everyone is willing to buy at that price.
    if price_all <= vals_t[-1]:
        price = price_all
        h = kappa_t * equity_t / price if price > 0 else np.zeros(n)
        buyers = h > 1e-12
        d[buyers] = (kappa_t - 1.0) * equity_t[buyers]
        return price, h, d, n - 1, "all_buyers_liquidity"

    # --------------------------------------------------------
    # Otherwise we are in the scarcity / marginal-buyer regime
    # --------------------------------------------------------
    for m in range(n):
        price_candidate = vals_t[m]

        if price_candidate <= 0:
            continue

        demand_before_m = kappa_t * equity_t[:m].sum() / price_candidate
        demand_up_to_m = kappa_t * equity_t[:m + 1].sum() / price_candidate

        # Need:
        # top m households alone do not exhaust supply,
        # but top m+1 households can.
        if demand_before_m <= H_supply + 1e-12 and demand_up_to_m >= H_supply - 1e-12:
            price = price_candidate

            if m > 0:
                h[:m] = kappa_t * equity_t[:m] / price

            residual = H_supply - h[:m].sum()
            h[m] = max(0.0, residual)

            buyers = h > 1e-12
            d[buyers] = (kappa_t - 1.0) * equity_t[buyers]

            return price, h, d, m, "scarcity_marginal_buyer"

    # --------------------------------------------------------
    # Fallback:
    # If no candidate worked, use the all-buyers price anyway.
    # This can happen numerically or if valuations are all very low.
    # --------------------------------------------------------
    price = price_all
    if price > 0:
        h = kappa_t * equity_t / price
        buyers = h > 1e-12
        d[buyers] = (kappa_t - 1.0) * equity_t[buyers]

    return price, h, d, n - 1, "fallback_liquidity"


def simulate_model(
    n: int,
    H_supply: float,
    E0: float,
    V_high: float,
    V_low: float,
    period_table: pd.DataFrame,
    cumulative_shocks: bool = True
):
    """
    Simulate the model for T periods.

    period_table columns:
        period
        kappa
        interest_rate
        valuation_shock_pct
    """
    T = len(period_table)
    base_vals = make_base_valuations(n, V_high, V_low)

    shock_pcts = period_table["valuation_shock_pct"].to_numpy(dtype=float) / 100.0
    kappas = period_table["kappa"].to_numpy(dtype=float)
    rates = period_table["interest_rate"].to_numpy(dtype=float) / 100.0

    valuation_path = build_valuation_path(base_vals, shock_pcts, cumulative=cumulative_shocks)

    # State variables at the start
    equity = np.full(n, E0, dtype=float)   # exogenous in period 0
    housing_prev = np.zeros(n, dtype=float)
    debt_prev = np.zeros(n, dtype=float)
    price_prev = np.nan

    summary_rows = []
    household_rows = []

    for t in range(T):
        vals_t = valuation_path[t, :]
        kappa_t = kappas[t]
        R_t = rates[t]

        # ----------------------------------------------------
        # Update equity at the beginning of period t
        # ----------------------------------------------------
        if t == 0:
            equity_before_purchase = equity.copy()
            residual_debt = np.zeros(n)
        else:
            raw_equity = price_prev * housing_prev - (1.0 + R_t) * debt_prev

            equity_before_purchase = np.maximum(raw_equity, 0.0)

            # If raw equity is negative, set equity to zero and carry shortfall as debt
            residual_debt = np.maximum(-raw_equity, 0.0)

        # ----------------------------------------------------
        # Solve the market in period t
        # ----------------------------------------------------
        price_t, housing_t, new_purchase_debt_t, marginal_idx, regime = solve_market_period(
            vals_t=vals_t,
            equity_t=equity_before_purchase,
            H_supply=H_supply,
            kappa_t=kappa_t
        )

        # End-of-period debt:
        # - buyers take on fresh purchase debt
        # - insolvent non-buyers carry residual debt
        debt_t = residual_debt.copy()
        buyers = housing_t > 1e-12
        debt_t[buyers] = new_purchase_debt_t[buyers]

        owners = int((housing_t > 1e-12).sum())
        insolvent = int((residual_debt > 1e-12).sum())

        summary_rows.append({
            "period": t,
            "price": price_t,
            "regime": regime,
            "kappa": kappa_t,
            "interest_rate_pct": 100.0 * R_t,
            "valuation_shock_pct": 100.0 * shock_pcts[t],
            "total_equity_before_purchase": equity_before_purchase.sum(),
            "total_housing_demand": housing_t.sum(),
            "owners": owners,
            "insolvent_households": insolvent,
            "marginal_household": None if marginal_idx is None else int(marginal_idx + 1),
            "top_valuation": vals_t[0],
            "bottom_valuation": vals_t[-1],
        })

        for i in range(n):
            household_rows.append({
                "period": t,
                "household": i + 1,
                "valuation": vals_t[i],
                "equity_before_purchase": equity_before_purchase[i],
                "housing": housing_t[i],
                "debt_end_of_period": debt_t[i],
                "residual_debt_from_insolvency": residual_debt[i],
                "buyer": int(housing_t[i] > 1e-12),
            })

        # Advance state
        equity = equity_before_purchase.copy()
        housing_prev = housing_t.copy()
        debt_prev = debt_t.copy()
        price_prev = price_t

    summary_df = pd.DataFrame(summary_rows)
    household_df = pd.DataFrame(household_rows)

    return summary_df, household_df


# ============================================================
# Sidebar controls
# ============================================================

st.title("Leverage Cycle Housing Simulator")

with st.sidebar:
    st.header("Global Parameters")

    n = st.number_input("Number of households (n)", min_value=2, max_value=500, value=10, step=1)

    # Constraint removed here: no max_value = n-1
    H_supply = st.number_input("Number of houses (H)", min_value=0.1, value=5.0, step=1.0)

    T = st.number_input("Number of periods (T)", min_value=1, max_value=100, value=6, step=1)

    st.markdown("---")

    V_high = st.number_input(r"Upper valuation bound ($V^h$)", min_value=0.01, value=500.0, step=10.0)
    V_low = st.number_input(r"Lower valuation bound ($V^l$)", min_value=0.01, value=50.0, step=10.0)

    if V_low > V_high:
        st.warning(r"Please make sure $V^l \leq V^h$.")

    E0 = st.number_input(r"Initial equity ($E_0$)", min_value=0.0, value=100.0, step=10.0)

    default_kappa = st.number_input(r"Default leverage multiple ($\kappa$)", min_value=0.01, value=5.0, step=0.1)
    default_r = st.number_input("Default interest rate per period (%)", value=5.0, step=0.5)
    default_shock = st.number_input("Default valuation shock per period (%)", value=0.0, step=1.0)

    cumulative_shocks = st.checkbox("Make valuation shocks cumulative over time", value=True)

    st.markdown("---")
    st.subheader("Period-by-Period Inputs")

    default_period_df = pd.DataFrame({
        "period": np.arange(T),
        "kappa": np.full(T, default_kappa),
        "interest_rate": np.full(T, default_r),
        "valuation_shock_pct": np.full(T, default_shock),
    })

    period_df = st.data_editor(
        default_period_df,
        num_rows="fixed",
        use_container_width=True,
        key="period_table"
    )

# ============================================================
# Validation
# ============================================================

if V_low > V_high:
    st.stop()

if H_supply <= 0:
    st.error("Housing supply H must be positive.")
    st.stop()

# ============================================================
# Run simulation
# ============================================================

summary_df, household_df = simulate_model(
    n=n,
    H_supply=H_supply,
    E0=E0,
    V_high=V_high,
    V_low=V_low,
    period_table=period_df,
    cumulative_shocks=cumulative_shocks
)

# ============================================================
# Summary table
# ============================================================

st.subheader("Simulation Summary")
st.dataframe(summary_df, use_container_width=True)

# ============================================================
# Charts
# ============================================================

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary_df["period"], summary_df["price"], marker="o")
    ax.set_title("Equilibrium Price by Period")
    ax.set_xlabel("Period")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary_df["period"], summary_df["owners"], marker="o", label="Owners")
    ax.plot(summary_df["period"], summary_df["insolvent_households"], marker="o", label="Insolvent")
    ax.set_title("Owners and Insolvent Households")
    ax.set_xlabel("Period")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary_df["period"], summary_df["total_equity_before_purchase"], marker="o")
    ax.set_title("Total Equity Before Purchases")
    ax.set_xlabel("Period")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary_df["period"], summary_df["kappa"], marker="o", label="kappa")
    ax.plot(summary_df["period"], summary_df["interest_rate_pct"], marker="o", label="interest rate (%)")
    ax.plot(summary_df["period"], summary_df["valuation_shock_pct"], marker="o", label="valuation shock (%)")
    ax.set_title("Period Inputs")
    ax.set_xlabel("Period")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ============================================================
# Regime display
# ============================================================

st.subheader("Pricing Regime by Period")
regime_counts = summary_df["regime"].value_counts().rename_axis("regime").reset_index(name="count")
st.dataframe(regime_counts, use_container_width=True)

# ============================================================
# Household-level inspection
# ============================================================

st.subheader("Household-Level Results")
selected_period = st.slider("Select period to inspect", min_value=0, max_value=int(T - 1), value=0, step=1)

period_households = household_df[household_df["period"] == selected_period].copy()
st.dataframe(period_households, use_container_width=True)

col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(period_households["household"], period_households["housing"])
    ax.set_title(f"Housing Holdings in Period {selected_period}")
    ax.set_xlabel("Household")
    ax.set_ylabel("Housing units")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

with col6:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(period_households["household"], period_households["debt_end_of_period"])
    ax.set_title(f"Debt at End of Period {selected_period}")
    ax.set_xlabel("Household")
    ax.set_ylabel("Debt")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

# ============================================================
# Extra diagnostic chart
# ============================================================

st.subheader("Valuations vs. Price in Selected Period")
vals_selected = period_households["valuation"].to_numpy()
price_selected = summary_df.loc[summary_df["period"] == selected_period, "price"].iloc[0]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(period_households["household"], vals_selected, marker="o", label="Valuation")
ax.axhline(price_selected, linestyle="--", label="Price")
ax.set_title(f"Valuations and Price in Period {selected_period}")
ax.set_xlabel("Household")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ============================================================
# Notes
# ============================================================

st.subheader("Notes on the Implementation")
st.markdown(
    r"""
**Demand.** If household \(i\) has valuation \(V_t^i > P_t\), it buys as much housing as its leverage constraint allows:
\[
H_t^i = \frac{\kappa_t E_t^i}{P_t}.
\]

**Debt.** Buyers take on debt equal to:
\[
D_t^i = (\kappa_t - 1) E_t^i.
\]

**Equity update.** For \(t \ge 1\), equity before new purchases is:
\[
E_t^i = \max \left\{0,\; P_{t-1} H_{t-1}^i - (1+R_t) D_{t-1}^i \right\}.
\]
If this expression is negative, the shortfall is carried as residual debt.

**Price regimes.**

1. **Scarcity / marginal-buyer regime:**  
   The price equals the valuation of the marginal buyer. Higher-valuation households buy as much as they can, and the marginal household absorbs the residual supply.

2. **All-buyers / liquidity regime:**  
   Everyone is willing to buy at the clearing price, and the price satisfies:
   \[
   P_t = \frac{\kappa_t \sum_i E_t^i}{H}.
   \]

This second regime becomes more likely when housing supply \(H\) is large relative to the number of households.
"""
)
