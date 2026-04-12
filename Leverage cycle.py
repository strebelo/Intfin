import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sequential Leverage Cycle Housing Simulator", layout="wide")


# ============================================================
# Helpers
# ============================================================

def make_base_valuations(n: int, v_high: float, v_low: float) -> np.ndarray:
    """
    Equally spaced valuations from high to low.
    Household 1 has the highest valuation.
    """
    return np.linspace(v_high, v_low, n)


def solve_market_period(vals_t: np.ndarray, equity_t: np.ndarray, H_supply: float, kappa_t: float):
    """
    Solve one period of the housing market.

    Demand rule:
        if V_i > P, household i buys as much as possible:
            h_i = kappa_t * E_i / P

    Regimes:
    1) all-buyers / liquidity regime
    2) scarcity / marginal-buyer regime

    IMPORTANT FIX:
    Debt is based on actual purchases:
        d_i = P * h_i - equity_t[i]
    so the marginal buyer is not assigned excessive debt.
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
    # First check all-buyers / liquidity regime
    # --------------------------------------------------------
    price_all = kappa_t * total_equity / H_supply

    if price_all <= vals_t[-1]:
        price = price_all
        if price > 0:
            h = kappa_t * equity_t / price
            buyers = h > 1e-12
            d[buyers] = np.maximum(price * h[buyers] - equity_t[buyers], 0.0)
        return price, h, d, n - 1, "all_buyers_liquidity"

    # --------------------------------------------------------
    # Scarcity / marginal-buyer regime
    # --------------------------------------------------------
    for m in range(n):
        price_candidate = vals_t[m]
        if price_candidate <= 0:
            continue

        demand_before_m = kappa_t * equity_t[:m].sum() / price_candidate
        demand_up_to_m = kappa_t * equity_t[:m + 1].sum() / price_candidate

        if demand_before_m <= H_supply + 1e-12 and demand_up_to_m >= H_supply - 1e-12:
            price = price_candidate

            # Inframarginal buyers buy at full capacity
            if m > 0:
                h[:m] = kappa_t * equity_t[:m] / price

            # Marginal buyer takes residual supply
            residual = H_supply - h[:m].sum()
            h[m] = max(0.0, residual)

            buyers = h > 1e-12

            # FIX: debt tied to actual purchase
            d[buyers] = np.maximum(price * h[buyers] - equity_t[buyers], 0.0)

            return price, h, d, m, "scarcity_marginal_buyer"

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------
    price = price_all
    if price > 0:
        h = kappa_t * equity_t / price
        buyers = h > 1e-12
        d[buyers] = np.maximum(price * h[buyers] - equity_t[buyers], 0.0)

    return price, h, d, n - 1, "fallback_liquidity"


def advance_one_period(
    period_number: int,
    vals_prev: np.ndarray,
    liquid_wealth_prev: np.ndarray,
    housing_prev: np.ndarray,
    debt_prev: np.ndarray,
    H_supply: float,
    kappa_t: float,
    interest_rate_t: float,
    valuation_shock_pct_t: float,
    cumulative_shocks: bool,
):
    """
    Advance the model one period.

    Timing:
    - Period 1 uses initial valuations and initial liquid wealth.
    - For later periods, first update liquid wealth from previous holdings/debt.
    - Then apply the current period valuation shock.
    - Then solve for the new equilibrium price and allocations.

    IMPORTANT FIX:
    Non-buyers keep their liquid wealth. They do NOT get wiped out.
    """

    n = len(vals_prev)

    # --------------------------------------------------------
    # Step 1: update liquid wealth available at the start of period
    # --------------------------------------------------------
    if period_number == 1:
        equity_before_purchase = liquid_wealth_prev.copy()
        residual_debt = np.zeros(n)
    else:
        raw_wealth = liquid_wealth_prev.copy()

        owners_prev = housing_prev > 1e-12
        raw_wealth[owners_prev] = (
            st.session_state.last_price * housing_prev[owners_prev]
            - (1.0 + interest_rate_t) * debt_prev[owners_prev]
        )

        nonowners_prev = ~owners_prev
        indebted_nonowners = nonowners_prev & (debt_prev > 1e-12)
        raw_wealth[indebted_nonowners] = (
            liquid_wealth_prev[indebted_nonowners]
            - (1.0 + interest_rate_t) * debt_prev[indebted_nonowners]
        )

        equity_before_purchase = np.maximum(raw_wealth, 0.0)
        residual_debt = np.maximum(-raw_wealth, 0.0)

    # --------------------------------------------------------
    # Step 2: update valuations
    # --------------------------------------------------------
    shock_decimal = valuation_shock_pct_t / 100.0

    if cumulative_shocks:
        vals_t = vals_prev * (1.0 + shock_decimal)
    else:
        base_vals = st.session_state.base_vals
        vals_t = base_vals * (1.0 + shock_decimal)

    # --------------------------------------------------------
    # Step 3: solve market for this period
    # --------------------------------------------------------
    price_t, housing_t, purchase_debt_t, marginal_idx, regime = solve_market_period(
        vals_t=vals_t,
        equity_t=equity_before_purchase,
        H_supply=H_supply,
        kappa_t=kappa_t
    )

    debt_t = residual_debt.copy()
    buyers = housing_t > 1e-12
    debt_t[buyers] = purchase_debt_t[buyers]

    # Liquid wealth carried forward:
    # non-buyers keep their cash/equity
    # buyers convert wealth into housing position
    liquid_wealth_next = equity_before_purchase.copy()
    liquid_wealth_next[buyers] = 0.0

    summary_row = {
        "period": period_number,
        "price": price_t,
        "regime": regime,
        "kappa": kappa_t,
        "interest_rate_pct": 100.0 * interest_rate_t,
        "valuation_shock_pct": valuation_shock_pct_t,
        "total_equity_before_purchase": equity_before_purchase.sum(),
        "owners": int((housing_t > 1e-12).sum()),
        "insolvent_households": int((residual_debt > 1e-12).sum()),
        "marginal_household": None if marginal_idx is None else int(marginal_idx + 1),
        "top_valuation": vals_t[0],
        "bottom_valuation": vals_t[-1],
    }

    household_rows = []
    for i in range(n):
        household_rows.append({
            "period": period_number,
            "household": i + 1,
            "valuation": vals_t[i],
            "equity_before_purchase": equity_before_purchase[i],
            "housing": housing_t[i],
            "debt_end_of_period": debt_t[i],
            "liquid_wealth_carried_forward": liquid_wealth_next[i],
            "buyer": int(buyers[i]),
        })

    return (
        vals_t,
        liquid_wealth_next,
        housing_t,
        debt_t,
        price_t,
        summary_row,
        pd.DataFrame(household_rows)
    )


# ============================================================
# Session-state initialization
# ============================================================

def initialize_simulation(n, V_high, V_low, E0):
    base_vals = make_base_valuations(n, V_high, V_low)

    st.session_state.base_vals = base_vals
    st.session_state.current_vals = base_vals.copy()
    st.session_state.current_liquid_wealth = np.full(n, E0, dtype=float)
    st.session_state.current_housing = np.zeros(n, dtype=float)
    st.session_state.current_debt = np.zeros(n, dtype=float)
    st.session_state.current_period = 1
    st.session_state.last_price = np.nan
    st.session_state.summary_history = []
    st.session_state.household_history = pd.DataFrame()


# ============================================================
# Sidebar controls
# ============================================================

st.title("Sequential Leverage Cycle Housing Simulator")

with st.sidebar:
    st.header("Global Parameters")

    n = st.number_input("Number of households (n)", min_value=2, max_value=500, value=10, step=1)
    H_supply = st.number_input("Number of houses (H)", min_value=0.1, value=10.0, step=1.0)

    V_high = st.number_input(r"Upper valuation bound ($V^h$)", min_value=0.01, value=500.0, step=10.0)
    V_low = st.number_input(r"Lower valuation bound ($V^l$)", min_value=0.01, value=50.0, step=10.0)
    E0 = st.number_input(r"Initial liquid wealth ($E_0$)", min_value=0.0, value=100.0, step=10.0)

    cumulative_shocks = st.checkbox("Make valuation shocks cumulative over time", value=True)

    st.markdown("---")
    st.subheader("Inputs for the NEXT period")

    next_kappa = st.number_input(r"Leverage multiple for next period ($\kappa_t$)", min_value=0.0, value=5.0, step=0.1)
    next_interest_rate_pct = st.number_input("Interest rate for next period (%)", value=0.0, step=0.5)
    next_valuation_shock_pct = st.number_input("Valuation shock for next period (%)", value=0.0, step=1.0)

    st.markdown("---")
    reset_clicked = st.button("Initialize / Reset Simulation", use_container_width=True)
    advance_clicked = st.button("Advance One Period", use_container_width=True)

# ============================================================
# Validation
# ============================================================

if V_low > V_high:
    st.error(r"Please make sure $V^l \leq V^h$.")
    st.stop()

if H_supply <= 0:
    st.error("Housing supply H must be positive.")
    st.stop()

# initialize if needed
if "current_period" not in st.session_state or reset_clicked:
    initialize_simulation(n, V_high, V_low, E0)

# ============================================================
# Advance one period
# ============================================================

if advance_clicked:
    (
        vals_t,
        liquid_wealth_next,
        housing_t,
        debt_t,
        price_t,
        summary_row,
        household_df_t
    ) = advance_one_period(
        period_number=st.session_state.current_period,
        vals_prev=st.session_state.current_vals,
        liquid_wealth_prev=st.session_state.current_liquid_wealth,
        housing_prev=st.session_state.current_housing,
        debt_prev=st.session_state.current_debt,
        H_supply=H_supply,
        kappa_t=next_kappa,
        interest_rate_t=next_interest_rate_pct / 100.0,
        valuation_shock_pct_t=next_valuation_shock_pct,
        cumulative_shocks=cumulative_shocks,
    )

    st.session_state.current_vals = vals_t
    st.session_state.current_liquid_wealth = liquid_wealth_next
    st.session_state.current_housing = housing_t
    st.session_state.current_debt = debt_t
    st.session_state.last_price = price_t
    st.session_state.summary_history.append(summary_row)

    if st.session_state.household_history.empty:
        st.session_state.household_history = household_df_t.copy()
    else:
        st.session_state.household_history = pd.concat(
            [st.session_state.household_history, household_df_t],
            ignore_index=True
        )

    st.session_state.current_period += 1

# ============================================================
# Display current status
# ============================================================

st.subheader("Current Status")
st.write(f"**Next period to be solved:** {st.session_state.current_period}")

if len(st.session_state.summary_history) == 0:
    st.info("No period has been solved yet. Click **Advance One Period** to compute period 1.")
else:
    latest = st.session_state.summary_history[-1]
    st.write(f"**Most recently solved period:** {latest['period']}")
    st.write(f"**Most recent price:** {latest['price']:.2f}")
    st.write(f"**Most recent regime:** {latest['regime']}")

# ============================================================
# Summary table
# ============================================================

st.subheader("Simulation Summary")

if len(st.session_state.summary_history) == 0:
    st.write("No data yet.")
else:
    summary_df = pd.DataFrame(st.session_state.summary_history)
    st.dataframe(summary_df, use_container_width=True)

# ============================================================
# Charts
# ============================================================

if len(st.session_state.summary_history) > 0:
    summary_df = pd.DataFrame(st.session_state.summary_history)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(summary_df["period"], summary_df["price"], marker="o")
        ax.set_title("Equilibrium Price Over Time")
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
        ax.set_title("Total Equity Before Purchase")
        ax.set_xlabel("Period")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(summary_df["period"], summary_df["kappa"], marker="o", label="kappa")
        ax.plot(summary_df["period"], summary_df["interest_rate_pct"], marker="o", label="interest rate (%)")
        ax.plot(summary_df["period"], summary_df["valuation_shock_pct"], marker="o", label="valuation shock (%)")
        ax.set_title("Inputs Used by Period")
        ax.set_xlabel("Period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ============================================================
# Household-level inspection
# ============================================================

st.subheader("Household-Level Results")

if st.session_state.household_history.empty:
    st.write("No household data yet.")
else:
    available_periods = sorted(st.session_state.household_history["period"].unique().tolist())
    selected_period = st.slider(
        "Select solved period to inspect",
        min_value=int(min(available_periods)),
        max_value=int(max(available_periods)),
        value=int(max(available_periods)),
        step=1
    )

    period_households = st.session_state.household_history[
        st.session_state.household_history["period"] == selected_period
    ].copy()

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

    st.subheader("Valuations vs. Price in Selected Period")
    vals_selected = period_households["valuation"].to_numpy()
    summary_df = pd.DataFrame(st.session_state.summary_history)
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

st.subheader("How This Version Works")
st.markdown(
    r"""
- **Sequential timing.** You solve one period at a time by clicking **Advance One Period**.
- **Period 1** is solved from the initial valuations and initial liquid wealth.
- For **later periods**, the app first updates household wealth from last period's holdings and debt, then applies the new shock, then computes the new price.
- **Non-buyers keep their liquid wealth.**
- **Debt is based on actual purchases**, so the marginal buyer is not assigned too much debt.
- The graph of prices grows over time only as you manually advance the simulation.

**Demand**
\[
h_t^i = \frac{\kappa_t E_t^i}{P_t}
\quad \text{if } V_t^i > P_t
\]

**Actual debt**
\[
D_t^i = P_t h_t^i - E_t^i
\]

**Wealth update for owners**
\[
W_t^i = P_{t-1} h_{t-1}^i - (1+r_t) D_{t-1}^i
\]

Non-owners who did not buy simply keep their liquid wealth unless they are carrying debt.
"""
)
