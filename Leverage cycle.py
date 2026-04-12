import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Leverage Cycle Housing Simulator - Fixed Point Timing", layout="wide")


# ============================================================
# Helpers
# ============================================================

def make_base_valuations(n: int, v_high: float, v_low: float) -> np.ndarray:
    """
    Equally spaced valuations from high to low.
    Household 1 has the highest valuation.
    """
    return np.linspace(v_high, v_low, n)


def apply_valuation_shock(
    vals_prev: np.ndarray,
    base_vals: np.ndarray,
    shock_pct: float,
    cumulative_shocks: bool
) -> np.ndarray:
    """
    Build current-period valuations.
    """
    shock = shock_pct / 100.0
    if cumulative_shocks:
        return vals_prev * (1.0 + shock)
    return base_vals * (1.0 + shock)


def compute_equity_at_candidate_price(
    period_number: int,
    candidate_price: float,
    liquid_wealth_prev: np.ndarray,
    housing_prev: np.ndarray,
    debt_prev: np.ndarray,
    interest_rate_t: float
):
    """
    Earlier timing with fixed point:

    At the beginning of period t, households liquidate old housing at the
    CURRENT candidate price P_t, repay debt, and then use the resulting equity
    to buy again.

    For period 1:
        equity before purchase = initial liquid wealth

    For period t >= 2:
        Previous owners:
            equity_i(P) = P * h_{t-1,i} - (1+r_t) d_{t-1,i}

        Previous non-owners:
            equity_i(P) = liquid_wealth_prev_i - (1+r_t) d_{t-1,i}

    Negative equity is clipped to zero and the shortfall becomes residual debt.
    """
    n = len(liquid_wealth_prev)

    if period_number == 1:
        equity = liquid_wealth_prev.copy()
        residual_debt = np.zeros(n)
        return equity, residual_debt

    raw_equity = liquid_wealth_prev.copy()

    owners_prev = housing_prev > 1e-12
    nonowners_prev = ~owners_prev

    raw_equity[owners_prev] = (
        candidate_price * housing_prev[owners_prev]
        - (1.0 + interest_rate_t) * debt_prev[owners_prev]
    )

    raw_equity[nonowners_prev] = (
        liquid_wealth_prev[nonowners_prev]
        - (1.0 + interest_rate_t) * debt_prev[nonowners_prev]
    )

    equity = np.maximum(raw_equity, 0.0)
    residual_debt = np.maximum(-raw_equity, 0.0)
    return equity, residual_debt


def allocate_given_price(
    vals_t: np.ndarray,
    equity_t: np.ndarray,
    residual_debt: np.ndarray,
    H_supply: float,
    candidate_price: float,
    kappa_t: float
):
    """
    For a given price and equity vector, compute desired demand and actual allocation.

    Demand rule:
        if V_i > P, desired h_i = kappa * E_i / P
        otherwise 0

    Actual allocation:
        households are already ordered by valuation high -> low.
        higher valuation households get full desired demand until supply is exhausted.
        the marginal buyer gets the residual.

    Debt is based on ACTUAL purchases:
        D_i = P * h_i - E_i
    for buyers.
    Non-buyers keep residual debt from insolvency if any.
    """
    n = len(vals_t)
    h_desired = np.zeros(n)

    if candidate_price <= 0 or kappa_t <= 0:
        return h_desired, np.zeros(n), residual_debt.copy(), 0.0, None

    active = vals_t > candidate_price
    h_desired[active] = kappa_t * equity_t[active] / candidate_price

    total_desired = h_desired.sum()

    h_actual = np.zeros(n)
    debt_end = residual_debt.copy()
    marginal_idx = None

    if total_desired <= 0:
        return h_desired, h_actual, debt_end, total_desired, marginal_idx

    cumulative = 0.0
    for i in range(n):
        if not active[i]:
            continue

        if cumulative + h_desired[i] < H_supply - 1e-12:
            h_actual[i] = h_desired[i]
            cumulative += h_actual[i]
        else:
            h_actual[i] = max(0.0, H_supply - cumulative)
            cumulative += h_actual[i]
            marginal_idx = i
            break

    buyers = h_actual > 1e-12
    debt_end[buyers] = np.maximum(candidate_price * h_actual[buyers] - equity_t[buyers], 0.0)

    return h_desired, h_actual, debt_end, total_desired, marginal_idx


def solve_market_period_fixed_point(
    period_number: int,
    vals_t: np.ndarray,
    liquid_wealth_prev: np.ndarray,
    housing_prev: np.ndarray,
    debt_prev: np.ndarray,
    H_supply: float,
    kappa_t: float,
    interest_rate_t: float,
    price_grid_size: int = 500
):
    """
    Solve current period with fixed-point timing:

    Equity before purchase depends on current price,
    and demand depends on that equity.

    We search over candidate prices and select the one that makes
    total desired demand closest to H_supply.

    To help the solver, candidate prices include:
    - a dense numerical grid
    - all valuation points
    - previous owners' breakpoints implied by debt/housing ratios, when meaningful
    """
    n = len(vals_t)

    if H_supply <= 0:
        return 0.0, np.zeros(n), np.zeros(n), np.zeros(n), None, "invalid_supply"

    if kappa_t <= 0:
        equity0, residual0 = compute_equity_at_candidate_price(
            period_number=period_number,
            candidate_price=max(vals_t[-1], 1e-8),
            liquid_wealth_prev=liquid_wealth_prev,
            housing_prev=housing_prev,
            debt_prev=debt_prev,
            interest_rate_t=interest_rate_t
        )
        return 0.0, np.zeros(n), residual0, equity0, None, "no_buyers"

    # Price bounds
    p_min = max(1e-8, 0.05 * vals_t[-1])
    p_max = max(vals_t[0] * 1.75, p_min + 1.0)

    dense_grid = np.linspace(p_min, p_max, price_grid_size)

    # Useful breakpoints where some owner's equity crosses zero:
    owner_breakpoints = []
    owners_prev = housing_prev > 1e-12
    if period_number >= 2:
        for i in np.where(owners_prev)[0]:
            if housing_prev[i] > 1e-12:
                bp = ((1.0 + interest_rate_t) * debt_prev[i]) / housing_prev[i]
                if np.isfinite(bp) and bp > 0:
                    owner_breakpoints.append(bp)

    candidate_prices = np.unique(
        np.concatenate([
            dense_grid,
            vals_t,
            np.array(owner_breakpoints) if len(owner_breakpoints) > 0 else np.array([])
        ])
    )

    best_price = None
    best_gap = np.inf
    best_equity = None
    best_residual_debt = None
    best_h_actual = None
    best_debt_end = None
    best_marginal_idx = None
    best_total_desired = None

    for price in candidate_prices:
        equity_t, residual_debt = compute_equity_at_candidate_price(
            period_number=period_number,
            candidate_price=price,
            liquid_wealth_prev=liquid_wealth_prev,
            housing_prev=housing_prev,
            debt_prev=debt_prev,
            interest_rate_t=interest_rate_t
        )

        _, h_actual, debt_end, total_desired, marginal_idx = allocate_given_price(
            vals_t=vals_t,
            equity_t=equity_t,
            residual_debt=residual_debt,
            H_supply=H_supply,
            candidate_price=price,
            kappa_t=kappa_t
        )

        gap = abs(total_desired - H_supply)

        # Tie-breaker: among similar gaps, prefer lower price discrepancy
        # and prices that do not overshoot absurdly.
        if (gap < best_gap - 1e-12) or (
            abs(gap - best_gap) <= 1e-12 and (best_price is None or price < best_price)
        ):
            best_gap = gap
            best_price = price
            best_equity = equity_t.copy()
            best_residual_debt = residual_debt.copy()
            best_h_actual = h_actual.copy()
            best_debt_end = debt_end.copy()
            best_marginal_idx = marginal_idx
            best_total_desired = total_desired

    regime = "fixed_point_market_clearing"

    return (
        best_price,
        best_h_actual,
        best_debt_end,
        best_equity,
        best_residual_debt,
        best_marginal_idx,
        regime,
        best_total_desired
    )


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
    base_vals: np.ndarray
):
    """
    Advance one period using the fixed-point timing.
    """
    vals_t = apply_valuation_shock(
        vals_prev=vals_prev,
        base_vals=base_vals,
        shock_pct=valuation_shock_pct_t,
        cumulative_shocks=cumulative_shocks
    )

    (
        price_t,
        housing_t,
        debt_t,
        equity_before_purchase,
        residual_debt,
        marginal_idx,
        regime,
        total_desired
    ) = solve_market_period_fixed_point(
        period_number=period_number,
        vals_t=vals_t,
        liquid_wealth_prev=liquid_wealth_prev,
        housing_prev=housing_prev,
        debt_prev=debt_prev,
        H_supply=H_supply,
        kappa_t=kappa_t,
        interest_rate_t=interest_rate_t
    )

    buyers = housing_t > 1e-12

    # Non-buyers keep liquid wealth; buyers deploy it into housing.
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
        "total_desired_demand": total_desired,
        "owners": int(buyers.sum()),
        "insolvent_households": int((residual_debt > 1e-12).sum()),
        "marginal_household": None if marginal_idx is None else int(marginal_idx + 1),
        "top_valuation": vals_t[0],
        "bottom_valuation": vals_t[-1],
    }

    household_rows = []
    for i in range(len(vals_t)):
        household_rows.append({
            "period": period_number,
            "household": i + 1,
            "valuation": vals_t[i],
            "equity_before_purchase": equity_before_purchase[i],
            "residual_debt_before_purchase": residual_debt[i],
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
# Session state
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
# Sidebar
# ============================================================

st.title("Sequential Leverage Cycle Housing Simulator (Fixed-Point Timing)")

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

if "current_period" not in st.session_state or reset_clicked:
    initialize_simulation(n, V_high, V_low, E0)


# ============================================================
# Advance period
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
        base_vals=st.session_state.base_vals
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
# Status
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
# Household inspection
# ============================================================

st.subheader("Household-Level Results")

if st.session_state.household_history.empty:
    st.write("No household data yet.")
else:
    available_periods = sorted(
        int(x) for x in st.session_state.household_history["period"].unique().tolist()
    )

    selected_period = st.selectbox(
        "Select solved period to inspect",
        options=available_periods,
        index=len(available_periods) - 1
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
    summary_df = pd.DataFrame(st.session_state.summary_history)
    price_selected = summary_df.loc[summary_df["period"] == selected_period, "price"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(period_households["household"], period_households["valuation"], marker="o", label="Valuation")
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
- You solve **one period at a time**.
- In each period, households first liquidate old positions at the **current period price**.
- That makes beginning-of-period equity depend on the same price that clears the new market.
- So the app solves a **fixed-point problem** numerically.

For previous owners:
\[
E_t^i(P_t)=\max\left\{P_t h_{t-1}^i-(1+r_t)d_{t-1}^i,\,0\right\}
\]

For previous non-owners:
\[
E_t^i(P_t)=\max\left\{W_{t-1}^i-(1+r_t)d_{t-1}^i,\,0\right\}
\]

Demand:
\[
h_t^i=\frac{\kappa_t E_t^i(P_t)}{P_t}
\quad \text{if } V_t^i>P_t
\]

Debt for actual buyers:
\[
D_t^i=P_t h_t^i - E_t^i(P_t)
\]

This timing allows a positive current shock to increase owners' current equity immediately.
"""
)
