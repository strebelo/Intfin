import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Leverage Cycle Housing Simulator - Predetermined Equity Timing",
    layout="wide"
)

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


def compute_beginning_equity_predetermined(
    period_number: int,
    last_price: float,
    liquid_wealth_prev: np.ndarray,
    housing_prev: np.ndarray,
    debt_prev: np.ndarray,
    interest_rate_t: float
):
    """
    Predetermined-equity timing:

    At the beginning of period t, households liquidate old housing at last period's price P_{t-1},
    repay debt, and then carry that equity into the current housing market.

    For period 1:
        beginning equity = initial liquid wealth

    For period t >= 2:
        raw_equity_i = liquid_wealth_prev_i + last_price * housing_prev_i - (1+r_t) * debt_prev_i

    Negative equity is clipped to zero, and the shortfall becomes residual debt.
    """
    n = len(liquid_wealth_prev)

    if period_number == 1:
        equity = liquid_wealth_prev.copy()
        residual_debt = np.zeros(n)
        return equity, residual_debt

    raw_equity = (
        liquid_wealth_prev
        + last_price * housing_prev
        - (1.0 + interest_rate_t) * debt_prev
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
    For a given price and predetermined equity vector, compute desired demand and actual allocation.

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
    Non-buyers keep residual debt from pre-existing insolvency if any.
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


def solve_market_period_predetermined_equity(
    period_number: int,
    vals_t: np.ndarray,
    equity_t: np.ndarray,
    residual_debt: np.ndarray,
    H_supply: float,
    kappa_t: float,
    price_grid_size: int = 1000
):
    """
    Solve the current market with predetermined equity.

    Because equity_t is fixed before current price is solved, there is no fixed point.
    We search over candidate prices and choose the one whose desired demand is closest to H_supply.

    Tie-breaker:
    - among equally good prices, choose the HIGHER price
    """
    n = len(vals_t)

    if H_supply <= 0:
        return 0.0, np.zeros(n), residual_debt.copy(), None, "invalid_supply", 0.0

    if kappa_t <= 0:
        return 0.0, np.zeros(n), residual_debt.copy(), None, "no_buyers", 0.0

    p_min = max(1e-8, 0.02 * vals_t[-1])
    p_max = max(vals_t[0] * 1.5, p_min + 1.0)

    candidate_prices = np.unique(
        np.concatenate([
            np.linspace(p_min, p_max, price_grid_size),
            vals_t
        ])
    )

    best_price = None
    best_gap = np.inf
    best_h_actual = None
    best_debt_end = None
    best_marginal_idx = None
    best_total_desired = None

    for price in candidate_prices:
        _, h_actual, debt_end, total_desired, marginal_idx = allocate_given_price(
            vals_t=vals_t,
            equity_t=equity_t,
            residual_debt=residual_debt,
            H_supply=H_supply,
            candidate_price=price,
            kappa_t=kappa_t
        )

        gap = abs(total_desired - H_supply)

        if (gap < best_gap - 1e-12) or (
            abs(gap - best_gap) <= 1e-12 and (best_price is None or price > best_price)
        ):
            best_gap = gap
            best_price = price
            best_h_actual = h_actual.copy()
            best_debt_end = debt_end.copy()
            best_marginal_idx = marginal_idx
            best_total_desired = total_desired

    regime = "predetermined_equity_market_clearing"

    return (
        best_price,
        best_h_actual,
        best_debt_end,
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
    last_price: float,
    H_supply: float,
    kappa_t: float,
    interest_rate_t: float,
    valuation_shock_pct_t: float,
    cumulative_shocks: bool,
    base_vals: np.ndarray
):
    """
    Advance one period under predetermined-equity timing.
    """
    equity_before_purchase, residual_debt = compute_beginning_equity_predetermined(
        period_number=period_number,
        last_price=last_price,
        liquid_wealth_prev=liquid_wealth_prev,
        housing_prev=housing_prev,
        debt_prev=debt_prev,
        interest_rate_t=interest_rate_t
    )

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
        marginal_idx,
        regime,
        total_desired
    ) = solve_market_period_predetermined_equity(
        period_number=period_number,
        vals_t=vals_t,
        equity_t=equity_before_purchase,
        residual_debt=residual_debt,
        H_supply=H_supply,
        kappa_t=kappa_t
    )

    buyers = housing_t > 1e-12

    # Buyers deploy all available equity into housing.
    # Non-buyers carry remaining liquid wealth forward.
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

    # Important:
    # Use an initial reference price before period 1 is solved.
    # A natural choice is the midpoint of the valuation range, though it is only used from period 2 on.
    st.session_state.last_price = 0.5 * (V_high + V_low)

    st.session_state.summary_history = []
    st.session_state.household_history = pd.DataFrame()


# ============================================================
# Sidebar
# ============================================================

st.title("Sequential Leverage Cycle Housing Simulator (Predetermined Equity Timing)")

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

    next_kappa = st.number_input(
        r"Leverage multiple for next period ($\kappa_t$)",
        min_value=0.0,
        value=5.0,
        step=0.1
    )
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
        last_price=st.session_state.last_price,
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
- At the beginning of period \(t\), households liquidate last period's housing at the **old price** \(P_{t-1}\).
- They repay debt, so beginning-of-period equity is determined **before** solving for the new price.
- Then current valuations are realized.
- The current price \(P_t\) clears the market using that predetermined equity.

Beginning-of-period equity:
\[
E_t^i=\max\left\{W_{t-1}^i + P_{t-1} h_{t-1}^i - (1+r_t)d_{t-1}^i,\ 0\right\}
\]

Residual debt if underwater:
\[
R_t^i=\max\left\{(1+r_t)d_{t-1}^i - W_{t-1}^i - P_{t-1} h_{t-1}^i,\ 0\right\}
\]

Demand:
\[
h_t^i=\frac{\kappa_t E_t^i}{P_t}
\quad \text{if } V_t^i>P_t
\]

Debt for actual buyers:
\[
D_t^i=P_t h_t^i - E_t^i
\]

This timing removes the current-period fixed-point loop between \(P_t\) and \(E_t^i\).
"""
)
