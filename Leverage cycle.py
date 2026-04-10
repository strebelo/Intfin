import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Leverage Cycle Housing Simulator", layout="wide")

# ============================================================
# Helpers
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

    shock_pcts should be in decimal form, e.g. 0.10 for +10%.
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

    Assumptions:
    - Households are already ordered by descending valuation.
    - Household i can spend at most kappa_t * E_i.
    - If V_i > P, household i buys as much as it can:
          h_i = kappa_t * E_i / P
    - The marginal household buys the residual supply.
    - If all households need to be active, price adjusts below the lowest valuation if needed.

    Returns:
    - price
    - housing vector h
    - new debt vector d (after purchases)
    - marginal index
    """
    n = len(vals_t)
    h = np.zeros(n)
    d = np.zeros(n)

    # If nobody has equity, no one can buy
    total_equity = equity_t.sum()
    if total_equity <= 0 or H_supply <= 0:
        return 0.0, h, d, None

    # Candidate logic:
    # Try price = valuation of household m.
    # At that price, top m households can buy "full" and household m can be marginal.
    for m in range(n):
        P_candidate = vals_t[m]

        if P_candidate <= 0:
            continue

        full_demand_before_m = kappa_t * equity_t[:m].sum() / P_candidate
        full_demand_up_to_m = kappa_t * equity_t[:m+1].sum() / P_candidate

        # Need:
        #   demand from higher-valuation households <= supply
        #   supply <= demand including household m
        if full_demand_before_m <= H_supply + 1e-12 and full_demand_up_to_m >= H_supply - 1e-12:
            price = P_candidate

            # Higher-valuation households buy as much as they can
            if m > 0:
                h[:m] = kappa_t * equity_t[:m] / price

            # Marginal household buys the residual
            residual = H_supply - h[:m].sum()
            h[m] = max(0.0, residual)

            # Debt for buyers with positive holdings
            buyers = h > 1e-12
            d[buyers] = (kappa_t - 1.0) * equity_t[buyers]

            return price, h, d, m

    # If no marginal household price among the valuation grid works,
    # all households are active and the price must satisfy:
    #   sum_i kappa_t * E_i / P = H
    price = kappa_t * total_equity / H_supply if H_supply > 0 else 0.0

    if price > 0:
        h = kappa_t * equity_t / price
        d[h > 1e-12] = (kappa_t - 1.0) * equity_t[h > 1e-12]

    return price, h, d, n - 1


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
    - period
    - kappa
    - interest_rate
    - valuation_shock_pct   (entered as percentages, e.g. 10 means +10%)

    Returns:
    - summary_df
    - household_df
    """
    T = len(period_table)
    base_vals = make_base_valuations(n, V_high, V_low)

    shock_pcts = period_table["valuation_shock_pct"].to_numpy(dtype=float) / 100.0
    kappas = period_table["kappa"].to_numpy(dtype=float)
    rates = period_table["interest_rate"].to_numpy(dtype=float) / 100.0

    valuation_path = build_valuation_path(base_vals, shock_pcts, cumulative=cumulative_shocks)

    # State variables
    equity = np.full(n, E0, dtype=float)  # period-0 equity is exogenous
    housing_prev = np.zeros(n, dtype=float)
    debt_prev = np.zeros(n, dtype=float)

    summary_rows = []
    household_rows = []

    for t in range(T):
        vals_t = valuation_path[t, :]
        kappa_t = kappas[t]
        R_t = rates[t]

        # In t=0, equity is exogenous as given.
        # For t>0, equity depends on last period's holdings and debt.
        if t > 0:
            raw_equity = price_t * housing_prev - (1.0 + R_t) * debt_prev

            # If raw equity is negative, set equity to zero
            # and carry residual debt as specified in your model.
            residual_debt = np.where(raw_equity < 0, -raw_equity, 0.0)
            equity = np.where(raw_equity > 0, raw_equity, 0.0)
        else:
            residual_debt = np.zeros(n)

        # Solve current period
        price_t, housing_t, new_debt_t, marginal_idx = solve_market_period(
            vals_t=vals_t,
            equity_t=equity,
            H_supply=H_supply,
            kappa_t=kappa_t
        )

        # Households with residual debt from negative equity cannot buy unless they have positive equity.
        # Since equity is zero for them, they naturally buy zero in the solver.

        # Total debt after current decisions:
        # - if household buys: debt = (kappa_t - 1) * equity
        # - if household does not buy and had positive equity: debt = 0
        # - if household had negative equity and was reset to zero: carry residual debt
        debt_t = residual_debt.copy()

        buyers = housing_t > 1e-12
        debt_t[buyers] = new_debt_t[buyers]

        owners = (housing_t > 1e-12).sum()
        insolvent = (residual_debt > 1e-12).sum()

        summary_rows.append({
            "period": t,
            "price": price_t,
            "kappa": kappa_t,
            "interest_rate_pct": 100 * R_t,
            "valuation_shock_pct": 100 * shock_pcts[t],
            "total_equity_before_purchase": equity.sum(),
            "total_housing_demand": housing_t.sum(),
            "owners": int(owners),
            "insolvent_households": int(insolvent),
            "marginal_household": None if marginal_idx is None else int(marginal_idx + 1),
            "valuation_top": vals_t[0],
            "valuation_bottom": vals_t[-1],
        })

        for i in range(n):
            household_rows.append({
                "period": t,
                "household": i + 1,
                "valuation": vals_t[i],
                "equity_before_purchase": equity[i],
                "housing": housing_t[i],
                "debt_end_of_period": debt_t[i],
                "residual_debt_from_insolvency": residual_debt[i],
                "buyer": int(housing_t[i] > 1e-12),
            })

        # Advance states
        housing_prev = housing_t.copy()
        debt_prev = debt_t.copy()

    summary_df = pd.DataFrame(summary_rows)
    household_df = pd.DataFrame(household_rows)
    return summary_df, household_df


# ============================================================
# Sidebar / left-column controls
# ============================================================

st.title("Leverage Cycle Housing Simulator")

with st.sidebar:
    st.header("Global Parameters")

    n = st.number_input("Number of households (n)", min_value=2, max_value=500, value=10, step=1)
    H_supply = st.number_input("Number of houses (H)", min_value=1.0, max_value=float(n - 1), value=5.0, step=1.0)
    T = st.number_input("Number of periods (T)", min_value=1, max_value=50, value=6, step=1)

    st.markdown("---")

    V_high = st.number_input(r"Upper valuation bound ($V^h$)", min_value=0.01, value=500.0, step=10.0)
    V_low = st.number_input(r"Lower valuation bound ($V^l$)", min_value=0.01, value=50.0, step=10.0)

    E0 = st.number_input(r"Initial equity ($E_0$)", min_value=0.0, value=100.0, step=10.0)

    default_kappa = st.number_input(r"Default leverage multiple ($\kappa$)", min_value=1.0, value=5.0, step=0.1)
    default_r = st.number_input("Default interest rate per period (%)", min_value=-99.0, value=5.0, step=0.5)
    default_shock = st.number_input("Default valuation shock per period (%)", value=0.0, step=1.0)

    cumulative_shocks = st.checkbox("Make valuation shocks cumulative over time", value=True)

    st.markdown("---")
    st.subheader("Period-by-Period Inputs")

    # Build editable period table
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
# Main output
# ============================================================

st.subheader("Simulation Summary")
st.dataframe(summary_df, use_container_width=True)

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

st.subheader("Notes on the Implementation")
st.markdown(
    r"""
1. Valuations are equally spaced between \(V^h\) and \(V^l\), and households are ordered from highest to lowest valuation.

2. A household with positive equity \(E_t^i\) can spend at most \(\kappa_t E_t^i\), so if it buys at price \(P_t\), its maximum housing demand is
   \[
   H_t^i = \frac{\kappa_t E_t^i}{P_t}.
   \]

3. Debt for buyers is set to
   \[
   D_t^i = (\kappa_t - 1) E_t^i.
   \]

4. Equity is updated from previous holdings and debt using your formula. If this is negative, equity is reset to zero and the shortfall is carried as residual debt.

5. The market-clearing price is chosen so that all households with valuation above the price buy as much as they can, and the marginal household absorbs the remaining supply.
"""
)
