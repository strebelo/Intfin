import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Commodity Super Cycle Simulator", layout="wide")

st.title("Commodity Super Cycle Simulator")

st.markdown(
    """
This app simulates a commodity cycle sequentially.

Timing in each period:
1. The app displays only past and current variables.
2. You choose current investment \(I_t\).
3. The app checks the credit constraint.
4. The economy advances one period.
5. Then the new current-period results become visible.

Model:
- \(Y_t = I_{t-k}\)
- \(P_t = A_t Y_t^{-0.5}\)
- \(A_t\) rises permanently from 100 to 110 in period \(k+1\)
- Debt evolves as \(D_{t+1} = D_t(1+R) + I_t - P_tY_t\)
- Constraint: \(D_{t+1} \le \phi P_tY_t\)
"""
)

# ============================================================
# Sidebar parameters
# ============================================================
st.sidebar.header("Parameters")

T = int(st.sidebar.number_input("Number of periods (T)", min_value=5, max_value=200, value=20, step=1))
k = int(st.sidebar.number_input("Production lag k", min_value=1, max_value=20, value=3, step=1))
R = st.sidebar.number_input("Interest rate R", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")

initial_investment = st.sidebar.number_input(
    "Initial investment in pre-sample periods",
    min_value=0.0,
    value=100.0,
    step=10.0
)

initial_debt = st.sidebar.number_input(
    "Initial debt D_1",
    min_value=0.0,
    value=0.0,
    step=10.0
)

demand_before = st.sidebar.number_input(
    "Demand shifter before shock",
    min_value=1.0,
    value=100.0,
    step=1.0
)

demand_after = st.sidebar.number_input(
    "Demand shifter after shock",
    min_value=1.0,
    value=110.0,
    step=1.0
)

credit_share = st.sidebar.number_input(
    "Credit line share of revenue",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.05
)

# ============================================================
# Reset logic if parameters change
# ============================================================
param_signature = {
    "T": T,
    "k": k,
    "R": R,
    "initial_investment": initial_investment,
    "initial_debt": initial_debt,
    "demand_before": demand_before,
    "demand_after": demand_after,
    "credit_share": credit_share,
}

if "param_signature" not in st.session_state or st.session_state.param_signature != param_signature:
    st.session_state.param_signature = param_signature
    st.session_state.current_period = 1
    st.session_state.investment_history = [initial_investment] * k
    st.session_state.rows = []
    st.session_state.current_debt = initial_debt
    st.session_state.finished = False

# ============================================================
# Helpers
# ============================================================
def current_A(t, k, demand_before, demand_after):
    return demand_before if t <= k else demand_after

def current_state(t, k, investment_history, debt_t, demand_before, demand_after, credit_share, R):
    Y_t = investment_history[t - 1]
    A_t = current_A(t, k, demand_before, demand_after)

    if Y_t > 0:
        P_t = A_t * (Y_t ** (-0.5))
    else:
        P_t = np.nan

    revenue_t = P_t * Y_t if Y_t > 0 else 0.0
    credit_line_t = credit_share * revenue_t

    max_feasible_I_t = max(0.0, credit_line_t - debt_t * (1 + R) + revenue_t)

    return {
        "Period": t,
        "Demand_Shifter_A_t": A_t,
        "Output_Y_t": Y_t,
        "Price_P_t": P_t,
        "Revenue_PY_t": revenue_t,
        "Debt_D_t": debt_t,
        "Credit_Line": credit_line_t,
        "Max_Feasible_Investment": max_feasible_I_t,
    }

def run_one_period(t, desired_I_t):
    Y_t = st.session_state.investment_history[t - 1]
    A_t = current_A(t, k, demand_before, demand_after)

    if Y_t > 0:
        P_t = A_t * (Y_t ** (-0.5))
    else:
        P_t = np.nan

    revenue_t = P_t * Y_t if Y_t > 0 else 0.0
    credit_line_t = credit_share * revenue_t
    debt_t = st.session_state.current_debt

    max_feasible_I_t = max(0.0, credit_line_t - debt_t * (1 + R) + revenue_t)
    actual_I_t = min(desired_I_t, max_feasible_I_t)

    cash_flow_t = revenue_t - actual_I_t
    D_next = debt_t * (1 + R) + actual_I_t - revenue_t
    D_next = max(0.0, D_next)

    constrained = actual_I_t < desired_I_t - 1e-10

    row = {
        "Period": t,
        "Demand_Shifter_A_t": A_t,
        "Output_Y_t": Y_t,
        "Price_P_t": P_t,
        "Revenue_PY_t": revenue_t,
        "Desired_Investment_I_t": desired_I_t,
        "Actual_Investment_I_t": actual_I_t,
        "Cash_Flow_PY_minus_I": cash_flow_t,
        "Debt_D_t": debt_t,
        "Debt_D_t_plus_1": D_next,
        "Credit_Line": credit_line_t,
        "Max_Feasible_Investment": max_feasible_I_t,
        "Constraint_Binds": constrained,
    }

    st.session_state.rows.append(row)
    st.session_state.investment_history.append(actual_I_t)
    st.session_state.current_debt = D_next
    st.session_state.current_period += 1

    if st.session_state.current_period > T:
        st.session_state.finished = True

# ============================================================
# Current state
# ============================================================
t = st.session_state.current_period

if not st.session_state.finished:
    state = current_state(
        t=t,
        k=k,
        investment_history=st.session_state.investment_history,
        debt_t=st.session_state.current_debt,
        demand_before=demand_before,
        demand_after=demand_after,
        credit_share=credit_share,
        R=R,
    )

    st.subheader(f"Current period: {t}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Output", f"{state['Output_Y_t']:.2f}")
    c2.metric("Price", f"{state['Price_P_t']:.2f}")
    c3.metric("Revenue", f"{state['Revenue_PY_t']:.2f}")
    c4.metric("Debt at start of period", f"{state['Debt_D_t']:.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Demand shifter", f"{state['Demand_Shifter_A_t']:.2f}")
    c6.metric("Credit line", f"{state['Credit_Line']:.2f}")
    c7.metric("Max feasible investment", f"{state['Max_Feasible_Investment']:.2f}")

    st.markdown("### Choose current investment")

    desired_I_t = st.number_input(
        f"Desired investment I_{t}",
        min_value=0.0,
        value=float(min(100.0, state["Max_Feasible_Investment"])) if state["Max_Feasible_Investment"] > 0 else 0.0,
        step=10.0,
        key=f"desired_investment_{t}"
    )

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Advance one period", use_container_width=True):
            run_one_period(t, desired_I_t)
            st.rerun()

    with colB:
        if st.button("Reset simulation", use_container_width=True):
            st.session_state.current_period = 1
            st.session_state.investment_history = [initial_investment] * k
            st.session_state.rows = []
            st.session_state.current_debt = initial_debt
            st.session_state.finished = False
            st.rerun()

else:
    st.success("Simulation completed.")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Reset simulation", use_container_width=True):
            st.session_state.current_period = 1
            st.session_state.investment_history = [initial_investment] * k
            st.session_state.rows = []
            st.session_state.current_debt = initial_debt
            st.session_state.finished = False
            st.rerun()

# ============================================================
# Equations
# ============================================================
with st.expander("Equations used"):
    st.latex(r"Y_t = I_{t-k}")
    st.latex(
        r"""
        A_t =
        \begin{cases}
        100 & \text{if } t \le k \\
        110 & \text{if } t \ge k+1
        \end{cases}
        """
    )
    st.latex(r"P_t = A_t Y_t^{-0.5}")
    st.latex(r"\text{Revenue}_t = P_tY_t")
    st.latex(r"\text{CreditLine}_t = \phi P_tY_t")
    st.latex(r"D_{t+1} = D_t(1+R) + I_t - P_tY_t")
    st.latex(r"D_{t+1} \le \text{CreditLine}_t")

# ============================================================
# History shown only through present
# ============================================================
st.subheader("Observed history up to the present")

if len(st.session_state.rows) > 0:
    df = pd.DataFrame(st.session_state.rows)

    display_cols = [
        "Period",
        "Demand_Shifter_A_t",
        "Output_Y_t",
        "Price_P_t",
        "Revenue_PY_t",
        "Desired_Investment_I_t",
        "Actual_Investment_I_t",
        "Cash_Flow_PY_minus_I",
        "Debt_D_t",
        "Debt_D_t_plus_1",
        "Credit_Line",
        "Max_Feasible_Investment",
        "Constraint_Binds",
    ]

    st.dataframe(
        df[display_cols].style.format({
            "Demand_Shifter_A_t": "{:.2f}",
            "Output_Y_t": "{:.2f}",
            "Price_P_t": "{:.2f}",
            "Revenue_PY_t": "{:.2f}",
            "Desired_Investment_I_t": "{:.2f}",
            "Actual_Investment_I_t": "{:.2f}",
            "Cash_Flow_PY_minus_I": "{:.2f}",
            "Debt_D_t": "{:.2f}",
            "Debt_D_t_plus_1": "{:.2f}",
            "Credit_Line": "{:.2f}",
            "Max_Feasible_Investment": "{:.2f}",
        }),
        use_container_width=True
    )

    def line_chart(x, y, ylabel, title):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.subheader("History charts")
    line_chart(df["Period"], df["Output_Y_t"], "Output", "Output observed so far")
    line_chart(df["Period"], df["Price_P_t"], "Price", "Price observed so far")
    line_chart(df["Period"], df["Revenue_PY_t"], "Revenue", "Revenue observed so far")
    line_chart(df["Period"], df["Actual_Investment_I_t"], "Investment", "Actual investment chosen so far")
    line_chart(df["Period"], df["Debt_D_t_plus_1"], "Debt", "Debt observed so far")
    line_chart(df["Period"], df["Credit_Line"], "Credit line", "Credit line observed so far")

else:
    st.info("No realized periods yet. Choose investment for the current period and click 'Advance one period'.")
