import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Commodity Super Cycle Simulator", layout="wide")

st.title("Commodity Super Cycle Simulator")

st.markdown(
    """
This app simulates a simple commodity cycle with:
- production lag: \(Y_t = I_{t-k}\)
- inverse demand: \(P_t = A_t Y_t^{-0.5}\)
- permanent demand shift from \(A=100\) to \(A=110\) in period \(k+1\)
- borrowing limit: debt cannot exceed 50% of current revenue
"""
)

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Parameters")

T = st.sidebar.number_input("Number of periods (T)", min_value=5, max_value=100, value=20, step=1)
k = st.sidebar.number_input("Production lag k", min_value=1, max_value=10, value=3, step=1)
R = st.sidebar.number_input("Interest rate R", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")

initial_investment = st.sidebar.number_input(
    "Initial investment for pre-sample periods",
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
# Investment planning interface: blocks of k investments
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Investment plan in blocks of k periods")

num_blocks = int(np.ceil(T / k))

st.sidebar.markdown(
    f"You have **T = {T}** periods and **k = {k}**, so the app uses **{num_blocks} block(s)** of size **k**."
)

user_investments = []

for b in range(num_blocks):
    start_period = b * k + 1
    end_period = min((b + 1) * k, T)

    st.sidebar.markdown(f"### Block {b+1}: periods {start_period} to {end_period}")

    n_rows = end_period - start_period + 1

    block_df = pd.DataFrame({
        "Period": list(range(start_period, end_period + 1)),
        "Investment": [100.0] * n_rows
    })

    edited_block = st.sidebar.data_editor(
        block_df,
        num_rows="fixed",
        hide_index=True,
        key=f"investment_block_{b+1}"
    )

    user_investments.extend(edited_block["Investment"].tolist())

# ============================================================
# Simulation
# ============================================================
investment_history = [initial_investment] * k

rows = []
D_t = initial_debt

for t in range(1, T + 1):
    # Output comes from investment k periods ago
    Y_t = investment_history[t - 1]

    # Demand shift occurs permanently in period k+1
    A_t = demand_before if t <= k else demand_after

    # Price
    if Y_t > 0:
        P_t = A_t * (Y_t ** (-0.5))
    else:
        P_t = np.nan

    revenue_t = P_t * Y_t if Y_t > 0 else 0.0
    credit_line_t = credit_share * revenue_t

    # User desired investment
    desired_I_t = user_investments[t - 1]

    # Borrowing constraint:
    # D_{t+1} = D_t(1+R) + I_t - revenue_t <= credit_line_t
    # => I_t <= credit_line_t - D_t(1+R) + revenue_t
    max_feasible_I_t = max(0.0, credit_line_t - D_t * (1 + R) + revenue_t)

    actual_I_t = min(desired_I_t, max_feasible_I_t)

    cash_flow_t = revenue_t - actual_I_t
    D_next = D_t * (1 + R) + actual_I_t - revenue_t

    D_next = max(0.0, D_next)

    constrained = actual_I_t < desired_I_t - 1e-10

    rows.append({
        "Period": t,
        "Demand_Shifter_A_t": A_t,
        "Output_Y_t": Y_t,
        "Price_P_t": P_t,
        "Revenue_PY_t": revenue_t,
        "Desired_Investment_I_t": desired_I_t,
        "Actual_Investment_I_t": actual_I_t,
        "Cash_Flow_PY_minus_I": cash_flow_t,
        "Debt_D_t": D_t,
        "Debt_D_t_plus_1": D_next,
        "Credit_Line": credit_line_t,
        "Max_Feasible_Investment": max_feasible_I_t,
        "Constraint_Binds": constrained
    })

    investment_history.append(actual_I_t)
    D_t = D_next

df = pd.DataFrame(rows)

# ============================================================
# Display equations
# ============================================================
st.subheader("Equations used")

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
st.latex(r"\text{Revenue}_t = P_t Y_t")
st.latex(r"CF_t = P_tY_t - I_t")
st.latex(r"\text{CreditLine}_t = \phi P_tY_t")
st.latex(r"D_{t+1} = D_t(1+R) + I_t - P_tY_t")
st.latex(r"D_{t+1} \le \text{CreditLine}_t")

st.markdown(
    r"""
Here the app uses

\[
D_{t+1} = D_t(1+R) + I_t - P_tY_t
\]

so debt rises when investment exceeds revenue and falls when revenue exceeds investment.
"""
)

# ============================================================
# Results table
# ============================================================
st.subheader("Simulation table")
st.dataframe(df.style.format({
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
}))

# ============================================================
# Plots
# ============================================================
def line_chart(x, y, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Period")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.subheader("Charts")

line_chart(df["Period"], df["Output_Y_t"], "Output", "Output over time")
line_chart(df["Period"], df["Price_P_t"], "Price", "Price over time")
line_chart(df["Period"], df["Revenue_PY_t"], "Revenue", "Revenue over time")
line_chart(df["Period"], df["Actual_Investment_I_t"], "Investment", "Actual investment over time")
line_chart(df["Period"], df["Debt_D_t_plus_1"], "Debt", "Debt over time")

# ============================================================
# Summary
# ============================================================
st.subheader("Summary statistics")

num_binding = int(df["Constraint_Binds"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Average price", f"{df['Price_P_t'].mean():.2f}")
col2.metric("Average output", f"{df['Output_Y_t'].mean():.2f}")
col3.metric("Times constraint binds", f"{num_binding}")

st.markdown(
    """
### Interpretation
- The permanent demand increase raises prices immediately once it arrives.
- Because output depends on past investment, supply reacts only after \(k\) periods.
- That lag can generate a boom in prices and investment, followed later by an increase in output and some price reversal.
- The credit line can amplify the cycle by restricting investment when revenue is weak.
"""
)
