# supercycle.py
import math
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------- APP SETUP -------------------------------------
st.set_page_config(page_title="Cattle Cycle Game (Team vs Bot)", page_icon="🐄", layout="wide")

TITLE = "🐄 Cattle Cycle Game — Team vs Computer"
INSTRUCTOR_PASSWORD = "bovine"  # <<< change this to your secret before class

# -------------------------- CORE ECON FUNCTIONS ------------------------------
def compute_price(Q_t: float, A_t: float, epsilon: float) -> float:
    Q_safe = max(Q_t, 1e-10)
    return A_t * (Q_safe ** (-1.0 / epsilon))

def bot_investment(p_t: float, theta: float, gamma: float, I_cap: float | None) -> float:
    I = gamma * max(0.0, p_t - theta)
    if I_cap is not None:
        I = min(I, I_cap)
    return max(0.0, I)

def realize_profit(p: float, r: float, I_mature: float, unit_cost: float) -> float:
    # Revenue at maturity less cost paid at investment time
    return p * (1.0 + r) * I_mature - unit_cost * I_mature

# --------------------------- STATE INITIALIZATION ----------------------------
def init_state():
    """Initialize once; parameters are set in instructor mode and saved to session_state."""
    ss = st.session_state

    # Default (hidden) parameters; override via Instructor Panel
    ss.k = 3                 # gestation lag
    ss.epsilon = 2.0         # demand elasticity (>1 for damped)
    ss.r = 0.20              # production return so (1+r)I delivers after k periods
    ss.unit_cost = 1.0       # cost per unit investment (paid when invested)
    ss.bot_theta = 1.00      # price threshold for bot
    ss.bot_gamma = 0.80      # slope for bot
    ss.bot_cap = None        # cap on bot investment (None = no cap)
    ss.use_shocks = True     # we will start with positive shocks, then flat 1.0 afterwards
    ss.pre_shock_rounds = 10 # number of initial rounds with positive shocks
    ss.pre_shock_level = 1.03  # A_t during the positive shock window (level > 1)
    ss.after_shock_A = 1.00  # A_t after the shock window
    ss.seed = 123            # randomness if needed later
    ss.show_instructor = False

    # Game dynamic state
    ss.t = 0
    # Queues of investments that will mature in k periods
    ss.queue_h = deque([0.0]*ss.k, maxlen=ss.k)   # human investment pipeline (unrealized)
    ss.queue_b = deque([0.0]*ss.k, maxlen=ss.k)   # bot pipeline

    # Pre-fill pipeline so the first k prices are defined and near steady state
    I_star = 1.0 / (1.0 + ss.r)
    for _ in range(ss.k):
        ss.queue_h.appendleft(0.5 * I_star)
        ss.queue_b.appendleft(0.5 * I_star)

    # Logs
    ss.hist = []  # list of dict rows
    ss.balance_h = 0.0
    ss.balance_b = 0.0

def current_A_t():
    """Demand level path: first 'pre_shock_rounds' rounds set to >1, then 1."""
    ss = st.session_state
    if ss.use_shocks and ss.t < ss.pre_shock_rounds:
        return ss.pre_shock_level
    return ss.after_shock_A

def do_one_round(I_h_t: float):
    """Advance the system by 1 period with human investment I_h_t decided now."""
    ss = st.session_state

    # 1) Realize deliveries that mature now
    I_mature_h = ss.queue_h.popleft()
    I_mature_b = ss.queue_b.popleft()
    Q_t = (1.0 + ss.r) * (I_mature_h + I_mature_b)

    # 2) Demand level (A_t)
    A_t = current_A_t()

    # 3) Price
    p_t = compute_price(Q_t, A_t, ss.epsilon)

    # 4) Bot decides investment at observed p_t
    I_b_t = bot_investment(p_t, ss.bot_theta, ss.bot_gamma, ss.bot_cap)

    # 5) Queue new investments for delivery at t+k
    ss.queue_h.append(I_h_t)
    ss.queue_b.append(I_b_t)

    # 6) Profits realized now for the matured cohorts, deposited to balances
    pi_h = realize_profit(p_t, ss.r, I_mature_h, ss.unit_cost)
    pi_b = realize_profit(p_t, ss.r, I_mature_b, ss.unit_cost)
    ss.balance_h += pi_h
    ss.balance_b += pi_b

    # 7) Log snapshot
    row = dict(
        t=ss.t,
        A_t=A_t,
        Q_t=Q_t,
        p_t=p_t,
        I_h_t=I_h_t,
        I_b_t=I_b_t,
        deliver_h_now=I_mature_h*(1.0+ss.r),
        deliver_b_now=I_mature_b*(1.0+ss.r),
        pi_h=pi_h,
        pi_b=pi_b,
        bal_h=ss.balance_h,
        bal_b=ss.balance_b,
    )
    ss.hist.append(row)

    # 8) Advance time
    ss.t += 1

def hist_df():
    ss = st.session_state
    if len(ss.hist) == 0:
        return pd.DataFrame(columns=[
            "t","A_t","Q_t","p_t","I_h_t","I_b_t","deliver_h_now","deliver_b_now","pi_h","pi_b","bal_h","bal_b"
        ])
    return pd.DataFrame(ss.hist)

def pipeline_table():
    """Show next k periods of deliveries from *current* pipeline (after last decision)."""
    ss = st.session_state
    # queue ends are the newest entries that will mature furthest in the future
    # Deliveries are (1+r) * investment when they mature
    future = list(ss.queue_h)  # left -> next to mature
    # We want t+1 ... t+k deliveries from human + bot
    fut_h = [(1.0 + ss.r) * x for x in future]
    fut_b = [(1.0 + ss.r) * x for x in list(ss.queue_b)]
    periods = [f"t+{i}" for i in range(1, ss.k+1)]
    return pd.DataFrame({"Period": periods,
                         "Your delivery": np.round(fut_h, 4),
                         "Bot delivery":  np.round(fut_b, 4),
                         "Total delivery": np.round(np.array(fut_h)+np.array(fut_b), 4)})

# ------------------------------ UI LAYOUT ------------------------------------
if "t" not in st.session_state:
    init_state()

ss = st.session_state

st.title(TITLE)
st.caption("You decide investment each round. Production arrives after a fixed gestation. "
           "Your account collects revenue minus costs when your cohorts mature.")

# ------- Instructor Access (Hidden) -------
colA, colB, colC = st.columns([1,5,1])
with colA:
    if st.button("🔐", help="Instructor login"):
        ss.show_instructor = True

if ss.show_instructor:
    with st.expander("Instructor Panel", expanded=True):
        pwd = st.text_input("Password", type="password")
        if pwd == INSTRUCTOR_PASSWORD:
            st.success("Instructor mode enabled. Parameters below will affect gameplay immediately.")
            col1, col2, col3 = st.columns(3)
            with col1:
                ss.k = st.number_input("Gestation lag k", 1, 20, ss.k, 1)
                ss.epsilon = st.number_input("Demand elasticity ε", 0.2, 10.0, ss.epsilon, 0.1, format="%.1f")
                ss.r = st.number_input("Return r (invisible to students)", 0.0, 1.0, ss.r, 0.01, format="%.2f")
            with col2:
                ss.unit_cost = st.number_input("Unit cost c", 0.0, 10.0, ss.unit_cost, 0.1)
                ss.pre_shock_rounds = st.number_input("Initial positive shocks (rounds)", 0, 100, ss.pre_shock_rounds, 1)
                ss.pre_shock_level = st.number_input("A_t level during shocks (>1)", 1.0, 5.0, ss.pre_shock_level, 0.01)
            with col3:
                ss.after_shock_A = st.number_input("A_t after shocks", 0.1, 5.0, ss.after_shock_A, 0.01)
                ss.bot_theta = st.number_input("Bot threshold θ", 0.0, 5.0, ss.bot_theta, 0.05)
                ss.bot_gamma = st.number_input("Bot aggressiveness γ", 0.0, 10.0, ss.bot_gamma, 0.05)
                cap = st.number_input("Bot I cap (0 = none)", 0.0, 10000.0, 0.0, 10.0)
                ss.bot_cap = None if cap == 0.0 else cap

            st.markdown("---")
            if st.button("🔄 Reset Game with Current Parameters"):
                # Save params, then re-init dynamic state (preserve chosen params)
                k, eps, rr = ss.k, ss.epsilon, ss.r
                uc, th, gm, bc = ss.unit_cost, ss.bot_theta, ss.bot_gamma, ss.bot_cap
                preR, preL, aftA = ss.pre_shock_rounds, ss.pre_shock_level, ss.after_shock_A

                # fresh init
                init_state()
                # restore chosen parameters
                ss.k, ss.epsilon, ss.r = k, eps, rr
                ss.unit_cost = uc
                ss.bot_theta, ss.bot_gamma, ss.bot_cap = th, gm, bc
                ss.pre_shock_rounds, ss.pre_shock_level, ss.after_shock_A = preR, preL, aftA
                st.experimental_rerun()
        else:
            st.info("Enter the correct password to reveal instructor controls.")
    st.divider()

# --------------------------- STUDENT VIEW ------------------------------------
# Metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Round", ss.t)
with m2:
    last_p = ss.hist[-1]["p_t"] if ss.hist else None
    st.metric("Last Price", f"{last_p:.3f}" if last_p is not None else "—")
with m3:
    last_Q = ss.hist[-1]["Q_t"] if ss.hist else None
    st.metric("Last Output", f"{last_Q:.3f}" if last_Q is not None else "—")
with m4:
    st.metric("Account Balance (You)", f"{ss.balance_h:,.2f}")

# Student decision input
st.subheader("Your Decision")
colI1, colI2 = st.columns([2,1])
with colI1:
    I_h_t = st.number_input("Investment this round (delivers in the future)", min_value=0.0, value=0.80, step=0.05)
with colI2:
    advance = st.button("✅ Submit & Advance", use_container_width=True)

# Advance the game
if advance:
    do_one_round(I_h_t)

# Show pipeline for next k periods
st.subheader("Your Upcoming Production Pipeline")
pipe_df = pipeline_table()
st.dataframe(pipe_df, use_container_width=True, hide_index=True)

# Charts and tables
df = hist_df()
if len(df) > 0:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Price and Demand Level")
        st.line_chart(df.set_index("t")[["p_t"]])
        st.caption("Price over time. (Demand level is >1 only in the initial shock window.)")
    with c2:
        st.markdown("#### Output (Arrivals)")
        st.line_chart(df.set_index("t")[["Q_t"]])
        st.caption("Output delivered each round.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Investments (Your Team vs Bot)")
        st.line_chart(df.set_index("t")[["I_h_t","I_b_t"]])
    with c4:
        st.markdown("#### Profits on Matured Cohorts")
        st.line_chart(df.set_index("t")[["pi_h","pi_b"]])

    st.markdown("### Round-by-Round Data")
    show = df[["t","A_t","Q_t","p_t","I_h_t","I_b_t","deliver_h_now","deliver_b_now","pi_h","pi_b","bal_h","bal_b"]].copy()
    show.rename(columns={
        "A_t":"Demand level",
        "Q_t":"Output",
        "p_t":"Price",
        "I_h_t":"Your investment",
        "I_b_t":"Bot investment",
        "deliver_h_now":"Your deliveries (now)",
        "deliver_b_now":"Bot deliveries (now)",
        "pi_h":"Your profit (now)",
        "pi_b":"Bot profit (now)",
        "bal_h":"Your balance",
        "bal_b":"Bot balance",
    }, inplace=True)
    st.dataframe(show, use_container_width=True, hide_index=True)

st.markdown("---")
with st.expander("How this works (for students)"):
    st.markdown("""
- Each round you choose how much to **invest**. Your investment **delivers after a fixed gestation**.
- The computer invests more when the **price is high**.
- When a cohort matures, it sells at that round’s price; **sales minus costs** are deposited in your account.
- Price comes from **constant-elasticity demand** and total output arriving that round.
- The game starts **after a run of positive demand shocks**, so you begin with a data history and then play forward.
""")
