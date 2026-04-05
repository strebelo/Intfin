import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Price Plan Detector", layout="wide")


# ============================================================
# Objects
# ============================================================

@dataclass
class Plan:
    regular: float
    sale: float

    def label(self) -> str:
        return f"R={self.regular:g}, S={self.sale:g}"


# ============================================================
# Data loading and validation
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def validate_input_df(df: pd.DataFrame) -> pd.DataFrame:
    # Robustify column names
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "time" not in df.columns or "price" not in df.columns:
        raise ValueError(
            f"Columns found: {list(df.columns)}. "
            "The file must contain columns named 'time' and 'price'."
        )

    out = df[["time", "price"]].copy()

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    if out[["time", "price"]].isna().any().any():
        raise ValueError("The columns 'time' and 'price' must be numeric and contain no missing values.")

    out = out.sort_values("time").reset_index(drop=True)

    if out["time"].duplicated().any():
        raise ValueError("The 'time' column must not contain duplicates.")

    return out


# ============================================================
# Initialization
# ============================================================

def normalize_probs(p_reg: float, p_sale: float, p_other: float) -> Tuple[float, float, float]:
    total = p_reg + p_sale + p_other
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive number.")
    return p_reg / total, p_sale / total, p_other / total


def infer_initial_plan_from_series(prices: np.ndarray) -> Tuple[float, float]:
    """
    Rule:
    - Let p0 be the first observed price.
    - Find the first later price p1 that differs from p0.
    - If p1 < p0: initial plan = (regular=p0, sale=p1)
    - If p1 > p0: initial plan = (regular=p1, sale=p0)

    If no different price is observed, return (p0, p0).
    """
    if len(prices) == 0:
        raise ValueError("Price series is empty.")

    first_price = float(prices[0])
    next_different = None

    for p in prices[1:]:
        p = float(p)
        if p != first_price:
            next_different = p
            break

    if next_different is None:
        return first_price, first_price

    if next_different < first_price:
        return first_price, next_different
    else:
        return next_different, first_price


def build_candidate_plans(
    prices: np.ndarray,
    initial_regular: float,
    initial_sale: float,
    max_plans: int,
) -> List[Plan]:
    """
    Build candidate plans from common observed prices plus the inferred initial plan.
    """
    plans = {(float(initial_regular), float(initial_sale))}

    counts = pd.Series(prices).value_counts()
    common_prices = list(counts.index[: min(len(counts), 8)])

    # Add plausible pairs regular > sale
    for r, s in itertools.permutations(common_prices, 2):
        if float(r) > float(s):
            plans.add((float(r), float(s)))

    # Fallback if data are sparse
    unique_prices = sorted(pd.Series(prices).dropna().unique())
    if len(plans) < 2:
        for r, s in itertools.permutations(unique_prices, 2):
            if float(r) > float(s):
                plans.add((float(r), float(s)))

    plan_objs = [Plan(r, s) for r, s in plans]

    # Score plans by how much support they have in the data
    def score(plan: Plan) -> float:
        return float(((prices == plan.regular) | (prices == plan.sale)).sum())

    plan_objs = sorted(plan_objs, key=score, reverse=True)

    # Keep inferred initial plan first
    initial = Plan(float(initial_regular), float(initial_sale))
    dedup: Dict[Tuple[float, float], Plan] = {(initial.regular, initial.sale): initial}
    for p in plan_objs:
        dedup[(p.regular, p.sale)] = p

    ordered = [initial] + [
        p for key, p in dedup.items()
        if key != (initial.regular, initial.sale)
    ]

    return ordered[:max_plans]


# ============================================================
# Model primitives
# ============================================================

def emission_prob(
    observed_price: float,
    plan: Plan,
    p_reg: float,
    p_sale: float,
    p_other: float,
    other_support: int,
) -> float:
    if observed_price == plan.regular:
        return p_reg
    if observed_price == plan.sale:
        return p_sale

    denom = max(other_support, 1)
    return p_other / denom


def build_transition_matrix(n_states: int, stay_prob: float) -> np.ndarray:
    if n_states == 1:
        return np.array([[1.0]])

    if not (0 < stay_prob < 1):
        raise ValueError("stay_prob must be strictly between 0 and 1.")

    switch_prob = (1.0 - stay_prob) / (n_states - 1)
    trans = np.full((n_states, n_states), switch_prob)
    np.fill_diagonal(trans, stay_prob)
    return trans


# ============================================================
# Decoding
# ============================================================

def viterbi_decode(
    prices: np.ndarray,
    plans: List[Plan],
    p_reg: float,
    p_sale: float,
    p_other: float,
    stay_prob: float,
    init_state_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Viterbi decoding with the constraint:
    at time t, the regular price of an admissible plan
    cannot exceed the maximum observed price up to and including t.
    """
    n = len(prices)
    k = len(plans)

    unique_prices = np.unique(prices)
    other_support = max(len(unique_prices) - 2, 1)

    trans = build_transition_matrix(k, stay_prob)
    log_trans = np.log(np.clip(trans, 1e-14, None))

    # Running max observed price up to t
    running_max_price = np.maximum.accumulate(prices)

    # Emission log-likelihoods with admissibility constraint
    log_emit = np.full((n, k), -np.inf)
    for t in range(n):
        for j, plan in enumerate(plans):
            if plan.regular > running_max_price[t]:
                continue
            prob = emission_prob(prices[t], plan, p_reg, p_sale, p_other, other_support)
            log_emit[t, j] = np.log(max(prob, 1e-14))

    # Admissible states at t=0
    admissible_t0 = np.isfinite(log_emit[0])
    if not admissible_t0.any():
        raise ValueError(
            "No admissible price plan at time 0. "
            "This can happen if the first observation looks like a sale and the inferred regular price has not appeared yet."
        )

    # If the inferred initial state is admissible, strongly favor it.
    # Otherwise, distribute mass across all admissible states at t=0.
    init_probs = np.zeros(k)
    if 0 <= init_state_index < k and admissible_t0[init_state_index]:
        init_probs[init_state_index] = 1.0
    else:
        init_probs[admissible_t0] = 1.0
        init_probs = init_probs / init_probs.sum()

    log_init = np.log(np.clip(init_probs, 1e-14, None))

    dp = np.full((n, k), -np.inf)
    ptr = np.zeros((n, k), dtype=int)

    dp[0] = log_init + log_emit[0]

    for t in range(1, n):
        for j in range(k):
            if not np.isfinite(log_emit[t, j]):
                continue
            candidates = dp[t - 1] + log_trans[:, j]
            best_prev = int(np.argmax(candidates))
            ptr[t, j] = best_prev
            dp[t, j] = candidates[best_prev] + log_emit[t, j]

    if not np.isfinite(dp[-1]).any():
        raise ValueError(
            "No feasible path satisfies the regular-price constraint. "
            "Try increasing the number of candidate plans or lowering persistence."
        )

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))

    for t in range(n - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]

    return states, dp


# ============================================================
# Output tables
# ============================================================

def summarize_results(df: pd.DataFrame, states: np.ndarray, plans: List[Plan]) -> pd.DataFrame:
    out = df.copy()
    out["state"] = states
    out["regular_plan"] = [plans[s].regular for s in states]
    out["sale_plan"] = [plans[s].sale for s in states]
    out["plan_label"] = [plans[s].label() for s in states]

    def classify(row) -> str:
        p = row["price"]
        r = row["regular_plan"]
        s = row["sale_plan"]
        if p == r:
            return "regular"
        if p == s:
            return "sale"
        return "other"

    out["observation_type"] = out.apply(classify, axis=1)
    return out


def spells_table(results: pd.DataFrame) -> pd.DataFrame:
    starts = [0]
    for i in range(1, len(results)):
        if results.loc[i, "plan_label"] != results.loc[i - 1, "plan_label"]:
            starts.append(i)

    rows = []
    for start_idx, end_idx in zip(starts, starts[1:] + [len(results)]):
        chunk = results.iloc[start_idx:end_idx]
        rows.append({
            "start_time": chunk["time"].iloc[0],
            "end_time": chunk["time"].iloc[-1],
            "plan": chunk["plan_label"].iloc[0],
            "regular_price": chunk["regular_plan"].iloc[0],
            "sale_price": chunk["sale_plan"].iloc[0],
            "n_obs": len(chunk),
            "n_regular": int((chunk["observation_type"] == "regular").sum()),
            "n_sale": int((chunk["observation_type"] == "sale").sum()),
            "n_other": int((chunk["observation_type"] == "other").sum()),
        })

    return pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

def plot_price_plan(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        results["time"],
        results["price"],
        marker="o",
        linewidth=1.8,
        label="Observed price",
    )

    ax.plot(
        results["time"],
        results["regular_plan"],
        linestyle="--",
        linewidth=2,
        label="Detected regular price",
    )

    ax.plot(
        results["time"],
        results["sale_plan"],
        linestyle=":",
        linewidth=2.4,
        label="Detected sale price",
    )

    mask_other = results["observation_type"] == "other"
    if mask_other.any():
        ax.scatter(
            results.loc[mask_other, "time"],
            results.loc[mask_other, "price"],
            s=80,
            marker="x",
            label="Other price",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title("Observed prices and detected price plan")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ============================================================
# Example data
# ============================================================

def default_example() -> pd.DataFrame:
    return pd.DataFrame({
        "time": list(range(1, 53)),
        "price": [
            10, 10, 5, 5, 10, 7, 10, 5, 10, 11, 6, 6, 6, 11, 7, 6, 6, 11,
            10, 5, 5, 7, 5, 10, 12, 12, 12, 12, 6, 6, 12, 6, 7, 8, 6, 12,
            12, 6, 12, 6, 12, 6, 12, 12, 7, 13, 12, 12, 12, 6, 6, 6,
        ],
    })


# ============================================================
# Streamlit UI
# ============================================================

st.title("Price plan detector")

st.write(
    "Upload a CSV with columns for time and price. The app will infer the initial "
    "price plan from the first two distinct prices unless you choose manual input."
)

with st.sidebar:
    st.header("Inputs")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.subheader("Initial plan")
    auto_infer_initial = st.checkbox(
        "Infer initial plan from the first two distinct prices",
        value=True,
    )

    manual_initial_regular = st.number_input(
        "Manual initial regular price",
        value=10.0,
        step=0.01,
    )
    manual_initial_sale = st.number_input(
        "Manual initial sale price",
        value=5.0,
        step=0.01,
    )

    st.subheader("Emission probabilities")
    p_reg = st.number_input("P(regular)", min_value=0.0, value=0.45, step=0.05)
    p_sale = st.number_input("P(sale)", min_value=0.0, value=0.45, step=0.05)
    p_other = st.number_input("P(other)", min_value=0.0, value=0.10, step=0.05)

    st.subheader("Persistence")
    stay_prob = st.slider(
        "Probability current plan stays active",
        min_value=0.50,
        max_value=0.999,
        value=0.92,
        step=0.01,
    )

    max_plans = st.slider(
        "Maximum number of candidate plans",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
    )

    run_button = st.button("Detect price plan", type="primary")


# ============================================================
# Main app logic
# ============================================================

try:
    if uploaded_file is not None:
        df_raw = load_csv(uploaded_file)
        df = validate_input_df(df_raw)
    else:
        st.info("No file uploaded yet. Showing built-in example data.")
        df = default_example()

    st.subheader("Input data")
    st.dataframe(df, use_container_width=True)

    prices = df["price"].to_numpy(dtype=float)

    if auto_infer_initial:
        initial_regular, initial_sale = infer_initial_plan_from_series(prices)
    else:
        initial_regular = float(manual_initial_regular)
        initial_sale = float(manual_initial_sale)

    st.subheader("Initial plan used")
    st.dataframe(
        pd.DataFrame({
            "initial_regular": [initial_regular],
            "initial_sale": [initial_sale],
        }),
        use_container_width=True,
    )

    if run_button or uploaded_file is None:
        p_reg_n, p_sale_n, p_other_n = normalize_probs(p_reg, p_sale, p_other)

        if initial_regular < initial_sale:
            raise ValueError("Initial regular price must be at least as large as initial sale price.")

        plans = build_candidate_plans(
            prices=prices,
            initial_regular=initial_regular,
            initial_sale=initial_sale,
            max_plans=max_plans,
        )

        states, _ = viterbi_decode(
            prices=prices,
            plans=plans,
            p_reg=p_reg_n,
            p_sale=p_sale_n,
            p_other=p_other_n,
            stay_prob=stay_prob,
            init_state_index=0,
        )

        results = summarize_results(df, states, plans)
        spell_df = spells_table(results)

        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("Detected plan over time")
            fig = plot_price_plan(results)
            st.pyplot(fig)

        with c2:
            st.subheader("Candidate plans")
            cand_df = pd.DataFrame({
                "state": list(range(len(plans))),
                "regular": [p.regular for p in plans],
                "sale": [p.sale for p in plans],
                "label": [p.label() for p in plans],
            })
            st.dataframe(cand_df, use_container_width=True)

        st.subheader("Detected spells")
        st.dataframe(spell_df, use_container_width=True)

        st.subheader("Observation-level results")
        st.dataframe(results, use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="detected_price_plan.csv",
            mime="text/csv",
        )

        with st.expander("Notes"):
            st.markdown(
                """
                **Initial-plan rule**

                - Take the first observed price.
                - Find the next observed price that is different.
                - If the next different price is lower, the first price is treated as the initial regular price.
                - If the next different price is higher, the first price is treated as the initial sale price.

                **Admissibility rule**

                - At time `t`, the regular price of the detected plan cannot exceed the highest observed price up to time `t`.

                **Persistence**

                - The model prefers to remain in the same plan from one period to the next.
                - The strength of that preference is controlled by `stay_prob`.

                **Observation labels**

                - If observed price = detected regular price, label = `regular`
                - If observed price = detected sale price, label = `sale`
                - Otherwise, label = `other`

                **Limitation**

                - This version uses an HMM-style persistent latent state rather than a full hidden semi-Markov model with explicit duration distributions.
                """
            )

except Exception as e:
    st.error(f"Error: {e}")
