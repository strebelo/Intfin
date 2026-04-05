import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Price Plan Detector", layout="wide")


# ------------------------------------------------------------
# Model intuition
# ------------------------------------------------------------
# The user supplies an initial plan: regular price R0, sale price S0,
# and probabilities of observing regular / sale / other.
#
# The program then searches over a set of candidate plans extracted from
# the data and estimates the most likely sequence of plans with a simple,
# transparent HMM-style filter/smoother:
#
# - latent state k = price plan (R_k, S_k)
# - observed price at time t is generated from:
#       regular with prob p_reg
#       sale    with prob p_sale
#       other   with prob p_other
# - plans are persistent: high probability of staying in the same plan
#
# This is not yet a full hidden semi-Markov model, but it is streamlit-
# friendly, easy to initialize, and already handles rare "other" prices.
#
# I structured the code so you can later replace the transition matrix or
# the Viterbi routine with a true HSMM if you want explicit durations.
# ------------------------------------------------------------


@dataclass
class Plan:
    regular: float
    sale: float

    def label(self) -> str:
        return f"R={self.regular:g}, S={self.sale:g}"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def normalize_probs(p_reg: float, p_sale: float, p_other: float) -> Tuple[float, float, float]:
    total = p_reg + p_sale + p_other
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive number.")
    return p_reg / total, p_sale / total, p_other / total


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return df


def validate_input_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "time" not in cols or "price" not in cols:
        raise ValueError("The data must contain columns named 'time' and 'price'.")

    out = df[[cols["time"], cols["price"]]].copy()
    out.columns = ["time", "price"]
    out = out.sort_values("time").reset_index(drop=True)

    if out["time"].duplicated().any():
        raise ValueError("The 'time' column must not contain duplicates.")
    if out[["time", "price"]].isna().any().any():
        raise ValueError("The columns 'time' and 'price' must not contain missing values.")

    return out


def build_candidate_plans(
    prices: np.ndarray,
    initial_regular: float,
    initial_sale: float,
    max_plans: int,
) -> List[Plan]:
    unique_prices = sorted(pd.Series(prices).dropna().unique())

    # Include the user-supplied initial plan no matter what.
    plans = {(float(initial_regular), float(initial_sale))}

    # Candidate prices are the most common observed prices.
    counts = pd.Series(prices).value_counts()
    common_prices = list(counts.index[: min(len(counts), 8)])

    # Add all pairs with regular > sale among common prices.
    for r, s in itertools.permutations(common_prices, 2):
        if r > s:
            plans.add((float(r), float(s)))

    # Fallback: if data are very sparse, use all unique prices.
    if len(plans) < 2:
        for r, s in itertools.permutations(unique_prices, 2):
            if r > s:
                plans.add((float(r), float(s)))

    plan_objs = [Plan(r, s) for r, s in sorted(plans, key=lambda x: (x[0], x[1]))]

    # Keep the initial plan plus the most plausible others.
    def score(plan: Plan) -> float:
        return float(((prices == plan.regular) | (prices == plan.sale)).sum())

    plan_objs = sorted(plan_objs, key=score, reverse=True)

    # Force initial plan to be included and first.
    initial = Plan(float(initial_regular), float(initial_sale))
    dedup: Dict[Tuple[float, float], Plan] = {(initial.regular, initial.sale): initial}
    for p in plan_objs:
        dedup[(p.regular, p.sale)] = p

    ordered = [initial] + [
        p for key, p in dedup.items()
        if key != (initial.regular, initial.sale)
    ]

    return ordered[:max_plans]


def emission_prob(
    observed_price: float,
    plan: Plan,
    p_reg: float,
    p_sale: float,
    p_other: float,
    other_support: int,
) -> float:
    # Discrete emission:
    # - exact regular price gets p_reg
    # - exact sale price gets p_sale
    # - any other price shares p_other evenly across remaining support
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


def viterbi_decode(
    prices: np.ndarray,
    plans: List[Plan],
    p_reg: float,
    p_sale: float,
    p_other: float,
    stay_prob: float,
    init_state_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    k = len(plans)

    unique_prices = np.unique(prices)
    other_support = max(len(unique_prices) - 2, 1)

    trans = build_transition_matrix(k, stay_prob)
    log_trans = np.log(np.clip(trans, 1e-14, None))

    init_probs = np.full(k, 1e-8)
    init_probs[init_state_index] = 1.0
    init_probs = init_probs / init_probs.sum()
    log_init = np.log(np.clip(init_probs, 1e-14, None))

    log_emit = np.zeros((n, k))
    for t in range(n):
        for j, plan in enumerate(plans):
            prob = emission_prob(prices[t], plan, p_reg, p_sale, p_other, other_support)
            log_emit[t, j] = np.log(max(prob, 1e-14))

    dp = np.full((n, k), -np.inf)
    ptr = np.zeros((n, k), dtype=int)

    dp[0] = log_init + log_emit[0]

    for t in range(1, n):
        for j in range(k):
            candidates = dp[t - 1] + log_trans[:, j]
            ptr[t, j] = int(np.argmax(candidates))
            dp[t, j] = candidates[ptr[t, j]] + log_emit[t, j]

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]

    return states, dp


def summarize_results(df: pd.DataFrame, states: np.ndarray, plans: List[Plan]) -> pd.DataFrame:
    out = df.copy()
    out["state"] = states
    out["regular_plan"] = [plans[s].regular for s in states]
    out["sale_plan"] = [plans[s].sale for s in states]
    out["plan_label"] = [plans[s].label() for s in states]

    def obs_type(row) -> str:
        p = row["price"]
        r = row["regular_plan"]
        s = row["sale_plan"]
        if p == r:
            return "regular"
        if p == s:
            return "sale"
        return "other"

    out["observation_type"] = out.apply(obs_type, axis=1)
    return out


def plot_price_plan(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(results["time"], results["price"], marker="o", linewidth=1.8, label="Observed price")
    ax.plot(results["time"], results["regular_plan"], linestyle="--", linewidth=2, label="Detected regular price")
    ax.plot(results["time"], results["sale_plan"], linestyle=":", linewidth=2.4, label="Detected sale price")

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
            "n_other_prices": int((chunk["observation_type"] == "other").sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("Price plan detector")
st.write(
    "Upload a CSV with columns **time** and **price**, provide an initial price plan, "
    "and the app will infer the most likely sequence of regular/sale plans."
)

with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.subheader("Initial plan")
    initial_regular = st.number_input("Initial regular price", value=10.0, step=1.0)
    initial_sale = st.number_input("Initial sale price", value=5.0, step=1.0)

    st.subheader("Emission probabilities")
    p_reg = st.number_input("P(regular)", min_value=0.0, value=0.45, step=0.05)
    p_sale = st.number_input("P(sale)", min_value=0.0, value=0.45, step=0.05)
    p_other = st.number_input("P(other)", min_value=0.0, value=0.10, step=0.05)

    st.subheader("Plan persistence")
    stay_prob = st.slider("Probability current plan stays active", 0.50, 0.999, 0.92, 0.01)
    max_plans = st.slider("Maximum number of candidate plans", 1, 10, 5, 1)

    run_button = st.button("Detect price plan", type="primary")


# ------------------------------------------------------------
# Default example if no file is uploaded
# ------------------------------------------------------------
def default_example() -> pd.DataFrame:
    data = {
        "time": list(range(1, 53)),
        "price": [
            10, 10, 5, 5, 10, 7, 10, 5, 10, 11, 6, 6, 6, 11, 7, 6, 6, 11,
            10, 5, 5, 7, 5, 10, 12, 12, 12, 12, 6, 6, 12, 6, 7, 8, 6, 12,
            12, 6, 12, 6, 12, 6, 12, 12, 7, 13, 12, 12, 12, 6, 6, 6,
        ],
    }
    return pd.DataFrame(data)


try:
    if uploaded_file is not None:
        df_raw = load_csv(uploaded_file)
        df = validate_input_df(df_raw)
    else:
        st.info("No file uploaded yet. Showing the built-in example data.")
        df = default_example()

    st.subheader("Input data")
    st.dataframe(df, use_container_width=True)

    if run_button or uploaded_file is None:
        p_reg_n, p_sale_n, p_other_n = normalize_probs(p_reg, p_sale, p_other)

        if initial_regular <= initial_sale:
            st.error("The initial regular price must be strictly greater than the sale price.")
            st.stop()

        prices = df["price"].to_numpy(dtype=float)
        plans = build_candidate_plans(prices, initial_regular, initial_sale, max_plans)
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

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="detected_price_plan.csv",
            mime="text/csv",
        )

        with st.expander("Notes"):
            st.markdown(
                """
                **How the app works**

                - Each hidden state is a candidate price plan with a regular price and a sale price.
                - The app favors staying in the current plan unless the data provide enough evidence to switch.
                - A price that matches neither the regular nor the sale price is labeled **other**.

                **Current limitation**

                This version uses an HMM-style persistence rule rather than a full hidden semi-Markov duration model.
                That keeps it fast and easy to tune in Streamlit. A natural next step would be to add explicit duration distributions.
                """
            )

except Exception as e:
    st.error(f"Error: {e}")
