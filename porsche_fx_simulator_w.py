import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Porsche FX Simulator (with w slider)", layout="wide")

st.title("Porsche FX Simulator — Blended Pricing with w")
st.caption("Academic convention: **EUR/USD = euros per 1 USD**. Demand is linear in the USD price. "
           "When **w = 0**, price is fixed in USD. When **w = 1**, price is fixed in EUR. "
           "For 0 < w < 1, the USD price is a weighted average of the USD sticker and the EUR sticker converted at spot.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Inputs")
    # Exchange rate parameters (academic convention: EUR/USD)
    s0 = st.number_input("Current EUR/USD exchange rate (euros per USD)", value=0.90, min_value=0.01, step=0.01, format="%.4f")
    vol = st.number_input("Volatility of log(EUR/USD)", value=0.10, min_value=0.0, step=0.01, format="%.4f")
    n_simulations = st.number_input("Number of simulations", value=5000, min_value=100, step=100)

    st.markdown("---")
    st.subheader("Pricing")
    usd_price = st.number_input("USD sticker price (used when w = 0)", value=100000.0, min_value=0.0, step=1000.0, format="%.2f")
    euro_price = st.number_input("EUR sticker price (used when w = 1)", value=90000.0, min_value=0.0, step=1000.0, format="%.2f")
    w = st.slider("w (0 = fixed USD, 1 = fixed EUR)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    st.markdown("---")
    st.subheader("Costs & Demand")
    cost_eur = st.number_input("Unit cost (EUR)", value=60000.0, min_value=0.0, step=1000.0, format="%.2f")
    a = st.number_input("Demand intercept a", value=1500.0, min_value=0.0, step=10.0, format="%.2f")
    b = st.number_input("Demand slope b (Q = max(a - b * P_usd, 0))", value=0.01, min_value=0.000001, step=0.01, format="%.6f")

# --- Simulation Assumptions ---
st.markdown("### Simulation assumptions")
st.write(
    "- We draw **EUR/USD** as a single lognormal sample around **s0** (no time path), with log-volatility **vol**.\n"
    "- USD price under blending: **P_usd = (1 - w) * USD_sticker + w * (EUR_sticker / (EUR/USD))**.\n"
    "- Quantity: **Q = max(a - b * P_usd, 0)**.\n"
    "- USD revenue: **Rev_usd = P_usd * Q**.\n"
    "- EUR revenue: **Rev_eur = (EUR/USD) * Rev_usd**.\n"
    "- Profit (EUR): **π = Rev_eur - Cost_eur * Q**."
)

# --- Simulate EUR/USD ---
rng = np.random.default_rng()
log_sims = np.log(s0) + rng.normal(0.0, vol, size=int(n_simulations))
sims = np.exp(log_sims)  # EUR/USD draws

# --- USD price under blended sticker (w in [0,1])
usd_prices = (1.0 - w) * usd_price + w * (euro_price / sims)

# --- Demand (linear in USD price)
quantities = np.maximum(a - b * usd_prices, 0.0)

# --- Revenues and profits
usd_sales = usd_prices * quantities
eur_revenue = sims * usd_sales
eur_profit = eur_revenue - cost_eur * quantities

# --- Summary stats helper
def summarize(x):
    x = np.asarray(x)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p5": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x))
    }

# --- Show summary cards
col1, col2, col3, col4 = st.columns(4)
for col, label, arr, unit in [
    (col1, "USD Price", usd_prices, "USD"),
    (col2, "Quantity", quantities, "units"),
    (col3, "USD Sales", usd_sales, "USD"),
    (col4, "Profit (EUR)", eur_profit, "EUR"),
]:
    stats = summarize(arr)
    with col:
        st.metric(f"{label} (mean)", f"{stats['mean']:.2f}")
        st.caption(f"median={stats['median']:.2f} | p5={stats['p5']:.2f} | p95={stats['p95']:.2f} | "
                   f"min={stats['min']:.2f} | max={stats['max']:.2f} ({unit})")

# --- Scatter plots
st.markdown("---")
st.subheader("Sensitivity to EUR/USD")
c1, c2 = st.columns(2)

with c1:
    st.write("**USD Price vs EUR/USD**")
    fig = plt.figure()
    plt.scatter(sims, usd_prices, s=6, alpha=0.5)
    plt.xlabel("EUR/USD (euros per USD)")
    plt.ylabel("USD Price")
    plt.title("USD Price vs EUR/USD")
    st.pyplot(fig, use_container_width=True)

    st.write("**USD Sales vs EUR/USD**")
    fig = plt.figure()
    plt.scatter(sims, usd_sales, s=6, alpha=0.5)
    plt.xlabel("EUR/USD (euros per USD)")
    plt.ylabel("USD Sales")
    plt.title("USD Sales vs EUR/USD")
    st.pyplot(fig, use_container_width=True)

with c2:
    st.write("**Quantity vs EUR/USD**")
    fig = plt.figure()
    plt.scatter(sims, quantities, s=6, alpha=0.5)
    plt.xlabel("EUR/USD (euros per USD)")
    plt.ylabel("Quantity")
    plt.title("Quantity vs EUR/USD")
    st.pyplot(fig, use_container_width=True)

    st.write("**Profit (EUR) vs EUR/USD**")
    fig = plt.figure()
    plt.scatter(sims, eur_profit, s=6, alpha=0.5)
    plt.xlabel("EUR/USD (euros per USD)")
    plt.ylabel("Profit (EUR)")
    plt.title("Profit (EUR) vs EUR/USD")
    st.pyplot(fig, use_container_width=True)

# --- Histograms
st.markdown("---")
st.subheader("Distributions")

dcols = st.columns(4)
plots = [
    ("USD Price", usd_prices, "USD"),
    ("Quantity", quantities, "units"),
    ("USD Sales", usd_sales, "USD"),
    ("Profit (EUR)", eur_profit, "EUR"),
]

for col, (label, arr, unit) in zip(dcols, plots):
    with col:
        fig = plt.figure()
        plt.hist(arr, bins=40, alpha=0.85)
        plt.xlabel(f"{label} ({unit})")
        plt.ylabel("Frequency")
        plt.title(f"{label} Distribution")
        st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.caption("Tip: Set **w** between 0 and 1 to explore partial pass-through of FX into the USD price. "
           "For example, **w = 0.4** means 40% weight on the EUR sticker (converted at spot) and 60% on the fixed USD sticker.")
