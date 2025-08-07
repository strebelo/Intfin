
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Netflix FX Hedging Simulator", layout="centered")

st.title("📺 Netflix FX Hedging Simulator")
st.markdown("""
Simulate how different foreign exchange (FX) hedging strategies impact Netflix's USD-denominated profits. 
This tool is designed to help students understand exchange rate risk and hedging trade-offs.
""")

# Input Parameters
fx_today = st.slider("📈 Current EUR/USD exchange rate", 0.9, 1.3, 1.10, step=0.01)
fx_future = st.slider("🔮 Future EUR/USD exchange rate", 0.8, 1.4, 1.00, step=0.01)
revenue_foreign_pct = st.slider("🌍 % of Revenue from Foreign Markets", 0.0, 1.0, 0.6, step=0.05)
costs_usd_pct = st.slider("💵 % of Costs in USD", 0.0, 1.0, 0.9, step=0.05)
hedge = st.selectbox("🛡️ Hedging Strategy", ["No hedge", "Forward", "Put Option", "Quote in USD", "Move production overseas"])

# Simulated base numbers
total_revenue = 1000  # in foreign currency
total_costs = 700     # in USD

# Revenue conversion based on hedging
if hedge == "No hedge":
    revenue_usd = revenue_foreign_pct * total_revenue * fx_future + (1 - revenue_foreign_pct) * total_revenue
elif hedge == "Forward":
    revenue_usd = revenue_foreign_pct * total_revenue * fx_today + (1 - revenue_foreign_pct) * total_revenue
elif hedge == "Put Option":
    fx_min = 1.05  # strike
    effective_fx = max(fx_future, fx_min)
    revenue_usd = revenue_foreign_pct * total_revenue * effective_fx + (1 - revenue_foreign_pct) * total_revenue
elif hedge == "Quote in USD":
    revenue_usd = total_revenue  # assume full pass-through
elif hedge == "Move production overseas":
    revenue_usd = revenue_foreign_pct * total_revenue * fx_future + (1 - revenue_foreign_pct) * total_revenue
    total_costs = 500  # reduce USD costs

# Profit
profit_usd = revenue_usd - total_costs

st.metric("💰 Profit in USD", f"${profit_usd:,.2f}")

# Optional: simulation
simulate = st.checkbox("Simulate FX uncertainty")

if simulate:
    fx_scenarios = np.random.normal(loc=fx_future, scale=0.05, size=1000)
    profits = []
    for fx in fx_scenarios:
        if hedge == "No hedge":
            r = revenue_foreign_pct * total_revenue * fx + (1 - revenue_foreign_pct) * total_revenue
            c = total_costs
        elif hedge == "Forward":
            r = revenue_foreign_pct * total_revenue * fx_today + (1 - revenue_foreign_pct) * total_revenue
            c = total_costs
        elif hedge == "Put Option":
            r = revenue_foreign_pct * total_revenue * max(fx, fx_min) + (1 - revenue_foreign_pct) * total_revenue
            c = total_costs
        elif hedge == "Quote in USD":
            r = total_revenue
            c = total_costs
        elif hedge == "Move production overseas":
            r = revenue_foreign_pct * total_revenue * fx + (1 - revenue_foreign_pct) * total_revenue
            c = 500
        profits.append(r - c)

    st.write(f"📊 Expected Profit: ${np.mean(profits):,.2f}")
    st.write(f"📉 Profit Volatility (Std Dev): ${np.std(profits):,.2f}")

    fig, ax = plt.subplots()
    ax.hist(profits, bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Profit Distribution Under FX Uncertainty")
    ax.set_xlabel("Profit (USD)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
