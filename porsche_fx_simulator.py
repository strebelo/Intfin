
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Porsche Pricing, Exchange Rates, and Profit Simulator")

# User Inputs
st.sidebar.header("Simulation Parameters")
vol = st.sidebar.slider("Volatility of log exchange rate", 0.01, 0.3, 0.1)
s0 = st.sidebar.number_input("Current EUR/USD exchange rate", value=0.7072)
price_mode = st.sidebar.selectbox("Pricing Strategy", ("Fixed USD price", "Fixed Euro price"))
usd_price = st.sidebar.number_input("USD Price (if Fixed USD price)", value=37200.0)
euro_price = st.sidebar.number_input("Euro Price (if Fixed Euro price)", value=26308.0)
cost_eur = st.sidebar.number_input("Production Cost (EUR)", value=24000.0)
a = st.sidebar.number_input("Demand Parameter a", value=123993.34)
b = st.sidebar.number_input("Demand Parameter b", value=3.064337)
n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)

# Simulate exchange rates using log-random walk
log_sims = np.log(s0) + np.random.normal(0, vol, size=n_simulations)
sims = np.exp(log_sims)

# Calculate results
if price_mode == "Fixed USD price":
    usd_prices = np.full(n_simulations, usd_price)
else:
    usd_prices = euro_price / sims

quantities = np.maximum(a - b * usd_prices, 0)
usd_sales = quantities * usd_prices
euro_profits = sims * usd_prices * quantities - cost_eur * quantities

# Plotting functions
def plot_scatter(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

def plot_hist(data, title, xlabel):
    plt.figure()
    plt.hist(data, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    st.pyplot(plt)

# Scatter plots
plot_scatter(sims, usd_prices, "USD Price vs EUR/USD", "EUR/USD", "USD Price")
plot_scatter(sims, quantities, "Quantity Sold vs EUR/USD", "EUR/USD", "Quantity Sold")
plot_scatter(sims, usd_sales, "USD Sales vs EUR/USD", "EUR/USD", "USD Sales")
plot_scatter(sims, euro_profits, "Euro Profits vs EUR/USD", "EUR/USD", "Euro Profits")

# Histograms
plot_hist(usd_prices, "Histogram of USD Prices", "USD Price")
plot_hist(quantities, "Histogram of Quantities Sold", "Quantity Sold")
plot_hist(usd_sales, "Histogram of USD Sales", "USD Sales")
plot_hist(euro_profits, "Histogram of Euro Profits", "Euro Profits")
