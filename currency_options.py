# app.py
# ------------------------------------------------------------
# currency_options.py
# ------------------------------------------------------------
# Features:
# - Sidebar controls: S0, K, T, sigma, r_d, r_f, call/put, quote convention
# - Report BOTH call and put prices for the current inputs
# - Plot price vs. strike over [0.8*K, 1.2*K] holding others fixed
# - Plot price vs. volatility over [0.8*sigma, 1.2*sigma] holding others fixed
#
# Notes:
# - Rates r_d (domestic) and r_f (foreign) are CONTINUOUSLY COMPOUNDED per year
# - Time T in years
# - Prices are reported in the DOMESTIC currency of the chosen quote convention
#   (for DC/FC, domestic=DC; for FC/DC, domestic=FC)
# ------------------------------------------------------------

import numpy as np
import streamlit as st
from math import log, sqrt, exp, isfinite
from scipy.stats import norm
import matplotlib.pyplot as plt

# ------------------------------
# Pricing: Garman–Kohlhagen (FX Black–Scholes)
# ------------------------------
def gk_price(S0, K, T, sigma, r_d, r_f, is_call=True):
    """
    Garman–Kohlhagen price of a European FX option.

    Parameters
    ----------
    S0 : float
        Spot exchange rate (price of 1 unit of *foreign* currency in domestic currency units).
        If quote convention is FC/DC, interpret S0 accordingly (domestic = FC).
    K : float
        Strike (in same quote as S0).
    T : float
        Time to maturity in years.
    sigma : float
        Volatility (annualized, in decimals, e.g., 0.20 for 20%).
    r_d : float
        Domestic continuously-compounded risk-free rate.
    r_f : float
        Foreign continuously-compounded risk-free rate.
    is_call : bool
        True for call, False for put.

    Returns
    -------
    float
        Option price in domestic currency.
    """
    # Edge cases: handle T ~ 0 or sigma ~ 0 with discounted intrinsic value
    eps = 1e-12
    T = max(T, 0.0)
    sigma = max(sigma, 0.0)

    if T < eps or sigma < eps:
        # Forward-like intrinsic value at expiry, discounted appropriately
        # Payout at T: call: max(S_T - K, 0), put: max(K - S_T, 0)
        # With no diffusion or zero time, approximate with current spot:
        intrinsic = max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)
        # For immediate expiry, discounting is negligible; keep it as intrinsic
        return intrinsic

    sqrtT = sqrt(T)
    try:
        d1 = (log(S0 / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
    except ValueError:
        # log domain error (e.g., non-positive inputs)
        return np.nan

    if is_call:
        return S0 * exp(-r_f * T) * norm.cdf(d1) - K * exp(-r_d * T) * norm.cdf(d2)
    else:
        return K * exp(-r_d * T) * norm.cdf(-d2) - S0 * exp(-r_f * T) * norm.cdf(-d1)

def forward_rate(S0, T, r_d, r_f):
    """Covered Interest Parity (continuous compounding)."""
    return S0 * exp((r_d - r_f) * T)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="FX Options (Garman–Kohlhagen)", layout="wide")

st.title("FX Options Pricing — Garman–Kohlhagen (Black–Scholes for Currencies)")

with st.sidebar:
    st.header("Inputs")

    quote = st.selectbox(
        "Quote convention",
        options=["DC per 1 FC (DC/FC)", "FC per 1 DC (FC/DC)"],
        index=0,
        help=("Determines which currency is 'domestic' for pricing.\n"
              "Prices are reported in the domestic currency of the chosen convention.")
    )

    # Labels adjust to remind which is domestic vs foreign under the convention
    if "DC/FC" in quote:
        dom_lbl = "Domestic (first) currency"
        for_lbl = "Foreign (second) currency"
    else:
        dom_lbl = "Domestic (first) currency = FC"
        for_lbl = "Foreign (second) currency = DC"

    st.caption(f"Domestic = first currency in the quote; Foreign = second.  \n"
               f"Here: **{dom_lbl}**, **{for_lbl}**.")

    col1, col2 = st.columns(2)
    with col1:
        S0 = st.number_input("Spot S₀", min_value=1e-12, value=1.1000, step=0.005, format="%.6f",
                             help="Exchange rate in the selected convention.")
    with col2:
        K = st.number_input("Strike K", min_value=1e-12, value=1.1000, step=0.005, format="%.6f")

    col3, col4, col5 = st.columns(3)
    with col3:
        T = st.number_input("Maturity T (years)", min_value=0.0, value=1.00, step=0.25)
    with col4:
        sigma = st.number_input("Volatility σ (decimal, e.g., 0.20)", min_value=0.0, value=0.20, step=0.01, format="%.4f")
    with col5:
        opt_type = st.radio("Highlight option", options=["Call", "Put"], horizontal=True)

    col6, col7 = st.columns(2)
    with col6:
        r_d = st.number_input("Domestic rate r_d (cont. comp.)", value=0.03, step=0.005, format="%.4f")
    with col7:
        r_f = st.number_input("Foreign rate r_f (cont. comp.)", value=0.01, step=0.005, format="%.4f")

# Compute current prices
call_price = gk_price(S0, K, T, sigma, r_d, r_f, is_call=True)
put_price  = gk_price(S0, K, T, sigma, r_d, r_f, is_call=False)
F0 = forward_rate(S0, T, r_d, r_f)

# ------------------------------
# Price Summary
# ------------------------------
st.subheader("Price Summary (Domestic currency)")

colA, colB, colC = st.columns([1, 1, 1.2])
with colA:
    st.metric(label="Call Price", value=f"{call_price:,.6f}")
with colB:
    st.metric(label="Put Price", value=f"{put_price:,.6f}")
with colC:
    st.metric(label="Forward F₀", value=f"{F0:,.6f}",
              help="Covered Interest Parity: F₀ = S₀ · exp((r_d − r_f)·T)")

st.caption(
    "Model: C = S₀·e^{-r_f T}·N(d₁) − K·e^{-r_d T}·N(d₂),  "
    "P = K·e^{-r_d T}·N(−d₂) − S₀·e^{-r_f T}·N(−d₁).  "
    "Rates are continuously compounded; prices in the domestic currency of the selected quote."
)

# ------------------------------
# Plot 1: Prices vs Strike
# ------------------------------
st.subheader("Prices vs. Strike (holding S₀, σ, T, r_d, r_f fixed)")
K_low, K_high = 0.8 * K, 1.2 * K
Ks = np.linspace(K_low, K_high, 101)

call_vsK = np.array([gk_price(S0, k, T, sigma, r_d, r_f, True) for k in Ks])
put_vsK  = np.array([gk_price(S0, k, T, sigma, r_d, r_f, False) for k in Ks])

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(Ks, call_vsK, label="Call")
ax1.plot(Ks, put_vsK,  label="Put")
ax1.axvline(K, linestyle="--", linewidth=1)
ax1.set_xlabel("Strike K")
ax1.set_ylabel("Price (domestic)")
ax1.set_title("Option Prices vs. Strike")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1, clear_figure=True)

# ------------------------------
# Plot 2: Prices vs Volatility
# ------------------------------
st.subheader("Prices vs. Volatility (holding S₀, K, T, r_d, r_f fixed)")
sig_low, sig_high = max(0.0, 0.8 * sigma), 1.2 * sigma if sigma > 0 else 0.4
# Ensure a sensible range when sigma=0
if abs(sig_high - sig_low) < 1e-9:
    sig_low, sig_high = 0.01, 0.50
sigmas = np.linspace(sig_low, sig_high, 101)

call_vsSig = np.array([gk_price(S0, K, T, s, r_d, r_f, True) for s in sigmas])
put_vsSig  = np.array([gk_price(S0, K, T, s, r_d, r_f, False) for s in sigmas])

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(sigmas, call_vsSig, label="Call")
ax2.plot(sigmas, put_vsSig,  label="Put")
ax2.axvline(sigma, linestyle="--", linewidth=1)
ax2.set_xlabel("Volatility σ")
ax2.set_ylabel("Price (domestic)")
ax2.set_title("Option Prices vs. Volatility")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2, clear_figure=True)

# ------------------------------
# Optional: highlight the selected type in text
# ------------------------------
if opt_type == "Call":
    st.info(f"Highlighted: **Call** at K={K:.6f}, σ={sigma:.4f} → Price = {call_price:,.6f}")
else:
    st.info(f"Highlighted: **Put** at K={K:.6f}, σ={sigma:.4f} → Price = {put_price:,.6f}")

# Footer
st.caption(
    "This app is intended for teaching purposes. "
    "Forwards use covered interest parity under continuous compounding. "
    "European options; no dividends beyond foreign rate carry."
)
