# currency_options_basic.py
# ------------------------------------------------------------
# FX Options (Garman–Kohlhagen) — Teaching App for Streamlit
# ------------------------------------------------------------
# Features:
# - Sidebar controls: S₀, K, T, σ, r_d, r_f, quote convention
# - Report BOTH call and put prices for the current inputs
# - Plot price vs. strike over [0.8*K, 1.2*K]
# - Plot price vs. volatility over [0.8*σ, 1.2*σ]
#
# Notes:
# - r_d and r_f are CONTINUOUSLY COMPOUNDED per year
# - T in years
# - Prices are reported in the DOMESTIC currency of the chosen quote convention
# ------------------------------------------------------------

import numpy as np
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt

# ------------------------------
# Pricing: Garman–Kohlhagen (FX Black–Scholes)
# ------------------------------
def gk_price(S0, K, T, sigma, r_d, r_f, is_call=True):
    eps = 1e-12
    T = max(T, 0.0)
    sigma = max(sigma, 0.0)

    if T < eps or sigma < eps:
        intrinsic = max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)
        return intrinsic

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if is_call:
        return S0 * exp(-r_f * T) * norm.cdf(d1) - K * exp(-r_d * T) * norm.cdf(d2)
    else:
        return K * exp(-r_d * T) * norm.cdf(-d2) - S0 * exp(-r_f * T) * norm.cdf(-d1)

def forward_rate(S0, T, r_d, r_f):
    return S0 * exp((r_d - r_f) * T)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="FX Options", layout="wide")
st.title("Currency Options Pricing (Black–Scholes)")

with st.sidebar:
    st.header("Inputs")

    quote = st.selectbox(
        "Quote convention",
        options=["Domestic/Foreign", "Foreign/Domestic"],
        index=0,
        help="Prices are reported in the domestic currency of the chosen convention."
    )

    if "DC/FC" in quote:
        dom_lbl = "Domestic (first) currency"
        for_lbl = "Foreign (second) currency"
    else:
        dom_lbl = "Domestic (first) currency = FC"
        for_lbl = "Foreign (second) currency = DC"

    c1, c2 = st.columns(2)
    with c1:
        S0 = st.number_input("Spot $S_0$", min_value=1e-12, value=1.1000, step=0.005, format="%.6f")
    with c2:
        K = st.number_input("Strike $K$", min_value=1e-12, value=1.1000, step=0.005, format="%.6f")

    c3, c4, _ = st.columns(3)
    with c3:
        T = st.number_input("Maturity $T$ (years)", min_value=0.0, value=1.00, step=0.25)
    with c4:
        sigma = st.number_input("Volatility $\\sigma$ (e.g., 0.1)", min_value=0.0, value=0.10, step=0.01, format="%.4f")

    c5, c6 = st.columns(2)
    with c5:
        r_d = st.number_input("Domestic rate $r_d$ (cont. comp.)", value=0.03, step=0.005, format="%.4f")
    with c6:
        r_f = st.number_input("Foreign rate $r_f$ (cont. comp.)", value=0.01, step=0.005, format="%.4f")

# Compute prices
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
    st.metric(label="Forward $F_0$", value=f"{F0:,.6f}",
              help="Covered Interest Parity: $F_0 = S_0 \\cdot e^{(r_d - r_f)T}$")

st.caption(
    r"""
    Model equations:  
    $$
    C = S_0 e^{-r_f T} N(d_1) - K e^{-r_d T} N(d_2), \quad
    P = K e^{-r_d T} N(-d_2) - S_0 e^{-r_f T} N(-d_1)
    $$
    where  
    $$
    d_1 = \frac{\ln(S_0/K) + (r_d - r_f + \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad
    d_2 = d_1 - \sigma \sqrt{T}
    $$
    """
)



# Footer
st.caption("This app is designed for teaching purposes")
