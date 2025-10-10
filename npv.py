# npv.py
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interest Rates & Present Value", page_icon="📉", layout="wide")

# ---------- Core Finance Helpers ----------
def discount_factors(r: float, n: int) -> np.ndarray:
    """Per-period discount factors for periods 1..n at rate r."""
    k = np.arange(1, n + 1)
    return 1.0 / (1.0 + r) ** k

def pv_from_cf(cashflows: np.ndarray, r: float) -> float:
    n = len(cashflows)
    dfs = discount_factors(r, n)
    return float(np.dot(cashflows, dfs))

def macaulay_duration(cashflows: np.ndarray, r: float) -> float:
    """
    Macaulay duration in periods.
    D_M = sum[t * PV(CF_t)] / Price
    """
    n = len(cashflows)
    dfs = discount_factors(r, n)
    pv_cf = cashflows * dfs
    k = np.arange(1, n + 1)
    price = pv_cf.sum()
    if price <= 0:
        return np.nan
    return float(np.dot(k, pv_cf) / price)

def modified_duration(cashflows: np.ndarray, r: float) -> float:
    """Modified duration D_mod = D_M / (1+r)."""
    DM = macaulay_duration(cashflows, r)
    if math.isnan(DM):
        return np.nan
    return DM / (1.0 + r)

def convexity(cashflows: np.ndarray, r: float) -> float:
    """
    Discrete-time convexity measure:
    C = [ sum( t(t+1) * CF_t / (1+r)^(t+2) ) ] / Price
    """
    n = len(cashflows)
    k = np.arange(1, n + 1)
    denom = (1.0 + r) ** (k + 2)
    weighted = cashflows * (k * (k + 1)) / denom
    price = pv_from_cf(cashflows, r)
    if price <= 0:
        return np.nan
    return float(weighted.sum() / price)

# ---------- Cash Flow Generators ----------
def cf_annuity(n: int, c1: float) -> np.ndarray:
    """Level cash flow C each period."""
    return np.full(n, c1, dtype=float)

def cf_growing_annuity(n: int, c1: float, g: float) -> np.ndarray:
    """Growing cash flow: C1, C1(1+g), ..., for n periods."""
    k = np.arange(n)
    return c1 * (1.0 + g) ** k

def cf_front_loaded(n: int, c1: float) -> np.ndarray:
    """Big early flows, tapering later."""
    # e.g., geometric decay
    base = np.linspace(1.0, 0.3, n)
    base = base / base.sum()
    total = c1 * n  # scale so first equals ~c1-ish overall
    return total * base

def cf_back_loaded(n: int, c1: float) -> np.ndarray:
    """Small early flows, big later flows."""
    base = np.linspace(0.3, 1.0, n)
    base = base / base.sum()
    total = c1 * n
    return total * base

def cf_lump_sum(n: int, cT: float) -> np.ndarray:
    """Single payoff in period n."""
    arr = np.zeros(n, dtype=float)
    if n > 0:
        arr[-1] = cT
    return arr

def build_cashflows(pattern: str, n: int, c1: float, g: float, terminal: float) -> np.ndarray:
    if pattern == "Level annuity":
        cf = cf_annuity(n, c1)
    elif pattern == "Growing annuity":
        cf = cf_growing_annuity(n, c1, g)
    elif pattern == "Front-loaded":
        cf = cf_front_loaded(n, c1)
    elif pattern == "Back-loaded":
        cf = cf_back_loaded(n, c1)
    elif pattern == "Lump sum at T":
        cf = cf_lump_sum(n, max(c1, terminal if terminal > 0 else c1 * n))
    else:
        cf = cf_annuity(n, c1)

    if terminal > 0 and pattern not in ["Lump sum at T"]:
        cf[-1] += terminal
    return cf

# ---------- UI ----------
st.title("📉 How Interest Rates Affect Present Value")
st.caption("Interactive exploration of discount rate sensitivity, duration, and convexity.")

with st.sidebar:
    st.header("Global Settings")
    r_base = st.slider("Base discount rate r (per period)", min_value=0.0, max_value=0.20, value=0.05, step=0.005, format="%.3f")
    r_shock = st.slider("Add a rate shock Δr", min_value=0.0, max_value=0.20, value=0.03, step=0.005, format="%.3f")
    r_after = r_base + r_shock
    st.write(f"**New rate:** {r_after:.3%}")

    n = st.slider("Horizon (number of periods)", min_value=3, max_value=50, value=10, step=1)
    st.markdown("---")
    st.subheader("Visualization Range")
    r_min = st.slider("Min rate for PV curve", 0.0, 0.20, 0.00, 0.005, format="%.3f")
    r_max = st.slider("Max rate for PV curve", 0.02, 0.50, 0.20, 0.005, format="%.3f")
    r_points = st.slider("Points in PV curve", 25, 400, 150, 25)

st.markdown("### Define Projects")
colA, colB = st.columns(2)

def project_panel(col, label_prefix="Project A"):
    with col:
        st.subheader(label_prefix)
        pattern = st.selectbox(
            "Cash flow pattern",
            ["Level annuity", "Growing annuity", "Front-loaded", "Back-loaded", "Lump sum at T"],
            key=f"pattern_{label_prefix}"
        )
        c1 = st.number_input("Initial cash flow (C₁) or scale", min_value=0.0, value=100.0, step=10.0, key=f"c1_{label_prefix}")
        g = st.slider("Growth rate g (for growing annuity)", 0.0, 0.50, 0.05, 0.01, key=f"g_{label_prefix}")
        terminal = st.number_input("Optional terminal value (added in final period)", min_value=0.0, value=0.0, step=50.0, key=f"tv_{label_prefix}")
        cf = build_cashflows(pattern, n, c1, g, terminal)
        df = pd.DataFrame({"Period": np.arange(1, n + 1), "Cash Flow": cf})
        return {"name": label_prefix, "pattern": pattern, "c1": c1, "g": g, "terminal": terminal, "cf": cf, "df": df}

projA = project_panel(colA, "Project A")
projB = project_panel(colB, "Project B")

# ---------- Computations ----------
def project_metrics(cf: np.ndarray, r: float) -> Tuple[float, float, float]:
    price = pv_from_cf(cf, r)
    D_mod = modified_duration(cf, r)
    C = convexity(cf, r)
    return price, D_mod, C

pvA_base, durA, convA = project_metrics(projA["cf"], r_base)
pvA_after, _, _ = project_metrics(projA["cf"], r_after)

pvB_base, durB, convB = project_metrics(projB["cf"], r_base)
pvB_after, _, _ = project_metrics(projB["cf"], r_after)

def pct_change(new, old):
    if old == 0:
        return np.nan
    return (new - old) / old

dropA = pct_change(pvA_after, pvA_base)
dropB = pct_change(pvB_after, pvB_base)

# Linear/quadratic approximation using duration/convexity
def dv_over_v_approx(D_mod: float, C: float, r0: float, dr: float):
    # ΔV/V ≈ -D_mod * Δr + 0.5 * C * (Δr)^2
    return -D_mod * dr + 0.5 * C * (dr ** 2)

approxA = dv_over_v_approx(durA, convA, r_base, r_shock)
approxB = dv_over_v_approx(durB, convB, r_base, r_shock)

# ---------- Display: Key Numbers ----------
st.markdown("### Results")
kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
kcol1.metric("r (base)", f"{r_base:.2%}")
kcol2.metric("Δr (shock)", f"{r_shock:.2%}")
kcol3.metric("r (after)", f"{r_after:.2%}")
kcol4.metric("Periods", f"{n}")
kcol5.metric("PV curve rates", f"{r_min:.2%} → {r_max:.2%}")

def pretty_metrics(title, pv0, pv1, drop, D, C, approx):
    st.markdown(f"#### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("PV at base r", f"{pv0:,.2f}")
    c2.metric("PV after shock", f"{pv1:,.2f}")
    c3.metric("Change", f"{drop:+.2%}")
    st.caption(f"Modified duration: **{D:.2f}** periods | Convexity: **{C:.2f}** | Duration/convexity approx ΔV/V ≈ **{approx:+.2%}**")

pretty_metrics("Project A", pvA_base, pvA_after, dropA, durA, convA, approxA)
pretty_metrics("Project B", pvB_base, pvB_after, dropB, durB, convB, approxB)

# Teaching hint
with st.expander("Why back-loaded projects fall more when rates rise"):
    st.write(
        "Projects with later cash flows have **higher duration**—their value is "
        "more sensitive to the discount rate. A rate increase shrinks the present "
        "value of far-future cash flows by more, so **back-loaded** profiles typically "
        "drop more than **front-loaded** ones."
    )

# ---------- Plots ----------
tab1, tab2, tab3 = st.tabs(["Cash Flows", "PV vs Discount Rate", "Duration & Convexity"])

with tab1:
    st.markdown("#### Cash Flow Profiles")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(projA["df"]["Period"] - 0.2, projA["df"]["Cash Flow"], width=0.4, label="Project A")
    ax.bar(projB["df"]["Period"] + 0.2, projB["df"]["Cash Flow"], width=0.4, label="Project B")
    ax.set_xlabel("Period")
    ax.set_ylabel("Cash Flow")
    ax.set_title("Cash Flow Timing: Front vs Back Loaded")
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

with tab2:
    st.markdown("#### PV Sensitivity to the Discount Rate")
    rates = np.linspace(max(1e-9, r_min), max(r_min + 1e-6, r_max), r_points)
    pvA_curve = [pv_from_cf(projA["cf"], r) for r in rates]
    pvB_curve = [pv_from_cf(projB["cf"], r) for r in rates]

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(rates, pvA_curve, label="Project A PV(r)")
    ax2.plot(rates, pvB_curve, label="Project B PV(r)")
    ax2.axvline(r_base, linestyle="--", linewidth=1)
    ax2.axvline(r_after, linestyle="--", linewidth=1)
    ax2.set_xlabel("Discount rate r")
    ax2.set_ylabel("Present Value")
    ax2.set_title("PV declines as rates rise (curvature shows convexity)")
    ax2.legend(loc="upper right")
    st.pyplot(fig2, clear_figure=True)

with tab3:
    st.markdown("#### Duration & Convexity Illustration")
    st.write(
        "Below, we approximate the PV change using duration and convexity. The dots show exact PVs; "
        "the dashed lines show the local duration-based tangent at the base rate."
    )

    # Tangent lines at r_base for both projects
    # PV(r) ≈ PV0 * [1 - D_mod*(r-r0) + 0.5*C*(r-r0)^2]
    def approx_curve(P0, Dm, Cc, r0, rr: np.ndarray):
        dr = rr - r0
        return P0 * (1 - Dm * dr + 0.5 * Cc * (dr ** 2))

    pvA_approx = approx_curve(pvA_base, durA, convA, r_base, rates)
    pvB_approx = approx_curve(pvB_base, durB, convB, r_base, rates)

    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(rates, pvA_curve, label="A exact PV(r)")
    ax3.plot(rates, pvA_approx, linestyle="--", label="A duration/convexity approx")
    ax3.plot(rates, pvB_curve, label="B exact PV(r)")
    ax3.plot(rates, pvB_approx, linestyle="--", label="B duration/convexity approx")
    ax3.scatter([r_base, r_after], [pvA_base, pvA_after], s=30)
    ax3.scatter([r_base, r_after], [pvB_base, pvB_after], s=30)
    ax3.set_xlabel("Discount rate r")
    ax3.set_ylabel("Present Value")
    ax3.set_title("Duration (slope) and Convexity (curvature) at r = base")
    ax3.legend(loc="upper right")
    st.pyplot(fig3, clear_figure=True)

# ---------- Classroom Prompts ----------
with st.expander("Classroom prompts & takeaways"):
    st.markdown(
        """
- **Experiment**: Keep the same horizon and scale; switch Project A to **Front-loaded** and Project B to **Back-loaded**.  
  Increase Δr and observe which PV drops more.
- **Grow vs Level**: Compare a **Growing annuity** to a **Level annuity**—g pushes cash flows later, raising duration.
- **Approximation quality**: Contrast exact PV change with the **duration/convexity** approximation as Δr grows.
- **Key lesson**: The **timing** of cash flows drives rate sensitivity. Back-loaded = higher duration ⇒ bigger PV impact.
        """
    )

st.markdown("---")
st.caption("Tip: Use the Back-loaded vs Front-loaded presets and vary Δr to make the effect jump off the page.")

