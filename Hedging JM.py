"""
Streamlit app: PLN dividend hedging simulator

Run with:
    streamlit run fx_hedging_streamlit.py

Model summary
-------------
- Dividend is generated evenly from January to December.
- At each month-end, a constant fraction of that month's accrued dividend is
  hedged with a forward and/or a put on PLN.
- Every contract matures on April 30 of the following year.
- The app displays EUR/PLN in the market convention (PLN per EUR), but works
  internally with X = EUR per PLN because that makes the cash-flow formulas
  transparent.
- X follows a driftless geometric random walk under the user's physical
  simulation assumption: dX/X = sigma_realized dW.
- Forwards are priced by covered interest parity.
- PLN puts are priced with the Garman-Kohlhagen FX option formula.
- Realized volatility (used for simulation) and implied volatility (used for
  option pricing) can be set separately.

This is a risk-analysis tool, not trading advice or an accounting valuation.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Core model helpers
# -----------------------------


def norm_cdf(z: float) -> float:
    """Standard normal CDF for scalar z, avoiding a SciPy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class ModelInputs:
    dividend_pln: float
    spot_eurpln: float          # PLN per 1 EUR, e.g. 4.30
    r_eur: float                # decimal, e.g. 0.022
    r_pln: float                # decimal, e.g. 0.039
    sigma_realized: float       # annualized decimal
    sigma_implied: float        # annualized decimal
    n_sims: int
    seed: int
    forward_cost_bps: float     # execution cost, bps of EUR value at hedge date
    option_markup_pct: float    # markup on model option premium, decimal
    strike_pct_of_forward: float


@dataclass
class SimulationOutput:
    hedge_dates: list[date]
    settlement_date: date
    tau_to_settlement: np.ndarray
    x_hedge: np.ndarray         # shape (n_sims, 12), EUR per PLN
    x_terminal: np.ndarray      # shape (n_sims,), EUR per PLN
    payoff_unhedged: np.ndarray
    payoff_forward_100: np.ndarray
    payoff_put_100: np.ndarray
    monthly_reference: pd.DataFrame


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def make_timeline(base_year: int = 2026) -> tuple[date, list[date], date]:
    start = date(base_year, 1, 1)
    hedge_dates = [month_end(base_year, m) for m in range(1, 13)]
    settlement = date(base_year + 1, 4, 30)
    return start, hedge_dates, settlement


@st.cache_data(show_spinner=False)
def simulate_paths(
    spot_eurpln: float,
    sigma_realized: float,
    n_sims: int,
    seed: int,
    base_year: int = 2026,
) -> tuple[np.ndarray, np.ndarray, tuple[date, ...], date]:
    """
    Simulate X = EUR per PLN as a level martingale:
        dX/X = sigma dW
    so log X has drift -0.5*sigma^2.

    Returns X at the 12 hedge dates and at April-30 settlement.
    """
    start, hedge_dates, settlement = make_timeline(base_year)
    target_dates = hedge_dates + [settlement]

    rng = np.random.default_rng(seed)
    x = np.full(n_sims, 1.0 / spot_eurpln, dtype=float)
    values = np.empty((n_sims, len(target_dates)), dtype=float)

    prev = start
    for j, d in enumerate(target_dates):
        dt = (d - prev).days / 365.0
        z = rng.standard_normal(n_sims)
        x *= np.exp(-0.5 * sigma_realized**2 * dt + sigma_realized * math.sqrt(dt) * z)
        values[:, j] = x
        prev = d

    return values[:, :12], values[:, 12], tuple(hedge_dates), settlement


def put_price_per_pln(
    x_spot: np.ndarray,
    tau: float,
    r_eur: float,
    r_pln: float,
    sigma_implied: float,
    strike_pct_of_forward: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Garman-Kohlhagen value of a put on PLN when X = EUR per PLN.

    Strike is set as a fixed percentage of the forward rate:
        K = alpha * F
    This makes d1 and d2 scalar for a given maturity, while price scales with X.

    Returns (premium_per_pln_eur, strike_X, forward_X).
    """
    if tau <= 0:
        raise ValueError("Option maturity must be positive.")

    fwd_x = x_spot * math.exp((r_eur - r_pln) * tau)
    alpha = strike_pct_of_forward
    strike_x = alpha * fwd_x

    if sigma_implied <= 1e-12:
        # Deterministic limiting case under risk-neutral pricing.
        premium = np.maximum(
            strike_x * math.exp(-r_eur * tau) - x_spot * math.exp(-r_pln * tau),
            0.0,
        )
        return premium, strike_x, fwd_x

    vol_sqrt_t = sigma_implied * math.sqrt(tau)
    d1 = (-math.log(alpha) + 0.5 * sigma_implied**2 * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    n_minus_d1 = norm_cdf(-d1)
    n_minus_d2 = norm_cdf(-d2)

    premium = (
        strike_x * math.exp(-r_eur * tau) * n_minus_d2
        - x_spot * math.exp(-r_pln * tau) * n_minus_d1
    )
    return premium, strike_x, fwd_x


def build_simulation(inp: ModelInputs, base_year: int = 2026) -> SimulationOutput:
    x_hedge, x_terminal, hedge_dates_t, settlement = simulate_paths(
        inp.spot_eurpln,
        inp.sigma_realized,
        inp.n_sims,
        inp.seed,
        base_year,
    )
    hedge_dates = list(hedge_dates_t)
    taus = np.array([(settlement - d).days / 365.0 for d in hedge_dates])

    monthly_dividend = inp.dividend_pln / 12.0

    # No hedge: all PLN converted into EUR at April settlement.
    payoff_unhedged = inp.dividend_pln * x_terminal

    # Build the payoff if 100% of each monthly accrual were hedged with forwards
    # and, separately, if 100% were protected with puts.
    payoff_forward_100 = np.zeros(inp.n_sims)
    payoff_put_100 = np.zeros(inp.n_sims)

    monthly_rows: list[dict] = []
    x0 = 1.0 / inp.spot_eurpln

    for m, (hedge_date, tau) in enumerate(zip(hedge_dates, taus)):
        x_t = x_hedge[:, m]

        # Forward to the common April settlement date.
        fwd_x = x_t * math.exp((inp.r_eur - inp.r_pln) * tau)

        # Optional execution cost, measured as bps of contemporaneous EUR value
        # and carried to settlement at the EUR rate.
        fwd_exec_cost_t = (
            monthly_dividend
            * x_t
            * (inp.forward_cost_bps / 10_000.0)
        )
        fwd_exec_cost_T = fwd_exec_cost_t * math.exp(inp.r_eur * tau)
        payoff_forward_100 += monthly_dividend * fwd_x - fwd_exec_cost_T

        # Put on PLN: underlying PLN is still converted at terminal spot, while
        # the option supplies a floor in EUR per PLN.
        put_premium_model, strike_x, _ = put_price_per_pln(
            x_t,
            tau,
            inp.r_eur,
            inp.r_pln,
            inp.sigma_implied,
            inp.strike_pct_of_forward,
        )
        put_premium_paid = put_premium_model * (1.0 + inp.option_markup_pct)
        put_premium_T = (
            monthly_dividend
            * put_premium_paid
            * math.exp(inp.r_eur * tau)
        )
        protected_conversion = monthly_dividend * (
            x_terminal + np.maximum(strike_x - x_terminal, 0.0)
        )
        payoff_put_100 += protected_conversion - put_premium_T

        # Reference values if EUR/PLN spot happened to remain unchanged at the
        # hedge date. These are for intuition only; actual simulated hedge rates
        # are path dependent.
        fwd_ref_x = x0 * math.exp((inp.r_eur - inp.r_pln) * tau)
        put_prem_ref, strike_ref_x, _ = put_price_per_pln(
            np.array([x0]),
            tau,
            inp.r_eur,
            inp.r_pln,
            inp.sigma_implied,
            inp.strike_pct_of_forward,
        )
        put_prem_ref_paid = put_prem_ref[0] * (1.0 + inp.option_markup_pct)

        monthly_rows.append(
            {
                "Month": hedge_date.strftime("%b"),
                "Hedge date": hedge_date.strftime("%d-%b-%Y"),
                "Tenor (months approx.)": round(tau * 12.0, 1),
                "Illustrative forward EUR/PLN": 1.0 / fwd_ref_x,
                "Illustrative put barrier EUR/PLN": 1.0 / strike_ref_x[0],
                "Put premium (% of PLN spot EUR value)": 100.0 * put_prem_ref_paid / x0,
            }
        )

    monthly_reference = pd.DataFrame(monthly_rows)

    return SimulationOutput(
        hedge_dates=hedge_dates,
        settlement_date=settlement,
        tau_to_settlement=taus,
        x_hedge=x_hedge,
        x_terminal=x_terminal,
        payoff_unhedged=payoff_unhedged,
        payoff_forward_100=payoff_forward_100,
        payoff_put_100=payoff_put_100,
        monthly_reference=monthly_reference,
    )


def strategy_payoff(
    sim: SimulationOutput,
    forward_pct: float,
    put_pct: float,
) -> np.ndarray:
    """Combine the three base payoff arrays. Percentages are decimals."""
    if forward_pct < 0 or put_pct < 0 or forward_pct + put_pct > 1.0 + 1e-12:
        raise ValueError("Forward % + put % must be between 0% and 100%.")
    unhedged_pct = 1.0 - forward_pct - put_pct
    return (
        unhedged_pct * sim.payoff_unhedged
        + forward_pct * sim.payoff_forward_100
        + put_pct * sim.payoff_put_100
    )


def metrics(payoff: np.ndarray, downside_floor_eur: float) -> dict[str, float]:
    return {
        "Mean": float(np.mean(payoff)),
        "Std": float(np.std(payoff, ddof=1)),
        "P01": float(np.quantile(payoff, 0.01)),
        "P05": float(np.quantile(payoff, 0.05)),
        "Median": float(np.quantile(payoff, 0.50)),
        "P95": float(np.quantile(payoff, 0.95)),
        "ShortfallProb": float(np.mean(payoff < downside_floor_eur)),
    }


def build_strategy_grid(
    sim: SimulationOutput,
    downside_floor_eur: float,
    step_pct: int = 5,
) -> pd.DataFrame:
    rows = []
    for f_pct in range(0, 101, step_pct):
        for p_pct in range(0, 101 - f_pct, step_pct):
            f = f_pct / 100.0
            p = p_pct / 100.0
            payoff = strategy_payoff(sim, f, p)
            m = metrics(payoff, downside_floor_eur)
            rows.append(
                {
                    "Forward %": f_pct,
                    "Put %": p_pct,
                    "Unhedged %": 100 - f_pct - p_pct,
                    "Mean EUR m": m["Mean"] / 1e6,
                    "Std EUR m": m["Std"] / 1e6,
                    "P05 EUR m": m["P05"] / 1e6,
                    "Shortfall %": 100.0 * m["ShortfallProb"],
                }
            )
    return pd.DataFrame(rows)


def efficient_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-standard-deviation frontier: keep points that are not dominated by a
    strategy with weakly lower risk and strictly higher mean.
    """
    d = df.sort_values(["Std EUR m", "Mean EUR m"], ascending=[True, False]).copy()
    best_mean = -np.inf
    keep = []
    for idx, row in d.iterrows():
        if row["Mean EUR m"] > best_mean + 1e-10:
            keep.append(idx)
            best_mean = row["Mean EUR m"]
    return d.loc[keep].sort_values("Std EUR m")


def pct_label(x: float) -> str:
    return f"{100*x:.0f}%"


# -----------------------------
# Streamlit interface
# -----------------------------

st.set_page_config(
    page_title="PLN Dividend Hedging Simulator",
    page_icon="📈",
    layout="wide",
)

st.title("PLN Dividend Hedging Simulator")
st.caption(
    "January-December monthly hedging; every contract matures on April 30 of the following year. "
    "Results are terminal euro proceeds from a PLN dividend."
)

with st.sidebar:
    st.header("1. Economic assumptions")
    dividend_m_pln = st.number_input(
        "Annual dividend (PLN millions)",
        min_value=1.0,
        value=100.0,
        step=10.0,
    )
    spot_eurpln = st.number_input(
        "Starting EUR/PLN (PLN per €1)",
        min_value=0.5,
        value=4.30,
        step=0.01,
        format="%.4f",
    )
    r_pln_pct = st.number_input(
        "PLN interest rate (%)",
        min_value=-5.0,
        max_value=30.0,
        value=3.90,
        step=0.10,
    )
    r_eur_pct = st.number_input(
        "EUR interest rate (%)",
        min_value=-5.0,
        max_value=20.0,
        value=2.20,
        step=0.10,
    )
    realized_vol_pct = st.number_input(
        "Realized FX volatility for simulation (%)",
        min_value=0.1,
        max_value=50.0,
        value=8.0,
        step=0.5,
    )
    implied_vol_pct = st.number_input(
        "Implied FX volatility for put pricing (%)",
        min_value=0.1,
        max_value=50.0,
        value=8.0,
        step=0.5,
    )

    st.header("2. Selected hedging policy")
    forward_pct_ui = st.slider(
        "Forward hedge (% of each month's accrued dividend)",
        0,
        100,
        50,
        5,
    )
    max_put = 100 - forward_pct_ui
    put_pct_ui = st.slider(
        "Put hedge (% of each month's accrued dividend)",
        0,
        max_put,
        min(10, max_put),
        5,
    )
    strike_pct_ui = st.slider(
        "Put strike (% of each month's forward rate)",
        80,
        110,
        100,
        1,
    )

    st.header("3. Simulation")
    n_sims = st.select_slider(
        "Number of simulated years",
        options=[5_000, 10_000, 25_000, 50_000, 100_000],
        value=50_000,
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    downside_floor_m_eur = st.number_input(
        "Downside floor for shortfall probability (€ millions)",
        min_value=0.0,
        value=21.0,
        step=0.25,
    )

    with st.expander("Advanced transaction-cost assumptions"):
        forward_cost_bps = st.number_input(
            "Forward execution cost (bps of EUR notional)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        option_markup_pct_ui = st.number_input(
            "Option premium dealer markup (%)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )


inp = ModelInputs(
    dividend_pln=dividend_m_pln * 1e6,
    spot_eurpln=spot_eurpln,
    r_eur=r_eur_pct / 100.0,
    r_pln=r_pln_pct / 100.0,
    sigma_realized=realized_vol_pct / 100.0,
    sigma_implied=implied_vol_pct / 100.0,
    n_sims=int(n_sims),
    seed=int(seed),
    forward_cost_bps=forward_cost_bps,
    option_markup_pct=option_markup_pct_ui / 100.0,
    strike_pct_of_forward=strike_pct_ui / 100.0,
)

with st.spinner("Simulating FX paths and pricing hedges..."):
    sim = build_simulation(inp)

f_selected = forward_pct_ui / 100.0
p_selected = put_pct_ui / 100.0
u_selected = 1.0 - f_selected - p_selected
selected_payoff = strategy_payoff(sim, f_selected, p_selected)

floor_eur = downside_floor_m_eur * 1e6
selected_metrics = metrics(selected_payoff, floor_eur)
no_hedge_metrics = metrics(sim.payoff_unhedged, floor_eur)

# -----------------------------
# Overview
# -----------------------------

st.subheader("Selected policy")
st.write(
    f"Each month: **{pct_label(f_selected)} forward**, **{pct_label(p_selected)} put**, "
    f"**{pct_label(u_selected)} unhedged**. All contracts settle on "
    f"**{sim.settlement_date.strftime('%d %B %Y')}**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Expected EUR proceeds",
    f"€{selected_metrics['Mean']/1e6:,.3f}m",
    delta=f"€{(selected_metrics['Mean']-no_hedge_metrics['Mean'])/1e6:+,.3f}m vs no hedge",
)
c2.metric(
    "Standard deviation",
    f"€{selected_metrics['Std']/1e6:,.3f}m",
    delta=f"{100*(selected_metrics['Std']/no_hedge_metrics['Std']-1):+.1f}% vs no hedge",
    delta_color="inverse",
)
c3.metric("5th percentile", f"€{selected_metrics['P05']/1e6:,.3f}m")
c4.metric(
    f"Probability below €{downside_floor_m_eur:.2f}m",
    f"{100*selected_metrics['ShortfallProb']:.2f}%",
)

st.info(
    "Interpretation: a higher hedge ratio generally lowers FX risk, but when PLN rates exceed EUR rates, "
    "forwards also reduce expected euro proceeds through negative carry. Puts preserve more upside, "
    "but their premium reduces expected proceeds."
)

# Comparison table
comparison_specs = [
    ("No hedge", 0.0, 0.0),
    ("25% monthly forwards", 0.25, 0.0),
    ("50% monthly forwards", 0.50, 0.0),
    ("100% monthly forwards", 1.00, 0.0),
    ("Selected policy", f_selected, p_selected),
]

comparison_rows = []
for name, f, p in comparison_specs:
    payoff = strategy_payoff(sim, f, p)
    m = metrics(payoff, floor_eur)
    comparison_rows.append(
        {
            "Strategy": name,
            "Forward %": 100*f,
            "Put %": 100*p,
            "Unhedged %": 100*(1-f-p),
            "Expected EUR (€m)": m["Mean"] / 1e6,
            "Std dev (€m)": m["Std"] / 1e6,
            "5th pct (€m)": m["P05"] / 1e6,
            "Shortfall prob (%)": 100*m["ShortfallProb"],
        }
    )
comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.drop_duplicates(
    subset=["Forward %", "Put %", "Unhedged %"], keep="last"
)

st.subheader("Policy comparison")
st.dataframe(
    comparison_df.style.format(
        {
            "Forward %": "{:.0f}",
            "Put %": "{:.0f}",
            "Unhedged %": "{:.0f}",
            "Expected EUR (€m)": "{:.3f}",
            "Std dev (€m)": "{:.3f}",
            "5th pct (€m)": "{:.3f}",
            "Shortfall prob (%)": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Tabs for deeper analysis
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Distribution", "Mean-risk frontier", "Monthly pricing", "Model notes"]
)

with tab1:
    st.subheader("Distribution of terminal euro proceeds")

    plot_df = pd.DataFrame(
        {
            "Selected policy": selected_payoff / 1e6,
            "No hedge": sim.payoff_unhedged / 1e6,
            "100% monthly forwards": sim.payoff_forward_100 / 1e6,
        }
    ).melt(var_name="Strategy", value_name="EUR proceeds (€m)")

    fig_hist = px.histogram(
        plot_df,
        x="EUR proceeds (€m)",
        color="Strategy",
        barmode="overlay",
        histnorm="probability density",
        nbins=80,
        opacity=0.45,
    )
    fig_hist.update_layout(legend_title_text="")
    st.plotly_chart(fig_hist, use_container_width=True)

    quantile_rows = []
    for name, payoff in [
        ("No hedge", sim.payoff_unhedged),
        ("Selected policy", selected_payoff),
        ("100% monthly forwards", sim.payoff_forward_100),
    ]:
        quantile_rows.append(
            {
                "Strategy": name,
                "1%": np.quantile(payoff, 0.01) / 1e6,
                "5%": np.quantile(payoff, 0.05) / 1e6,
                "Median": np.quantile(payoff, 0.50) / 1e6,
                "95%": np.quantile(payoff, 0.95) / 1e6,
                "99%": np.quantile(payoff, 0.99) / 1e6,
            }
        )
    qdf = pd.DataFrame(quantile_rows)
    st.dataframe(
        qdf.style.format({c: "{:.3f}" for c in ["1%", "5%", "Median", "95%", "99%"]}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Expected proceeds versus risk")
    st.caption(
        "Each point is a constant monthly mix of forwards, puts and unhedged exposure. "
        "The grid uses 5 percentage-point increments."
    )

    grid_df = build_strategy_grid(sim, floor_eur, step_pct=5)
    frontier_df = efficient_frontier(grid_df)

    fig_frontier = px.scatter(
        grid_df,
        x="Std EUR m",
        y="Mean EUR m",
        hover_data=["Forward %", "Put %", "Unhedged %", "P05 EUR m", "Shortfall %"],
        labels={
            "Std EUR m": "Standard deviation of EUR proceeds (€m)",
            "Mean EUR m": "Expected EUR proceeds (€m)",
        },
    )
    fig_frontier.add_trace(
        go.Scatter(
            x=frontier_df["Std EUR m"],
            y=frontier_df["Mean EUR m"],
            mode="lines+markers",
            name="Efficient frontier",
            hovertemplate="Frontier<extra></extra>",
        )
    )

    selected_m = metrics(selected_payoff, floor_eur)
    fig_frontier.add_trace(
        go.Scatter(
            x=[selected_m["Std"] / 1e6],
            y=[selected_m["Mean"] / 1e6],
            mode="markers",
            marker={"size": 14, "symbol": "star"},
            name="Selected policy",
            hovertemplate=(
                f"Selected: {forward_pct_ui}% forward, {put_pct_ui}% put"
                "<extra></extra>"
            ),
        )
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    # Practical optimization: highest mean subject to a shortfall constraint.
    max_shortfall = st.slider(
        "Maximum acceptable probability of falling below the downside floor (%)",
        0.0,
        25.0,
        5.0,
        0.5,
    )
    feasible = grid_df[grid_df["Shortfall %"] <= max_shortfall].copy()
    if len(feasible):
        best = feasible.sort_values(["Mean EUR m", "Std EUR m"], ascending=[False, True]).iloc[0]
        st.success(
            "Highest expected proceeds satisfying that constraint: "
            f"**{best['Forward %']:.0f}% forward, {best['Put %']:.0f}% put, "
            f"{best['Unhedged %']:.0f}% unhedged** — expected €{best['Mean EUR m']:.3f}m, "
            f"std. dev. €{best['Std EUR m']:.3f}m, 5th percentile €{best['P05 EUR m']:.3f}m."
        )
    else:
        st.warning("No strategy on the 5% grid satisfies that shortfall constraint.")

    st.download_button(
        "Download strategy grid as CSV",
        data=grid_df.to_csv(index=False).encode("utf-8"),
        file_name="hedging_strategy_grid.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Monthly contract intuition")
    st.caption(
        "Reference rates below assume EUR/PLN happened to remain at its starting spot on each hedge date. "
        "Actual simulated forward rates and option strikes vary with the simulated spot path."
    )
    st.dataframe(
        sim.monthly_reference.style.format(
            {
                "Tenor (months approx.)": "{:.1f}",
                "Illustrative forward EUR/PLN": "{:.4f}",
                "Illustrative put barrier EUR/PLN": "{:.4f}",
                "Put premium (% of PLN spot EUR value)": "{:.3f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    avg_tenor = float(np.mean(sim.tau_to_settlement))
    st.write(
        f"Average hedge tenor is **{avg_tenor*12:.1f} months**. "
        "January carries the longest forward exposure; December the shortest."
    )

with tab4:
    st.subheader("What the model assumes")
    st.markdown(
        """
- **Dividend:** generated evenly across January-December. Each month's hedge is applied only to that month's accrued dividend.
- **Settlement:** every hedge matures on April 30 of the following year.
- **FX process:** the model simulates **EUR per PLN** as a driftless geometric random walk in levels. The interface displays the reciprocal market quote, EUR/PLN.
- **Forward pricing:** covered interest parity using the entered EUR and PLN interest rates.
- **Put pricing:** Garman-Kohlhagen FX option pricing. The strike is expressed as a percentage of that month's forward rate.
- **Option premium:** paid on the hedge date and carried forward to April at the EUR interest rate before being deducted from terminal proceeds.
- **Volatility:** realized volatility drives simulated spot paths; implied volatility prices the puts. They may be set differently.
- **Transaction costs:** optional forward execution cost and option premium markup are user inputs; default is zero.
- **Common random numbers:** all strategies are evaluated on exactly the same simulated FX paths, making comparisons much less noisy.
        """
    )

    st.warning(
        "The model intentionally ignores dividend forecast error, taxes, hedge-accounting constraints, credit exposure, "
        "collateral/margin, liquidity limits, and changing interest rates. Those are natural extensions once the basic "
        "risk-return comparison is working."
    )
