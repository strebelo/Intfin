"""Currency hedging laboratory, with bounded memory and safe chart rendering.

Run: streamlit run hedging_basic_app.py
Dependencies: numpy, pandas, streamlit>=1.51, matplotlib.

INSTALLATION: Replace the ENTIRE contents of the deployed Python file with
this file. Do not append it to the previous app. The UI must run only once
per Streamlit rerun; explicit widget keys do not make duplicate app copies safe.

The financial model, default inputs, seed, and cash-flow timing are unchanged.
Revenue at t=0 is ignored; domestic costs at t=0 are included.
For each simulated path, revenue is affine in h. Simulating the unhedged and
fully hedged endpoints therefore suffices for every hedge fraction.
"""

from io import BytesIO
import logging
from threading import RLock

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


MAX_SIMULATIONS = 100_000
MAX_YEARS = 50
COST_COLUMN = "Costs in domestic currency"
REVENUE_COLUMN = "Revenue in foreign currency"
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_plot_lock():
    """One shared lock across sessions/reruns, for Matplotlib's shared state."""
    return RLock()


def make_discount_factors_constant(r: float, T: int):
    if not np.isfinite(r) or r <= -1.0:
        raise ValueError("Interest rates must be finite and greater than -100%.")
    return np.array([1.0 / ((1.0 + float(r)) ** t) for t in range(T + 1)])


def forward_dc_per_fc_constant_rate(S_t_dc_fc, r_d, r_f, t, m):
    if m <= t:
        raise ValueError("Forward maturity must be later than the trade date.")
    return S_t_dc_fc * ((1.0 + r_d) ** (m - t) / (1.0 + r_f) ** (m - t))


def bid_ask_from_mid_tenor_scaled(mid, spread_bps_per_year, years):
    spread = float(spread_bps_per_year) * float(years) / 10000.0
    return mid * (1.0 - spread / 2.0), mid * (1.0 + spread / 2.0)


def validate_inputs(S0, sigma, infl_diff, n_sims, T, seed,
                    r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year):
    values = [S0, sigma, infl_diff, r_d, r_f, spread_bps_per_year]
    if not np.all(np.isfinite(values)):
        raise ValueError("All inputs must be finite numbers.")
    if S0 <= 0 or sigma < 0:
        raise ValueError("The spot rate must be positive and volatility non-negative.")
    if r_d <= -1 or r_f <= -1:
        raise ValueError("Interest rates must be greater than -100%.")
    if infl_diff <= -1:
        raise ValueError("The expected annual change in the spot rate must exceed -100%.")
    if not 1 <= n_sims <= MAX_SIMULATIONS:
        raise ValueError(f"Choose between 1 and {MAX_SIMULATIONS:,} simulations.")
    if not 1 <= T <= MAX_YEARS:
        raise ValueError(f"Choose a time horizon between 1 and {MAX_YEARS} years.")
    if seed < 0:
        raise ValueError("The random seed must be non-negative.")
    if spread_bps_per_year < 0 or spread_bps_per_year * T >= 20000:
        raise ValueError("Reduce the spread or horizon so every forward bid remains positive.")
    for name, flows in [("Costs", costs_dc), ("Revenue", revenue_fc)]:
        flows = np.asarray(flows)
        if flows.shape != (T + 1,) or not np.all(np.isfinite(flows)):
            raise ValueError(f"{name} must contain one finite number per year, including year 0.")


def simulate_pv_components(S0, sigma, infl_diff, n_sims, T, seed,
                           r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year):
    """Accumulate PVs year by year using O(n_sims), rather than O(n_sims*T), memory.

    Draws are made in the original order (all simulations for each year), so a
    given seed generates the same paths as the original implementation.
    Only the current spot and two revenue vectors are retained. Fully hedging
    at t=0 and domestic costs are deterministic scalars, not repeated matrices.
    No simulation arrays are cached or kept in session state.
    """
    costs_dc = np.asarray(costs_dc, dtype=float)
    revenue_fc = np.asarray(revenue_fc, dtype=float)
    validate_inputs(S0, sigma, infl_diff, n_sims, T, seed,
                    r_d, r_f, costs_dc, revenue_fc, spread_bps_per_year)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        DF = make_discount_factors_constant(r_d, T)
        pv_cost = float(np.sum(costs_dc[1:] * DF[1:]) + costs_dc[0])
        rng = np.random.default_rng(seed)
        spot = np.full(n_sims, float(S0))
        pv_unhedged = np.zeros(n_sims)
        pv_roll = np.zeros(n_sims)
        pv_all0 = 0.0
        mu_adj = np.log1p(infl_diff) - 0.5 * sigma ** 2
        ratio = (1.0 + float(r_d)) / (1.0 + float(r_f))

        for t in range(1, T + 1):
            # Hedge at t-1, using the spot observed BEFORE the next shock.
            bid_roll, _ = bid_ask_from_mid_tenor_scaled(
                spot * ratio, spread_bps_per_year, 1.0
            )
            pv_roll += revenue_fc[t] * bid_roll * DF[t]

            mid_all0 = forward_dc_per_fc_constant_rate(S0, r_d, r_f, 0, t)
            bid_all0, _ = bid_ask_from_mid_tenor_scaled(
                mid_all0, spread_bps_per_year, t
            )
            pv_all0 += revenue_fc[t] * bid_all0 * DF[t]

            eps = rng.standard_normal(n_sims)
            growth = np.exp(mu_adj + sigma * eps)
            spot = np.maximum(1e-12, spot * growth)
            pv_unhedged += revenue_fc[t] * spot * DF[t]

        components = {
            "unhedged_revenue": pv_unhedged,
            "all0_revenue": float(pv_all0),
            "roll_revenue": pv_roll,
            "pv_cost": pv_cost,
        }
        if not all(np.all(np.isfinite(value)) for value in components.values()):
            raise ValueError("These inputs produce values too large to calculate reliably.")
        return components


def results_from_components(components, h, strategy):
    """Evaluate any h from the same simulated endpoints; no new paths needed."""
    if not np.isfinite(h) or not 0 <= h <= 1:
        raise ValueError("The hedge fraction must be between zero and one.")
    if strategy == "all_at_t0":
        hedged = components["all0_revenue"]
    elif strategy == "roll_one_year":
        hedged = components["roll_revenue"]
    else:
        raise ValueError("Unknown hedging strategy.")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pv_revenue = (1.0 - h) * components["unhedged_revenue"] + h * hedged
        pv_profit = pv_revenue - components["pv_cost"]
        # Exact constants have zero dispersion, including the h=1, t=0 hedge.
        std = 0.0
        if pv_profit.size > 1 and not np.all(pv_profit == pv_profit[0]):
            std = float(np.std(pv_profit, ddof=1))
        return {
            "pv_profit_per_sim": pv_profit,
            "avg_pv_revenue": float(np.mean(pv_revenue)),
            "avg_pv_cost": components["pv_cost"],
            "avg_pv_profit": float(np.mean(pv_profit)),
            "std_pv_profit": std,
            "frac_neg_profit": float(np.mean(pv_profit < 0.0)),
        }


def hedge_frontier(components):
    rows = []
    for strategy in ("all_at_t0", "roll_one_year"):
        for h in np.linspace(0.0, 1.0, 11):
            result = results_from_components(components, float(h), strategy)
            rows.append({"strategy": strategy, "h": h,
                         "mean": result["avg_pv_profit"],
                         "std": result["std_pv_profit"]})
    return pd.DataFrame(rows)


def render_chart(draw):
    """Render a private figure under a shared lock, and return only PNG bytes.

    Direct Figure construction avoids pyplot's global figure registry.
    Clearing in finally releases chart contents even when rendering fails.
    Streamlit receives an image, so it holds no Matplotlib figure references.
    """
    with get_plot_lock():
        fig = Figure(figsize=(8.0, 4.8))
        FigureCanvasAgg(fig)
        try:
            ax = fig.subplots()
            draw(ax)
            with BytesIO() as buffer:
                fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
                return buffer.getvalue()
        finally:
            fig.clear()


def distribution_image(A, B):
    def draw(ax):
        # Both distributions use the same bin edges for a fair comparison.
        bins = np.histogram_bin_edges(np.concatenate((A, B)), bins=40)
        ax.hist(A, bins=bins, alpha=0.5, label="Hedge at 0")
        ax.hist(B, bins=bins, alpha=0.5, label="Hedge 1 year before cash flow")
        ax.set_xlabel("PV Profit (DOM)")
        ax.set_ylabel("Frequency")
        ax.set_title("PV Profit: Overlaid Distributions")
        ax.legend()
    return render_chart(draw)


def frontier_image(frontier):
    def draw(ax):
        for strategy, label in [
            ("all_at_t0", "Hedge at 0"),
            ("roll_one_year", "Hedge 1 year before cash flow"),
        ]:
            rows = frontier.loc[frontier["strategy"] == strategy]
            ax.scatter(rows["std"], rows["mean"], label=label)
            for row in rows.itertuples(index=False):
                if strategy == "roll_one_year" and row.h == 0.0:
                    continue  # The unhedged point is common to both strategies.
                offset = (-6, 7) if strategy == "all_at_t0" else (6, -12)
                align = "right" if strategy == "all_at_t0" else "left"
                ax.annotate(f"{round(row.h * 100):d}%", (row.std, row.mean),
                            textcoords="offset points", xytext=offset, ha=align, fontsize=8)
        ax.set_xlabel(r"$\sigma(\mathrm{PV\ Profit})$")
        ax.set_ylabel(r"$\mathbb{E}[\mathrm{PV\ Profit}]$")
        ax.set_title("Frontier over $h$ (both strategies)")
        ax.legend()
        ax.margins(x=0.12, y=0.15)
    return render_chart(draw)


def show_simulation(components, h):
    results = [
        results_from_components(components, 0.0, "all_at_t0"),
        results_from_components(components, h, "all_at_t0"),
        results_from_components(components, h, "roll_one_year"),
    ]
    summary = pd.DataFrame({
        "Strategy": ["Unhedged", "Hedge at 0", "Hedge 1 year before cash flow"],
        "Hedge h": ["0.0%", f"{h * 100:.1f}%", f"{h * 100:.1f}%"],
        "Avg PV Revenue (DOM)": [r["avg_pv_revenue"] for r in results],
        "Avg PV Cost (DOM)": [r["avg_pv_cost"] for r in results],
        "Avg PV Profit (DOM)": [r["avg_pv_profit"] for r in results],
        "StdDev PV Profit": [r["std_pv_profit"] for r in results],
        "Frac(PV Profit < 0)": [r["frac_neg_profit"] for r in results],
    })
    for column in ["Avg PV Revenue (DOM)", "Avg PV Cost (DOM)",
                   "Avg PV Profit (DOM)", "StdDev PV Profit"]:
        summary[column] = summary[column].map(lambda value: f"{value:,.2f}")
    summary["Frac(PV Profit < 0)"] = summary["Frac(PV Profit < 0)"].map(
        lambda value: f"{100 * value:.1f}%"
    )
    st.dataframe(summary, width="stretch")
    st.markdown("#### PV Profit — Overlaid Distributions "
                "(Hedge at 0 vs Hedge 1 year before cash flow)")
    st.image(distribution_image(results[1]["pv_profit_per_sim"],
                                results[2]["pv_profit_per_sim"]))


def main():
    st.set_page_config(page_title="Currency Hedging — Basic (constant h_t)", layout="wide")
    st.title("💱 Currency Risk Hedging Laboratory")
    with st.expander("Show math / notation"):
        st.latex(r"F_{t,m} = S_t \cdot \frac{(1+r_d)^{m-t}}{(1+r_f)^{m-t}}")
        st.latex(r"\mu = \ln(1+\pi_{\Delta}) - \tfrac{1}{2}\sigma^2,"
                 r"\quad \mathbb{E}\!\left[\frac{S_t}{S_{t-1}}\right]=1+\pi_{\Delta}")
        st.latex(r"\text{h is the fraction of foreign currency revenue hedged}")

    st.sidebar.header("Inputs")
    S0 = st.sidebar.number_input("Current spot S0 (Dom./For. currency)",
                                min_value=1e-9, value=1.17, step=0.01, format="%.6f",
                                key="hedging_s0")
    T = int(st.sidebar.number_input("Time horizon T (years)",
                                   min_value=1, max_value=MAX_YEARS, value=2, step=1,
                                   key="hedging_horizon"))
    sigma = st.sidebar.number_input("Annual FX volatility (%)",
                                   min_value=0.0, value=10.0, step=0.5,
                                   key="hedging_volatility") / 100.0
    infl_diff = st.sidebar.number_input("Expected annual change in spot exchange rate, (%/yr)",
                                        value=0.0, step=0.25, format="%.4f",
                                        key="hedging_spot_drift") / 100.0
    r_d = st.sidebar.number_input("Domestic interest rate (%/yr)",
                                  value=2.0, step=0.25, format="%.4f",
                                  key="hedging_domestic_rate") / 100.0
    r_f = st.sidebar.number_input("Foreign interest rate (%/yr)",
                                  value=3.0, step=0.25, format="%.4f",
                                  key="hedging_foreign_rate") / 100.0
    spread = st.sidebar.number_input("Forward bid–ask spread (bps per year of maturity)",
                                      min_value=0.0, value=0.0, step=0.5,
                                      key="hedging_forward_spread")
    st.sidebar.markdown("---")
    h = st.sidebar.number_input("Hedge fraction h (% of each year)", min_value=0.0,
                                max_value=100.0, value=50.0, step=1.0, format="%.1f",
                                key="hedging_fraction") / 100.0
    n_sims = int(st.sidebar.number_input("Number of simulations", min_value=1,
                                        max_value=MAX_SIMULATIONS, value=5000, step=100,
                                        key="hedging_simulations"))
    seed = int(st.sidebar.number_input("Random seed", min_value=0,
                                      max_value=2**32 - 1, value=42, step=1,
                                      key="hedging_seed"))

    st.subheader("Cash Flows (including t = 0)")
    st.caption("Costs at year 0 are included. Revenue at year 0 is ignored, as in the original model.")
    cash_df = pd.DataFrame({COST_COLUMN: [0.0] * (T + 1),
                            REVENUE_COLUMN: [0.0] * (T + 1)})
    cash_df.index = pd.Index(range(T + 1), name=f"Year (0–{T})")
    cash_df = st.data_editor(cash_df, num_rows="fixed", width="stretch",
                             key=f"cash_flows_{T}")
    costs = pd.to_numeric(cash_df[COST_COLUMN], errors="coerce").fillna(0).to_numpy(dtype=float)
    revenues = pd.to_numeric(cash_df[REVENUE_COLUMN], errors="coerce").fillna(0).to_numpy(dtype=float)

    col1, col2 = st.columns([1, 1])
    simulate_btn = col1.button("Run Simulation", key="hedging_run")
    sweep_btn = col2.button("Plot Vol vs Mean, vary h", key="hedging_sweep")
    # Keep the original button-triggered behavior: editing inputs does not run
    # a simulation. No additional caching or repeated full-path matrices.
    if not (simulate_btn or sweep_btn):
        return

    try:
        with st.spinner("Running simulation..."):
            components = simulate_pv_components(
                S0, sigma, infl_diff, n_sims, T, seed, r_d, r_f, costs, revenues, spread
            )
            if simulate_btn:
                show_simulation(components, h)
            if sweep_btn:
                st.image(frontier_image(hedge_frontier(components)))
    except ValueError as exc:
        LOGGER.exception("Invalid input or chart calculation")
        st.error(str(exc))
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        LOGGER.exception("Numerical overflow in the hedging simulation")
        st.error("These inputs produce numbers outside a reliable numerical range. "
                 "Try a shorter horizon or less extreme rates, volatility, or cash flows.")
    except MemoryError:
        LOGGER.exception("Insufficient memory for the hedging simulation")
        st.error("There is not enough available memory. Try fewer simulations.")
    except Exception:
        LOGGER.exception("Unexpected error in the hedging app")
        st.error("The calculation could not be completed. The error details are in the app logs.")


if __name__ == "__main__":
    main()
