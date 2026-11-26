
"""EUR/PLN hedge comparison with common settlement at December year-end.

Run this file with Python 3.11 or 3.12:
    python -m pip install "streamlit>=1.49,<2" "pandas>=2.2,<3" "numpy>=1.26,<3" python-dateutil
    python -m streamlit run Hedging_JM.py

This file contains the pricing and simulation engine, the Streamlit interface,
and snapshots of ECB spot fixings, Eurostat money-market curves and effective-dated
NBP and ECB policy rates. Historical forwards can also use uploaded dated rates.
"""
from __future__ import annotations

import calendar
import csv
import io
import json
from datetime import timedelta
from functools import lru_cache
from io import BytesIO, StringIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.easter import easter



# Hedge scheduling, forward pricing and random walks

def annual_yoy_vol(spot_series, end_month=None):
    """Sample SD of 180 month-end, year-on-year log changes (decimal annual units).

    Supply at least 192 complete, consecutive calendar months of daily or
    month-end PLN/EUR observations. end_month is the final *complete* month.
    As a convenience, omission uses the last month in the supplied series.
    """
    spots = pd.Series(spot_series, dtype=float).dropna().sort_index()
    if not isinstance(spots.index, pd.DatetimeIndex) or spots.empty:
        raise ValueError("Volatility calibration needs a date-indexed spot series.")
    if not np.isfinite(spots).all() or (spots <= 0).any():
        raise ValueError("FX observations must be finite and positive.")
    monthly = spots.groupby(spots.index.to_period("M")).last()
    end = pd.Period(end_month, freq="M") if end_month is not None else monthly.index[-1]
    periods = pd.period_range(end=end, periods=192, freq="M")
    window = monthly.reindex(periods)
    if window.isna().any():
        missing = ", ".join(map(str, window.index[window.isna()][:3]))
        raise ValueError(f"192 consecutive month-end observations are required. Missing {missing}.")
    return float(np.log(window).diff(12).dropna().std(ddof=1))


def simulate_spots(dates, start_spot, n_paths, mode, annual_vol=None,
                   daily_log_returns=None, seed=1234):
    """Draw independent daily log increments; the first supplied date has start_spot.

    Gaussian annual_vol is a decimal; 252 trading days per year. Bootstrap draws
    are independent resamples of historical daily log changes after demeaning.
    User-entered Gaussian volatility does not rescale empirical innovations.
    """
    n_days = len(dates)
    if n_days < 2 or not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("Supply at least two trading dates and one path.")
    if not np.isfinite(start_spot) or start_spot <= 0:
        raise ValueError("The initial spot must be positive.")
    rng = np.random.default_rng(seed)
    size = (n_paths, n_days - 1)
    if mode == "gaussian":
        if annual_vol is None or not np.isfinite(annual_vol) or annual_vol < 0:
            raise ValueError("Annual Gaussian volatility must be a nonnegative decimal.")
        increments = rng.normal(0.0, annual_vol / np.sqrt(252), size=size)
    elif mode == "bootstrap":
        innovations = np.asarray(daily_log_returns, dtype=float)
        if innovations.ndim != 1 or innovations.size < 2 or not np.isfinite(innovations).all():
            raise ValueError("Supply at least two finite historical daily log returns.")
        increments = rng.choice(innovations - innovations.mean(), size=size, replace=True)
    else:
        raise ValueError("Simulation mode must be 'gaussian' or 'bootstrap'.")
    paths = np.empty((n_paths, n_days))
    paths[:, 0] = start_spot
    paths[:, 1:] = start_spot * np.exp(np.cumsum(increments, axis=1))
    return paths


def _daily_rates(values, n_days, name):
    """Effective annual yields supplied as percentages, scalar or daily."""
    rates = np.asarray(values, dtype=float)
    if rates.ndim == 0:
        rates = np.full(n_days, float(rates))
    if rates.shape != (n_days,) or not np.isfinite(rates).all() or (rates <= -100).any():
        raise ValueError(f"{name} must be finite effective annual percentages above -100, one per date or a scalar.")
    return rates / 100


def _signal_fraction(spot, average, band):
    """Return cumulative fraction due this month. Additional anticipation is separate."""
    # Rounding in the arithmetic mean must not break a flat market's tie.
    tolerance = 16 * np.finfo(float).eps * average
    return np.select(
        [spot > average * (1 + band) + tolerance, spot > average + tolerance],
        [0.0, 0.5], default=1.0
    )


def evaluate_policies(dates, spots, cashflows, pln_yields, eur_yields,
                      hedge_fraction=0.25, direction="pln_receipts",
                      delay="2w", band_bps=5.0, spread_bps=0.0, audit=False,
                      schedule="once", year_end_catchup=False,
                      rate_convention="annual_effective", rate_dates=None):
    """Return December cash by strategy for each paired path. Positive advantage favors tactical.

    One January–December trading grid is used per call. Cash flows are fixed
    nonnegative amounts known to the scenario. Entries January–November first
    come due for a hedge on the *next* month's last trading day. Under 'once',
    just one fraction of each flow is scheduled. Under 'layered', that fraction
    is also scheduled each subsequent month until 100% or maturity. December's own
    cash flow converts entirely at December spot. The contractual settlement
    date for every booking is that same final trading day. Optional year-end
    catch-up strikes a December-maturity forward at the penultimate fixing for
    all still-open January–November amounts. This overrides the regular target
    percentage and month-end trade date for those balances only.

    A decision on day t uses the preceding fixing. At scheduled month-end its
    reference is the arithmetic average of that month's daily observations
    through t-1. The reference stays fixed for the tranche's subsequent daily
    reviews. A half signal requires 50% cumulative coverage; it does not halve
    each day's unpaid balance. A 200% signal at the scheduled review can advance
    up to one current target from the next eligible target. Any remaining amount
    is forced by the last available business day on/before the absolute deadline.
    Each decision executes at the next (day t) reference fixing, a pricing proxy.

    'pln_receipts' means fixed PLN converted into EUR. 'eur_payments' means fixed
    EUR bought for PLN. Both buy EUR at the ask. spread_bps is a one-way markup
    to the modeled PLN/EUR rate, applied identically to spot and forwards.
    All results are terminal EUR proceeds or terminal PLN cost; the 'unhedged'
    comparator converts the whole cash total at the same December spot ask.
    'benchmark_locked' and 'tactical_locked' are the input-currency notionals
    contracted strictly *before* maturity. Requests for an audit return a
    list of trades and scheduled decisions for path 0 only.
    """
    days = pd.DatetimeIndex(dates)
    if days.empty or days.hasnans or not days.is_monotonic_increasing or days.has_duplicates:
        raise ValueError("The trading dates must be valid, distinct and chronological.")
    if len(days.year.unique()) != 1 or set(days.month) != set(range(1, 13)):
        raise ValueError("Supply one complete calendar year, January through December.")
    if (days != days.normalize()).any():
        raise ValueError("Use business dates without intraday times.")
    months = days.month.to_numpy()
    month_days = [np.flatnonzero(months == m) for m in range(1, 13)]
    if any(len(indices) < 2 for indices in month_days):
        raise ValueError("The signal needs at least two observations in every month.")
    if (days[-1].days_in_month - days[-1].day > 6 or days[0].day > 7
            or any(days[idx[-1]].days_in_month - days[idx[-1]].day > 6 for idx in month_days)):
        raise ValueError("Incomplete trading calendar. Check for missing month ends.")
    prices = np.asarray(spots, dtype=float)
    if prices.ndim == 1:
        prices = prices[None, :]
    if (prices.ndim != 2 or prices.shape[1] != len(days) or prices.shape[0] < 1
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError("Spots must be positive and finite, with one column per date.")
    flows = np.asarray(cashflows, dtype=float)
    if flows.shape != (12,) or not np.isfinite(flows).all() or (flows < 0).any():
        raise ValueError("Supply 12 nonnegative monthly cash flows in one currency.")
    if not np.isfinite(hedge_fraction) or not 0 <= hedge_fraction <= 1:
        raise ValueError("The hedge fraction must be between 0 and 1.")
    if direction not in ("pln_receipts", "eur_payments") or delay not in ("2w", "1m"):
        raise ValueError("Select a valid conversion direction and maximum delay.")
    if not np.isfinite([band_bps, spread_bps]).all() or min(band_bps, spread_bps) < 0:
        raise ValueError("Basis-point inputs must be finite and nonnegative.")

    if schedule not in ("once", "layered"):
        raise ValueError("Schedule must be 'once' or 'layered'.")
    pln_rate = _daily_rates(pln_yields, len(days), "PLN yields")
    euro_rate = _daily_rates(eur_yields, len(days), "EUR yields")
    maturity = len(days) - 1
    days_left = (days[-1] - days).days.to_numpy()
    if rate_convention == "annual_effective":
        carry = np.exp(days_left / 365 * (np.log1p(pln_rate) - np.log1p(euro_rate)))
    elif rate_convention in ("simple_money_market", "simple_interbank"):
        pln_basis = 365 if rate_convention == "simple_interbank" else 360
        pln_gross = 1 + pln_rate * days_left / pln_basis
        eur_gross = 1 + euro_rate * days_left / 360
        if min(pln_gross.min(), eur_gross.min()) <= 0:
            raise ValueError("Rates must give positive gross accumulation factors to December.")
        carry = pln_gross / eur_gross
    else:
        raise ValueError("Unknown interest rate convention.")
    if rate_dates is not None:
        rate_dates = pd.DatetimeIndex(rate_dates)
        if len(rate_dates) != len(days) or rate_dates.hasnans or (rate_dates > days).any():
            raise ValueError("Each trade date needs a valid rate date on or before execution.")
    ask_factor = 1 + spread_bps / 10_000
    band = band_bps / 10_000
    n_paths = len(prices)
    # Rows are successive scheduled hedge months (February to December).
    # Columns are input cash-flow months (January to December).
    allocation = np.zeros((11, 12))
    unscheduled = flows.copy()
    for tranche in range(11):
        first_month = tranche if schedule == "once" else 0
        eligible = slice(first_month, tranche + 1)
        amount = np.minimum(unscheduled[eligible], hedge_fraction * flows[eligible])
        allocation[tranche, eligible] = amount
        unscheduled[eligible] -= amount
    targets = allocation.sum(axis=1)
    prepaid = np.zeros((11, n_paths))
    totals = {key: np.zeros(n_paths) for key in ("benchmark", "tactical")}
    locked = {key: np.zeros(n_paths) for key in totals}
    records = []
    one = np.ones(n_paths)
    proceeds = direction == "pln_receipts"

    def book(strategy, tranche, day, amount, action, reference=None, signal_date=None, residual_month=None):
        # amount is fixed in the input currency; each trade settles on the same date.
        ask = prices[:, day] * (carry[day] * ask_factor)
        value = amount / ask if proceeds else amount * ask
        totals[strategy] += value
        if day < maturity:
            locked[strategy] += amount
        if not audit:
            return
        if amount[0] == 0 and action != "Scheduled signal":
            return
        if residual_month is None:
            weights = allocation[tranche] / targets[tranche]
            scheduled_month = calendar.month_abbr[tranche + 2]
        else:
            weights = np.eye(12)[residual_month]
            scheduled_month = None
        for month in np.flatnonzero(weights):
            records.append({
                "strategy": strategy,
                "cash_flow_month": calendar.month_abbr[month + 1],
                "scheduled_hedge_month": scheduled_month,
                "trade_date": days[day].date().isoformat(),
                "settlement_date": days[-1].date().isoformat(),
                "action": action,
                "instrument": "Forward" if day < maturity else "Spot",
                "signal_date": None if signal_date is None else days[signal_date].date().isoformat(),
                "signal_spot": None if signal_date is None else float(prices[0, signal_date]),
                "reference_average": None if reference is None else float(reference[0]),
                "spot_at_trade": float(prices[0, day]),
                "days_to_settlement": int(days_left[day]),
                "rate_date": (rate_dates[day].date().isoformat()
                              if rate_dates is not None and day < maturity else None),
                "pln_rate_pct": float(pln_rate[day] * 100) if day < maturity else None,
                "eur_rate_pct": float(euro_rate[day] * 100) if day < maturity else None,
                "rate_convention": rate_convention if day < maturity else None,
                "agreed_rate_pln_per_eur": float(ask[0]),
                "notional_input_currency": float(amount[0] * weights[month]),
                "settlement_amount_output_currency": float(value[0] * weights[month]),
            })

    for tranche in range(11):
        target = targets[tranche]
        if target == 0:
            continue
        trading_days = month_days[tranche + 1]
        due = trading_days[-1]
        if due == maturity:
            close = maturity - 1 if year_end_catchup else maturity
            action = "Year-end catch-up" if year_end_catchup else "Target settled at spot"
            book("benchmark", tranche, close, one * target, action)
            book("tactical", tranche, close, target - prepaid[tranche], action)
            continue
        book("benchmark", tranche, due, one * target, "Scheduled target")

        average = prices[:, trading_days[0]:due].mean(axis=1)
        prior = prices[:, due - 1]
        desired = target * _signal_fraction(prior, average, band)
        amount = np.maximum(0.0, desired - prepaid[tranche])
        book("tactical", tranche, due, amount, "Scheduled signal", average, due - 1)
        filled = prepaid[tranche] + amount

        favorable = prior < average * (1 - band) - 16 * np.finfo(float).eps * average
        advance = np.where(favorable, min(target, targets[tranche + 1]), 0.0)
        if np.any(advance):
            prepaid[tranche + 1] = advance
            book("tactical", tranche + 1, due, advance, "Anticipated next target", average, due - 1)

        cap = days[due] + (pd.Timedelta(days=14) if delay == "2w" else pd.DateOffset(months=1))
        deadline = min(maturity, days.searchsorted(cap, side="right") - 1)
        last_review = min(deadline, maturity - 1) if year_end_catchup else deadline
        for day in range(due + 1, last_review + 1):
            if np.all(filled >= target):
                break
            forced = day == deadline
            desired = target if forced else target * _signal_fraction(prices[:, day - 1], average, band)
            amount = np.maximum(0.0, desired - filled)
            if np.any(amount):
                action = "Forced at deadline" if forced else "Deferred balance"
                book("tactical", tranche, day, amount, action, average, None if forced else day - 1)
                filled += amount
        if year_end_catchup and deadline == maturity and np.any(filled < target):
            book("tactical", tranche, maturity - 1, target - filled, "Year-end catch-up")

    for month, residual in enumerate(unscheduled):
        if residual:
            catch_up = year_end_catchup and month < 11
            day = maturity - 1 if catch_up else maturity
            action = "Year-end catch-up" if catch_up else "Unhedged balance at spot"
            for strategy in totals:
                book(strategy, None, day, one * residual, action, residual_month=month)
    terminal = prices[:, -1] * ask_factor
    advantage = totals["tactical"] - totals["benchmark"]
    return {
        **totals,
        "advantage": advantage if proceeds else -advantage,
        "unhedged": flows.sum() / terminal if proceeds else flows.sum() * terminal,
        "benchmark_locked": locked["benchmark"],
        "tactical_locked": locked["tactical"],
        "audit": sorted(records, key=lambda row: (row["trade_date"], row["strategy"])),
    }


# Polish and German government yield sources

POLISH_CURVE = "https://yieldcartography.com/data/yields.json"
GERMAN_SERIES = "D.I.ZST.ZI.EUR.S1311.B.A604.R01XX.R.A.A._Z._Z.A"
GERMAN_CURVE = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/" + GERMAN_SERIES


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "December-FX-hedge-simulator/1.0"})
    with urlopen(request, timeout=25) as response:
        return response.read()


def _polish_yields(raw: bytes) -> pd.Series:
    daily = json.loads(raw)["ts_daily"]
    series = pd.Series(daily["y1"], index=pd.to_datetime(daily["dates"]), name="pln_yield")
    return pd.to_numeric(series, errors="raise").dropna()


def _german_yields(raw: bytes) -> pd.Series:
    # The Bundesbank CSV puts descriptive rows before observations, uses
    # semicolons, and represents German decimals with commas and missing days with ".".
    observations = {}
    for row in csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"):
        if len(row) < 2 or len(row[0]) != 10 or row[0][4:5] != "-":
            continue
        if row[1].strip() in ("", "."):
            continue
        observations[pd.Timestamp(row[0])] = float(row[1].replace(",", "."))
    return pd.Series(observations, name="eur_yield", dtype=float)


def load_government_yields(start="2010-01-01", end=None) -> pd.DataFrame:
    """Fetch contemporaneous PLN and German observations on common quote dates.

    Returns columns date, pln_yield, eur_yield. Rates are in percent (4.0 means
    4% per year). Missing dates are left to the client's explicit as-of rule.
    The caller can cache the return value and offer CSV upload on failure.
    """
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    if first > last:
        raise ValueError("The requested yield start date falls after the end date.")
    params = urlencode({"format": "csv", "startPeriod": first.date(), "endPeriod": last.date()})
    try:
        polish = _polish_yields(_fetch(POLISH_CURVE))
        german = _german_yields(_fetch(GERMAN_CURVE + "?" + params))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("Could not retrieve the Polish and German one-year yield series. "
                         "Use a yield CSV or try the live source again.") from exc

    panel = pd.concat([polish, german], axis=1, join="inner").sort_index().loc[first:last]
    if panel.empty or not panel.index.is_unique:
        raise ValueError("The government-yield sources returned no usable common date range.")
    return panel.rename_axis("date").reset_index()


# Market data, calibration and scenario comparison

ECB_API = "https://data-api.ecb.europa.eu/service/data/EXR/D.PLN.EUR.SP00.A"
SNAPSHOT_DATE = pd.Timestamp("2026-11-24")
BENCHMARK_RATES = (4.061, 3.000)  # November 24, 2026 market benchmark yields, percent
RESULT_COLUMNS = ["benchmark", "tactical", "unhedged", "advantage", "benchmark_locked", "tactical_locked"]
TENOR_MONTHS = (1, 3, 6, 12)
CURVE_COLUMNS = [f"{ccy}_{months}m" for ccy in ("pln", "eur") for months in TENOR_MONTHS]


def _validated(frame, columns, label):
    if not {"date", *columns}.issubset(frame.columns):
        raise ValueError(f"{label} needs columns: date, {', '.join(columns)}.")
    frame = frame[["date", *columns]].copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} needs ISO dates (YYYY-MM-DD) and numeric observations.") from exc
    if frame.empty or frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError(f"{label} needs at least one observation and unique dates.")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains a blank or non-finite value.")
    return frame.set_index("date").sort_index()


def load_inputs(fx_upload=None, rates_upload=None, refresh_fx=False):
    """Read spot and optional paired yield observations. Empty yields select the bundled rate history."""
    if fx_upload is not None:
        frame = pd.read_csv(BytesIO(fx_upload))
        fx_source = "Uploaded daily spot observations (PLN per EUR)."
    elif refresh_fx:
        args = urlencode({"startPeriod": "2010-11-01", "format": "csvdata"})
        try:
            req = Request(f"{ECB_API}?{args}", headers={"User-Agent": "December-FX-Hedge/1"})
            with urlopen(req, timeout=30) as res:
                frame = pd.read_csv(res, usecols=["TIME_PERIOD", "OBS_VALUE"]).rename(
                    columns={"TIME_PERIOD": "date", "OBS_VALUE": "spot"})
        except (OSError, ValueError) as exc:
            raise ValueError("ECB refresh failed. Deselect refresh to use the bundled daily observations, or upload a daily FX CSV.") from exc
        fx_source = "ECB daily euro reference rates, refreshed from EXR.D.PLN.EUR.SP00.A."
    else:
        frame = pd.read_csv(StringIO(ECB_SPOT_CSV))
        fx_source = "Bundled ECB daily euro reference rates, EXR.D.PLN.EUR.SP00.A."
    fx = _validated(frame, ["spot"], "FX file")["spot"]
    if (fx <= 0).any():
        raise ValueError("Every daily FX rate must be positive, in PLN per EUR.")
    rate_source = ""
    rate_note = ""
    rates = pd.DataFrame(columns=["pln_yield", "eur_yield"], dtype=float,
                         index=pd.DatetimeIndex([], name="date"))
    if rates_upload is not None:
        frame = pd.read_csv(BytesIO(rates_upload))
        is_curve = set(CURVE_COLUMNS).issubset(frame.columns)
        columns = CURVE_COLUMNS if is_curve else list(rates.columns)
        rates = _validated(frame, columns, "Yield file")
        if (rates <= -100).any().any():
            raise ValueError("All annual interest rates must exceed −100%.")
        rate_source = ("Uploaded 1-, 3-, 6- and 12-month PLN and EUR deposit curves." if is_curve else
                       "Uploaded dated PLN and EUR annual effective yields.")
        rate_note = (f"Uploaded observations from {rates.index.min():%Y-%m-%d} through "
                     f"{rates.index.max():%Y-%m-%d}, expressed as annual percentages.")
    return fx, rates, {"fx_source": fx_source, "rate_source": rate_source, "rate_note": rate_note}


def _calendar(year):
    """Shared ECB/TARGET and standard Polish public business days."""
    e = pd.Timestamp(easter(year))
    fixed = [(1, 1), (5, 1), (5, 3), (8, 15), (11, 1), (11, 11), (12, 25), (12, 26)]
    if year >= 2011:
        fixed.append((1, 6))
    if year >= 2025:
        fixed.append((12, 24))
    if year == 2018:
        fixed.append((11, 12))
    holidays = {pd.Timestamp(year, month, day) for month, day in fixed}
    holidays.update((e - timedelta(days=2), e + timedelta(days=1), e + timedelta(days=60)))
    weekdays = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    return weekdays[~weekdays.isin(holidays)]


def _complete_fx_years(fx):
    years = {}
    for year in sorted(set(fx.index.year)):
        expected = _calendar(year)
        observed = fx.index.intersection(expected)
        # All modeled trading days must have actual observations; a missing day
        # could hide a decision or change its execution date.
        if len(observed) == len(expected):
            years[year] = observed
    return years


def _last_full_month(fx):
    last = fx.index.max()
    month = last.to_period("M")
    if last < pd.offsets.BMonthEnd().rollforward(month.start_time):
        # A TARGET year-end or month-end closure can move the last published
        # fixing earlier than a plain weekday month-end.
        grid = _calendar(last.year)
        if last < grid[grid.month == last.month][-1]:
            return month - 1
    return month


def default_parameters(fx, rates):
    last_month = _last_full_month(fx)
    years = _complete_fx_years(fx)
    if not years:
        raise ValueError("At least one complete calendar year of daily FX observations is needed.")
    try:
        sigma = annual_yoy_vol(fx, end_month=last_month)
    except (TypeError, ValueError, IndexError):
        # Short uploads can still support historical replay and manual Gaussian calibration.
        sigma = None
    date = fx.index.max()
    available = rates.loc[:date]
    if available.empty:
        pln, eur = BENCHMARK_RATES
        rate_date = SNAPSHOT_DATE
    elif "pln_12m" in available:
        # Convert 12M simple deposit quotes to the one-year effective inputs used in simulations.
        pln, eur = available.iloc[-1][["pln_12m", "eur_12m"]]
        eur *= 365 / 360
        rate_date = available.index[-1]
    else:
        pln, eur = available.iloc[-1][["pln_yield", "eur_yield"]]
        rate_date = available.index[-1]
    return dict(spot=float(fx.iloc[-1]), pln_yield=float(pln), eur_yield=float(eur),
                valuation_date=date.strftime("%Y-%m-%d"), rate_date=rate_date.strftime("%Y-%m-%d"),
                annual_vol=sigma, vol_start=str(last_month - 179), vol_end=str(last_month), sim_year=max(years))


@lru_cache(maxsize=4)
def _live_yields(start, end):
    return load_government_yields(start, end).set_index("date")


@lru_cache(maxsize=1)
def _bundled_policy_rates():
    frame = pd.read_csv(StringIO(POLICY_CHANGES_CSV))
    return _validated(frame, ["pln_yield", "eur_yield"], "Bundled policy rates")


def _asof_rate_dates(rates, grid, *, policy=False):
    # The last policy decision effective on trade day is public. Market closes use
    # a strictly prior timestamp because today's close may follow today's ECB fixing.
    if policy and grid[-1] > POLICY_COVERAGE_END:
        raise ValueError("This year extends beyond the bundled policy-rate history.")
    pos = rates.index.searchsorted(grid, side="right" if policy else "left") - 1
    if np.any(pos < 0):
        raise ValueError("Missing prior yield observations.")
    used = rates.index[pos]
    if not policy:
        ages = (grid - used).days.to_numpy()
        gaps = rates.index.to_series().diff().dropna().dt.days
        limit = 45 if not gaps.empty and gaps.median() >= 20 else 10
        if np.max(ages) > limit:
            raise ValueError(f"Missing yield coverage (maximum permitted quote age {limit} days).")
    return used


@lru_cache(maxsize=1)
def _bundled_market_curves():
    frame = pd.read_csv(StringIO(MARKET_CURVES_CSV)).set_index("month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame[CURVE_COLUMNS]


def _market_grid_rates(curves, grid, *, bundled):
    """Return December-matched simple annual rates, interpolating in log discount factors."""
    if bundled:
        # Eurostat normally publishes a monthly series within 15 days. Allow a full
        # month after month-end. An August trade therefore uses June's average.
        months = grid.to_period("M") - 2
        observations = curves.reindex(months)
        if observations.isna().any().any():
            raise ValueError("No published monthly market curve for this year.")
        used = months.to_timestamp(how="end").normalize()
    else:
        used = _asof_rate_dates(curves, grid)
        observations = curves.loc[used]
    days_left = (grid[-1] - grid).days.to_numpy()
    tenor_days = np.column_stack([(grid + pd.DateOffset(months=m) - grid).days for m in TENOR_MONTHS])
    nodes = np.column_stack([np.zeros(len(grid)), tenor_days])

    def matched_rates(ccy, basis):
        quoted = observations[[f"{ccy}_{m}m" for m in TENOR_MONTHS]].to_numpy() / 100
        gross = 1 + quoted * tenor_days / basis
        if np.any(gross <= 0):
            raise ValueError("Market quotes imply a non-positive deposit accumulation factor.")
        log_discount = np.column_stack([np.zeros(len(grid)), -np.log(gross)])
        interpolated = np.array([
            np.interp(tenor, row_days, row_log_discount)
            for tenor, row_days, row_log_discount in zip(days_left, nodes, log_discount)
        ])
        return np.divide(np.expm1(-interpolated) * basis * 100, days_left,
                         out=np.zeros(len(grid)), where=days_left > 0)

    return matched_rates("pln", 365), matched_rates("eur", 360), used


def _bootstrap_innovations(fx):
    end_month = _last_full_month(fx)
    start = (end_month - 179).start_time
    end = end_month.end_time
    grids = [_calendar(y) for y in range(start.year - 1, end.year + 1)]
    all_days = grids[0].append(grids[1:])
    start_pos = all_days.searchsorted(start)
    required = all_days[(all_days >= all_days[start_pos - 1]) & (all_days <= end)]
    history = fx.reindex(required)
    if history.isna().any():
        raise ValueError("Bootstrap needs daily FX observations for 15 complete trailing years, plus a preceding fixing.")
    return np.log(history).diff().dropna().to_numpy(), start, end


def run_model(fx, rates, *, mode, cashflows, direction, hedge_fraction, delay,
              band_bps, spread_bps, start_spot, pln_yield, eur_yield, annual_vol,
              n_paths, seed, sim_year, historical_rates="observed", schedule="once", year_end_catchup=False):
    options = dict(cashflows=cashflows, direction=direction, hedge_fraction=hedge_fraction,
                   delay=delay, band_bps=band_bps, spread_bps=spread_bps, schedule=schedule,
                   year_end_catchup=year_end_catchup)
    years = _complete_fx_years(fx)
    if not years:
        raise ValueError("No complete joint-calendar year of daily FX data was found.")
    if mode == "historical":
        if historical_rates not in ("observed", "policy", "government", "fixed"):
            raise ValueError("Choose dated historical rates, live sovereign yields, or fixed rates.")
        # As-of rolling 15 *complete* years. Do not replace a missing year with an older one.
        asof = fx.index[-1]
        end_year = asof.year if asof >= _calendar(asof.year)[-1] else asof.year - 1
        wanted = list(range(end_year - 14, end_year + 1))
        rate_kind = historical_rates
        if historical_rates == "observed":
            rate_kind = ("market" if rates.empty else
                         "uploaded_curve" if set(CURVE_COLUMNS).issubset(rates.columns) else "uploaded")
        if rate_kind == "market":
            rates = _bundled_market_curves()
        elif rate_kind == "policy":
            rates = _bundled_policy_rates()
        elif rate_kind == "government":
            rates = _live_yields(f"{wanted[0] - 1}-11-01", f"{end_year}-12-31")
        rows, audit, skipped, ages = [], [], [], []
        for year in wanted:
            if year not in years:
                skipped.append(f"{year} (FX incomplete)")
                continue
            grid = years[year]
            pr, er = pln_yield, eur_yield
            rate_dates = None
            if rate_kind != "fixed":
                try:
                    if rate_kind in ("market", "uploaded_curve"):
                        pr, er, rate_dates = _market_grid_rates(rates, grid, bundled=rate_kind == "market")
                    else:
                        rate_dates = _asof_rate_dates(rates, grid, policy=rate_kind == "policy")
                        panel = rates.loc[rate_dates]
                        pr, er = panel.pln_yield.to_numpy(), panel.eur_yield.to_numpy()
                    ages.append(int((grid - rate_dates).days.max()))
                except ValueError as exc:
                    skipped.append(f"{year} ({exc})")
                    continue
            result = evaluate_policies(grid, fx.loc[grid].to_numpy(), pln_yields=pr,
                                       eur_yields=er, audit=True, rate_dates=rate_dates,
                                       rate_convention=({"policy": "simple_money_market",
                                                         "market": "simple_interbank",
                                                         "uploaded_curve": "simple_interbank"}
                                                        .get(rate_kind, "annual_effective")),
                                       **options)
            audit = result["audit"]
            rows.append({"label": year, **{k: float(result[k][0]) for k in RESULT_COLUMNS}})
        if len(rows) < 2:
            raise ValueError("Fewer than two complete years have usable data. Upload broader data or select fixed yields.")
        if rate_kind == "market":
            rate_note = ("Eurostat monthly 1M, 3M, 6M and 12M money-market rates (IRT_ST_M). "
                         "Trades use the average from two calendar months earlier to allow for publication. "
                         "The ledger's rate date is that observation month's calendar end. "
                         "Rates shown are maturity-matched equivalent simple yields.")
        elif rate_kind == "uploaded_curve":
            rate_note = ("Uploaded 1M, 3M, 6M and 12M money-market curves use the last observation strictly before "
                         "execution. The ledger gives its date and maturity-matched equivalent simple yields. "
                         f"Maximum quote age in included years: {max(ages)} calendar days.")
        elif rate_kind == "policy":
            rate_note = ("Each trade uses the NBP reference rate and ECB main refinancing rate in force on its "
                         "execution date. The ledger gives the most recent effective date of a change in either rate. "
                         "Both rates are held flat from that trade to December using simple ACT/360.")
        elif rate_kind == "fixed":
            rate_note = f"Fixed yields for every historical date: PLN {pln_yield:.3f}%, EUR {eur_yield:.3f}%."
        else:
            source = ("Live one-year Polish Treasury and German federal yields" if rate_kind == "government"
                      else "Uploaded PLN and EUR annual effective yields")
            rate_note = (f"{source}, using the last observation strictly before each trade. "
                         f"Maximum quote age in included years: {max(ages)} calendar days.")
        coverage = f"{len(rows)} annual observations; same input cash flows repeated in each year."
        if skipped:
            coverage += " Excluded: " + ", ".join(skipped) + "."
        return pd.DataFrame(rows), pd.DataFrame(audit), dict(period=f"{wanted[0]}–{wanted[-1]}",
            rate_kind=rate_kind, rate_note=rate_note, obs_note=coverage, audit_label=str(rows[-1]["label"]), n=len(rows))

    if mode not in ("gaussian", "bootstrap") or sim_year not in years:
        raise ValueError("Choose a simulated scenario and a complete reference calendar year.")
    grid = years[sim_year]
    innovations = None
    obs_note = f"{n_paths:,} paired paths on the {sim_year} calendar ({len(grid)} joint trading days)."
    if mode == "bootstrap":
        innovations, start, end = _bootstrap_innovations(fx)
        implied_vol = innovations.std(ddof=0) * np.sqrt(252)
        obs_note += (f" {len(innovations):,} daily log changes from {start:%Y-%m} through {end:%Y-%m}, "
                     f"centered and sampled with replacement. Under independent resampling, "
                     f"the implied annualized volatility is {implied_vol:.3%} (daily SD × √252).")
    spots = simulate_spots(grid, start_spot, n_paths, mode, annual_vol, innovations, seed)
    result = evaluate_policies(grid, spots, pln_yields=pln_yield, eur_yields=eur_yield, audit=True, **options)
    outcomes = pd.DataFrame({key: result[key] for key in RESULT_COLUMNS})
    outcomes["label"] = np.arange(1, len(outcomes) + 1)
    return outcomes, pd.DataFrame(result["audit"]), dict(period=f"Illustrative January–December ({sim_year} calendar)",
        rate_kind="fixed", rate_note=f"Fixed annual yields on every path: PLN {pln_yield:.3f}%, EUR {eur_yield:.3f}%.",
        obs_note=obs_note, audit_label="Simulation 1", n=len(outcomes))


def minimum_variance_hedges(outcomes):
    """Find the least-variance allocation to each configured policy on paired paths.

    A fraction h of every cash flow follows the policy, including its catch-up;
    the rest stays unhedged. Scaling all notionals preserves timing and prices,
    so each terminal outcome is U + h * (P - U) for 0 <= h <= 1. Both proceeds
    and costs use this formula. All risk measures are sample standard deviations.
    """
    base = outcomes["unhedged"].to_numpy(dtype=float)
    if len(base) < 2 or not np.isfinite(base).all():
        raise ValueError("Hedge optimization needs at least two finite paired outcomes.")
    # Subtract the first observation before centering to preserve variations
    # around a large common cash amount and give exactly zero for flat paths.
    base_moves = base - base[0]
    base_centered = base_moves - base_moves.mean()
    base_mean = base.mean()
    base_sd = base_moves.std(ddof=1)
    summary, frontier = [], []
    for policy in ("benchmark", "tactical"):
        full = outcomes[policy].to_numpy(dtype=float)
        locked = outcomes[f"{policy}_locked"].to_numpy(dtype=float)
        if not np.isfinite(full).all() or not np.isfinite(locked).all():
            raise ValueError("Hedge optimization needs finite policy outcomes and notionals.")
        delta = full - base
        delta_moves = delta - delta[0]
        delta_centered = delta_moves - delta_moves.mean()
        delta_scale = np.max(np.abs(delta_centered))
        amount_scale = max(np.max(np.abs(base)), np.max(np.abs(full)))
        tolerance = 32 * np.finfo(float).eps * amount_scale
        tied = delta_scale <= tolerance
        fraction = 0.0
        if not tied:
            # Normalisation avoids squaring very large or very small currency amounts.
            direction = delta_centered / delta_scale
            slope = np.sum((base_centered / delta_scale) * direction)
            fraction = float(np.clip(-slope / np.sum(direction * direction), 0.0, 1.0))

        full_mean, full_locked = full.mean(), locked.mean()
        # Include the analytic minimum on the display curve even between grid points.
        for h in np.unique(np.append(np.linspace(0, 1, 21), fraction)):
            row = dict(policy=policy, hedge_fraction=float(h),
                       mean=(1 - h) * base_mean + h * full_mean,
                       sd=(base_centered + h * delta_centered).std(ddof=1),
                       mean_locked=h * full_locked)
            frontier.append(row)
            if h == fraction:
                best = row
        summary.append(dict(**best, unhedged_mean=base_mean, unhedged_sd=base_sd,
                            full_mean=full_mean, full_sd=(full - full[0]).std(ddof=1),
                            full_mean_locked=full_locked, tied=bool(tied),
                            risk_reduction=(float(np.clip(1 - best["sd"] / base_sd, 0, 1))
                                            if base_sd > tolerance else np.nan)))
    return pd.DataFrame(summary).set_index("policy"), pd.DataFrame(frontier)


# Streamlit interface

SCENARIOS = {
    "Gaussian random walk": "gaussian",
    "Historical replay": "historical",
    "Bootstrapped random walk": "bootstrap",
}
DIRECTIONS = {
    "PLN receipts converted to EUR": "pln_receipts",
    "EUR payments funded in PLN": "eur_payments",
}
MONTHS = list(calendar.month_name)[1:]


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def read_inputs(fx_bytes, rate_bytes, refresh_fx):
    return load_inputs(fx_bytes, rate_bytes, refresh_fx=refresh_fx)


@st.cache_data(show_spinner=False, max_entries=3)
def calculate(fx, rates, options):
    return run_model(fx, rates, **options)


def format_amount(value, unit):
    if pd.isna(value):
        return "—"
    return f"{value:,.2f} {unit}"


def file_contents(upload):
    return None if upload is None else upload.getvalue()


def csv_download(label, frame, filename):
    st.download_button(
        label, frame.to_csv(index=False).encode("utf-8"), filename, "text/csv"
    )


@lru_cache(maxsize=1)
def biedronka_quarterly_ebit_pln():
    """Approximate 2025 quarterly Poland Retail EBIT in PLN using the bundled 2025 ECB daily fixings."""
    ebit_eur_m = np.diff([0, 291, 615, 963, 1_305])  # Cumulative EBIT in the Q1, H1, 9M and FY segment notes.
    spot = pd.read_csv(StringIO(ECB_SPOT_CSV), parse_dates=["date"], index_col="date").loc["2025", "spot"]
    quarter_fx = spot.groupby(spot.index.quarter).mean().reindex([1, 2, 3, 4])
    return ebit_eur_m * 1_000_000 * quarter_fx.to_numpy()


def flow_editor(unit, direction):
    editor_key = f"cashflows_{direction}"
    with st.expander("Monthly cash flows", expanded=False):
        st.write(
            f"Enter January–December amounts in {unit}. Each entry is a forecast known "
            "at the start of the year and is assumed to be received or paid in full. "
            "The historical replay applies this same seasonal cash-flow profile to each year."
        )
        if direction == "pln_receipts":
            basis = st.radio("Basis for the 2025 earnings defaults",
                             ["Estimated net profit", "Operating profit (EBIT)"],
                             horizontal=True, key="earnings_basis")
            quarterly = biedronka_quarterly_ebit_pln()
            if basis == "Estimated net profit":
                annual_net_profit = 3_623_931_554.0
                quarterly = annual_net_profit * (quarterly / quarterly.sum())
                st.caption(
                    "Jeronimo Martins Polska S.A. reported PLN 3,623,931,554 in annual net profit. "
                    "Quarterly net profit is estimated by allocating that total in proportion to "
                    "Biedronka Poland’s quarterly EBIT converted to PLN."
                )
            else:
                editor_key += "_ebit"
                st.caption("This preset uses the Polish segment’s operating profit before interest and tax.")
            monthly_defaults = np.repeat(quarterly / 3, 3)
            st.caption(
                "Quarterly EBIT was €291m, €324m, €348m and €342m. Each quarter uses its average 2025 ECB "
                "PLN/EUR rate and is split equally among its three months. You can enter actual remittances below. "
                "[Annual net profit](https://raportkrs.pl/krs/0000222483) · "
                "[Quarterly reports](https://www.jeronimomartins.com/en/investors/financial-results/) · "
                "[ECB exchange rates](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.PLN.EUR.SP00.A)"
            )
        else:
            monthly_defaults = np.full(12, 1_000_000.0)
            st.caption("EUR payments start at an illustrative EUR 1 million per month.")
        frame = pd.DataFrame({"Month": MONTHS, "Amount": monthly_defaults})
        edited = st.data_editor(
            frame,
            hide_index=True,
            disabled=["Month"],
            column_config={
                "Amount": st.column_config.NumberColumn(
                    f"Cash flow ({unit})", min_value=0.0, format="%.2f", required=True
                )
            },
            key=editor_key,
        )
    return edited["Amount"].to_numpy(dtype=float)


def show_assumptions(direction, cash_unit, report_unit, fraction, schedule, year_end_catchup, delay, band, spread,
                     mode, sigma, source_meta, result_meta, inputs, paths, seed):
    with st.expander("Assumptions, pricing and data", expanded=False):
        st.markdown("#### Cash flows and settlement")
        st.write(
            f"Amounts are fixed in {cash_unit} and the outcome is reported in {report_unit}. "
            "Receipts or payment obligations accumulate for a single December conversion. There is no interest on "
            "accumulated cash, interim discounting, tax, margin, credit loss or change "
            "in forecast amounts. Each year's opening hedge balance is zero."
        )
        st.write(
            (f"Both policies schedule a single hedge of {fraction:.1%} of each January–November flow "
             "on the last trading day of the following month. " if schedule == "once" else
             f"Both policies schedule {fraction:.1%} of each cash flow at the end of every "
             "following month until it is fully hedged or December arrives. The last installment "
             "may be smaller. For example, at 25%, January's flow has four scheduled installments "
             "in February, March, April and May. ")
            + "The first regular hedge is on the last trading day of February for January. "
            "The December cash flow always converts at spot. "
            + ("The year-end catch-up overrides the regular December date. Both policies buy "
               "forwards on the second-to-last trading day of December covering the entire "
               "remaining January–November balance, even when the regular hedge percentage "
               "is zero. The timing rule can still bring forward part of November's cash. "
               "Only December's own cash flow remains for spot conversion at settlement."
               if year_end_catchup else
               "The strict month-end schedule is active. November's scheduled trade is on "
               "the settlement date, so it converts at spot unless the timing rule brings "
               "part of it forward. Every other open amount converts at December spot.")
        )
        st.markdown("#### When the timing rule trades")
        st.write(
            "At each regular month-end decision, compare the preceding trading day's "
            "spot with the arithmetic mean of that month's daily spots through that same day. "
            "The backtest uses the next fixing as its execution-price proxy. The month-end "
            "benchmark executes its regular hedges on the last available trading day regardless "
            "of the signal. Any selected year-end catch-up has its own December execution date."
        )
        st.write(
            f"The band is ±{band:g} basis points of the average. A basis point is 0.01%. "
            "At an average of 4.4000 and a 5 bp band, the cutoffs are 4.3978 and 4.4022. "
            "Both exposure choices involve buying EUR with PLN, so a lower PLN-per-EUR rate "
            "is favorable."
        )
        st.table(pd.DataFrame({
            "Prior spot relative to the decision average": [
                "Above the upper cutoff", "Above average up to the upper cutoff",
                "From the lower cutoff through the average", "Below the lower cutoff",
            ],
            "Cumulative hedge at a regular decision": [
                "0% of this month's target", "50% of this month's target",
                "100% of this month's target",
                "100% plus up to one more current target from the next scheduled target",
            ],
        }).set_index("Prior spot relative to the decision average"))
        st.write(
            "An advance uses only the next eligible forecast and reduces its later hedge. "
            "No advance can hedge the December cash flow. Each trading day after deferral, "
            "the rule compares the preceding fixing with the original decision average, "
            "which stays fixed for that deferred amount. It fills only the additional amount "
            "required to reach 50% or 100%. Daily catch-up reviews cannot trigger new advances. "
            f"The remaining balance must be hedged by {'14 calendar days' if delay == '2w' else 'one calendar month'} "
            "after the scheduled date, or by December settlement if earlier. If that deadline "
            "is a nontrading day, the last available trading day on or before it applies. The "
            "calendar uses ECB fixing dates that are also Polish business days."
        )
        st.markdown("#### Forward and settlement prices")
        st.write(
            "Spot and forwards are PLN per EUR. Each contract has its own trade date and "
            "the same last-trading-day-of-December settlement. The forward rate equals spot "
            "times the EUR discount factor divided by the PLN discount factor for that settlement date."
        )
        st.latex(r"F_{t,D}=S_t\,\frac{DF_{\mathrm{EUR}}(t,D)}{DF_{\mathrm{PLN}}(t,D)}")
        st.write(
            f"The one-way execution spread is {spread:g} bp on every forward and spot "
            "conversion. To buy EUR, the company pays the modeled rate times "
            "(1 + spread/10,000). "
            + ("A fixed PLN receipt divided by this rate gives EUR proceeds."
               if direction == "pln_receipts" else
               "A fixed EUR amount multiplied by this rate gives the PLN cost.")
            + " These indicative prices omit the cross-currency basis and bank-specific funding charges."
        )
        if result_meta.get("rate_note"):
            st.write(result_meta["rate_note"])
        rate_kind = result_meta.get("rate_kind")
        if rate_kind in ("market", "uploaded_curve"):
            st.write(
                "At each trade, the app converts 1-, 3-, 6- and 12-month simple deposit rates to discount factors "
                "using actual days divided by 365 for PLN and 360 for EUR. It interpolates linearly in log discount "
                "factors to the December settlement date, including a discount factor of 1 at zero days. "
                "Tenors use calendar-month anniversaries of the trade date."
            )
            if rate_kind == "market":
                st.caption("This historical reconstruction uses the published monthly data available in the bundled snapshot. "
                           "For instance, a March trade uses January’s averages. A daily curve upload lets the replay "
                           "follow changes within the month.")
        elif rate_kind in ("fixed", "government", "uploaded"):
            st.write("For this source, the app holds each annual effective rate flat across maturities and compounds "
                     "it over the actual days to December divided by 365.")
        st.markdown("#### Scenario and statistics")
        if mode == "historical":
            st.write(
                "Historical outcomes use observed daily exchange rates and the same "
                "cash flows in every included calendar year. The mean gives equal weight "
                "to each full year. The standard deviation is the sample standard deviation "
                "of those annual outcomes."
            )
        else:
            st.write(
                f"Each simulated year starts from {inputs['start_spot']:.4f} PLN/EUR "
                f"with fixed effective annual rates of {inputs['pln_yield']:.3f}% (PLN) and "
                f"{inputs['eur_yield']:.3f}% (EUR). Each of the {paths:,} paths uses the "
                "same trading dates and cash-flow profile. The starting spot applies to the first trading day of January. Both strategies see exactly "
                f"the same path. The random-number seed is {seed}."
            )
            if mode == "gaussian":
                st.write(
                    f"Daily log changes are independent normal draws with mean zero and "
                    f"standard deviation {sigma:.2%}/√252. The default annual volatility "
                    "is the sample standard deviation of overlapping 12-month changes in "
                    "log month-end spot. Zero log drift implies a constant median spot."
                )
            else:
                st.write(
                    "Each daily innovation is an independent draw, with replacement, "
                    "from historical daily log changes after subtracting their sample "
                    "mean. The bootstrap uses the size and shape of observed daily returns. "
                    "It discards their original order, including volatility clustering."
                )
            st.write(
                "Expected value is the arithmetic mean across simulated years. Standard "
                "deviation measures the dispersion of annual cash outcomes across paths "
                "using the sample formula (n − 1)."
            )
        st.write(
            "Savings compare the selected policies on each matched year or path. Positive savings mean "
            "the first policy increased EUR proceeds or reduced PLN cost. The worst 5% "
            "average uses the lowest ceil(0.05 × number of outcomes) paired savings. "
            "Win frequency counts strict improvements and excludes ties from the win count."
        )
        st.markdown("#### Data used for this run")
        st.caption(
            "[ECB daily EUR/PLN reference rates](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.PLN.EUR.SP00.A)"
        )
        source_links = {
            "market": "[Eurostat monthly money-market rates](https://ec.europa.eu/eurostat/databrowser/view/irt_st_m/default/table?lang=en)",
            "policy": "[NBP reference rate history](https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml) · "
                      "[ECB main refinancing rate history](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html)",
            "government": "[Polish fitted sovereign yields](https://yieldcartography.com/curves/) · "
                          "[Bundesbank federal yield curve](https://www.bundesbank.de/en/statistics/money-and-capital-markets/"
                          "interest-rates-and-yields/daily-term-structure-on-listed-federal-securities-651570)",
        }
        if result_meta.get("rate_kind") in source_links:
            st.caption(source_links[result_meta["rate_kind"]])
        for note in (source_meta.get("fx_source"), result_meta.get("obs_note")):
            if note:
                st.write(note)


def show_results(outcomes, audit, meta, unit, direction, mode, year_end_catchup):
    noun = "proceeds" if direction == "pln_receipts" else "cost"
    history = mode == "historical"
    st.subheader(f"December {noun} in {unit}")
    st.caption(f"{meta['period']} · {meta['n']:,} {'historical years' if history else 'simulated paths'}")
    policy_keys = ["unhedged", "benchmark", "tactical"]
    suffix = " + December catch-up" if year_end_catchup else ""
    labels = ["No hedge", "Month-end benchmark" + suffix, "Page 170 timing rule" + suffix]
    sd = outcomes[policy_keys].std(ddof=1)
    means = outcomes[policy_keys].mean()
    sign = 1 if direction == "pln_receipts" else -1
    # Positive gains are additional EUR proceeds or avoided PLN cost.
    # Reconcile economically identical trades despite floating-point rounding.
    gains = {}
    for policy, reference in [("benchmark", "unhedged"), ("tactical", "unhedged"),
                              ("tactical", "benchmark")]:
        diff = sign * (outcomes[policy] - outcomes[reference])
        scale = np.maximum(outcomes[policy].abs(), outcomes[reference].abs())
        gains[policy, reference] = diff.mask(diff.abs() <= 32 * np.finfo(float).eps * scale, 0.0)
    sd_reductions = pd.Series(np.nan, index=policy_keys)
    if sd["unhedged"] > 32 * np.finfo(float).eps * abs(means["unhedged"]):
        sd_reductions = 1.0 - sd / sd["unhedged"]

    mean_name = "Historical mean" if history else "Expected value"
    gain_name = f"Mean gain vs no hedge ({unit})"
    risk_name = "SD reduction vs no hedge"
    values = pd.DataFrame({
        "Policy": labels,
        mean_name: means.to_numpy(),
        "Standard deviation": sd.to_numpy(),
        gain_name: [0.0, gains["benchmark", "unhedged"].mean(),
                    gains["tactical", "unhedged"].mean()],
        risk_name: sd_reductions.to_numpy(),
    })
    st.dataframe(
        values.style.format({mean_name: "{:,.2f}", "Standard deviation": "{:,.2f}",
                             gain_name: "{:,.2f}", risk_name: "{:.1%}"}, na_rep="—"),
        hide_index=True, width="stretch",
    )
    st.caption(
        "With no hedge, all twelve cash flows convert at the last December trading day's spot rate, "
        "including the same one-way spread. A positive gain means "
        + ("more EUR received." if direction == "pln_receipts" else "fewer PLN paid.")
        + " SD reduction is the percentage decrease in the standard deviation of December "
          "cash outcomes relative to leaving the exposure open."
    )
    input_unit = "PLN" if direction == "pln_receipts" else "EUR"
    st.caption(
        "Average cash flow locked by a forward before December settlement: "
        f"{format_amount(outcomes['benchmark_locked'].mean(), input_unit)} under the "
        f"month-end benchmark and {format_amount(outcomes['tactical_locked'].mean(), input_unit)} "
        "under the timing rule. "
        + ("The catch-up brings both policies to full coverage of January–November cash."
           if year_end_catchup else "The remainder converts at December spot.")
    )
    st.markdown("#### Paired policy comparison")
    pair_options = {
        "Page 170 vs no hedge": ("tactical", "unhedged"),
        "Month-end vs no hedge": ("benchmark", "unhedged"),
        "Page 170 vs month-end": ("tactical", "benchmark"),
    }
    selection = st.radio("Compare outcomes", list(pair_options), horizontal=True)
    policy, reference = pair_options[selection]
    advantage = gains[policy, reference]
    worst_n = max(1, int(np.ceil(0.05 * len(advantage))))
    a, b, c = st.columns(3)
    a.metric("Mean savings", format_amount(advantage.mean(), unit))
    b.metric("Outcomes with positive savings", f"{advantage.gt(0).mean():.1%}")
    c.metric("Mean savings in worst 5%", format_amount(advantage.nsmallest(worst_n).mean(), unit))
    st.caption(
        f"Positive savings mean {'higher EUR proceeds' if direction == 'pln_receipts' else 'lower PLN cost'} "
        f"for the first policy in the selected pair. Standard deviation of paired savings: "
        f"{format_amount(advantage.std(ddof=1), unit)}."
    )

    if not history:
        st.caption(
            "Monte Carlo standard error of mean savings: "
            + format_amount(advantage.std(ddof=1) / np.sqrt(len(advantage)), unit)
            + ". This measures sampling noise in the estimated mean."
        )
    if history:
        st.markdown("#### Savings by calendar year")
        year_chart = pd.DataFrame({
            "Year": outcomes["label"].astype(str),
            f"Savings ({unit})": advantage,
        })
        st.bar_chart(year_chart, x="Year", y=f"Savings ({unit})")
    else:
        st.markdown("#### Distribution of savings")
        q = np.linspace(0, 1, 101)
        percentiles = pd.DataFrame({"Percentile": np.arange(101), f"Savings ({unit})": advantage.quantile(q).to_numpy()})
        st.line_chart(percentiles, x="Percentile", y=f"Savings ({unit})")
    with st.expander("Annual outcomes and trade ledger"):
        st.write("Each row compares all three policies on the same exchange-rate path.")
        annual = outcomes.assign(
            month_end_gain_vs_no_hedge=gains["benchmark", "unhedged"],
            page_170_gain_vs_no_hedge=gains["tactical", "unhedged"],
        )
        st.dataframe(annual.head(200), hide_index=True, width="stretch")
        csv_download("Download all annual outcomes", annual, "annual_cash_outcomes.csv")
        st.markdown("**Trade ledger**")
        st.caption(str(meta["audit_label"]))
        st.dataframe(audit, hide_index=True, width="stretch")
        csv_download("Download this trade ledger", audit, "trade_ledger.csv")


def show_hedge_ratio_charts(frontier, summary, annual, cash_unit, report_unit, direction, mode, policy_names):
    noun = "proceeds" if direction == "pln_receipts" else "cost"
    mean_title = f"{'Historical mean' if mode == 'historical' else 'Expected'} December {noun}"
    sd_title = f"Standard deviation of December {noun}"
    st.markdown("#### December payout across hedge ratios")
    annual_label = "Share of annual cash flow hedged"
    assigned_label = "Share assigned to the hedge policy"
    if annual > 0:
        axis = st.radio("Horizontal axis", [annual_label, assigned_label], horizontal=True,
                        key="hedge_ratio_axis")
    else:
        axis = assigned_label
        st.caption("Annual cash flow is zero. The charts use the share assigned to each policy.")
    if axis == annual_label:
        st.caption("The hedge ratio is the average amount covered by forwards before settlement divided by "
                   "annual cash flow. Each line ends at the maximum coverage its schedule permits.")
    else:
        st.caption("At 0% all cash converts at December spot. At 100% the policy manages every cash flow "
                   "under the selected schedule. December’s own cash flow still converts at spot.")

    chart = frontier.copy()
    chart["assigned"] = chart["hedge_fraction"]
    chart["covered"] = chart["mean_locked"] / annual if annual > 0 else np.nan
    chart["locked_m"] = chart["mean_locked"] / 1e6
    chart["mean_m"] = chart["mean"] / 1e6
    chart["sd_m"] = chart["sd"] / 1e6
    chart["optimum"] = (chart["assigned"].eq(chart["policy"].map(summary["hedge_fraction"]))
                        & ~chart["policy"].map(summary["tied"]))
    chart["policy_name"] = chart["policy"].map(policy_names)
    x_field = "covered" if axis == annual_label else "assigned"
    tooltip = [
        {"field": "policy_name", "type": "nominal", "title": "Policy"},
        {"field": "assigned", "type": "quantitative", "title": assigned_label, "format": ".2%"},
    ]
    if annual > 0:
        tooltip.append({"field": "covered", "type": "quantitative", "title": annual_label, "format": ".2%"})
    tooltip.extend([
        {"field": "locked_m", "type": "quantitative", "title": f"Covered by forwards ({cash_unit} m)", "format": ",.2f"},
        {"field": "mean_m", "type": "quantitative", "title": f"{mean_title} ({report_unit} m)", "format": ",.2f"},
        {"field": "sd_m", "type": "quantitative", "title": f"{sd_title} ({report_unit} m)", "format": ",.2f"},
    ])
    if chart["optimum"].any():
        st.caption("Diamonds mark the minimum-variance allocations. Hover over a point to see both payout measures.")
    else:
        st.caption("Hover over a point to see both payout measures.")
    for panel, title, field in zip(st.columns(2), (mean_title, sd_title), ("mean_m", "sd_m")):
        spec = {
            "title": title,
            "height": 250,
            "encoding": {
                "x": {"field": x_field, "type": "quantitative", "title": axis,
                      "scale": {"domain": [0, 1]}, "axis": {"format": ".0%", "tickCount": 6}},
                "y": {"field": field, "type": "quantitative", "title": f"Million {report_unit}",
                      "scale": {"zero": False}, "axis": {"format": ",.6~g"}},
                "color": {"field": "policy_name", "type": "nominal", "title": None,
                          "sort": list(policy_names.values()),
                          "legend": {"orient": "bottom", "direction": "horizontal"}},
                "tooltip": tooltip,
            },
            "layer": [
                {"mark": {"type": "line", "strokeWidth": 2.5, "point": {"filled": True, "size": 24}},
                 "encoding": {"order": {"field": "assigned", "type": "quantitative"}}},
                {"transform": [{"filter": "datum.optimum"}],
                 "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 135,
                          "stroke": "white", "strokeWidth": 1}},
            ],
        }
        with panel:
            st.vega_lite_chart(chart, spec)


def show_optimum(outcomes, cashflows, cash_unit, report_unit, direction, mode, year_end_catchup):
    st.subheader("How much to hedge")
    noun = "EUR proceeds" if direction == "pln_receipts" else "PLN cost"
    st.write(
        f"Choose the share of each cash flow that minimizes the standard deviation of December {noun}. "
        "That share follows the selected installment schedule and timing rule. The rest converts at December spot."
    )
    summary, frontier = minimum_variance_hedges(outcomes)
    annual = float(np.sum(cashflows))
    policy_names = {"benchmark": "Month-end benchmark", "tactical": "Page 170 timing rule"}
    locked_column = f"Average hedged before settlement ({cash_unit} m)"
    mean_column = f"Mean December {'proceeds' if direction == 'pln_receipts' else 'cost'} ({report_unit} m)"
    sd_column = f"December SD ({report_unit} m)"
    percent_column = "Annual cash flow hedged"
    table = pd.DataFrame({
        "Policy": [policy_names[key] for key in summary.index],
        "Optimal share assigned": summary["hedge_fraction"].to_numpy(),
        locked_column: summary["mean_locked"].to_numpy() / 1_000_000,
        percent_column: summary["mean_locked"].to_numpy() / annual if annual else np.nan,
        mean_column: summary["mean"].to_numpy() / 1_000_000,
        sd_column: summary["sd"].to_numpy() / 1_000_000,
        "SD reduction vs no hedge": summary["risk_reduction"].to_numpy(),
    })
    st.dataframe(table.style.format({
        "Optimal share assigned": "{:.2%}", locked_column: "{:,.2f}",
        percent_column: "{:.2%}", mean_column: "{:,.2f}", sd_column: "{:,.2f}",
        "SD reduction vs no hedge": "{:.1%}",
    }, na_rep="—"), hide_index=True, width="stretch")
    base = summary.iloc[0]
    st.caption(
        f"Unhedged reference: mean {base['unhedged_mean'] / 1_000_000:,.2f} million {report_unit}, "
        f"standard deviation {base['unhedged_sd'] / 1_000_000:,.2f} million {report_unit}. "
        + ("The December catch-up covers only the share assigned to the policy. " if year_end_catchup else "")
        + "December’s own flow always converts at spot. The amount hedged counts only contracts booked before settlement."
    )
    if summary["tied"].any():
        tied_names = ", ".join(policy_names[key] for key in summary.index[summary["tied"]])
        st.caption(f"All allocations give the same risk to numerical precision for {tied_names}. The table shows 0% in this case.")
    if mode == "historical":
        st.caption("The historical optimum is fitted and evaluated on the same calendar years. Changes in each year’s starting exchange rate contribute to the measured risk.")
    else:
        st.caption("The estimated optimum depends on the simulated exchange-rate process, interest rates and cash-flow forecast.")
    show_hedge_ratio_charts(frontier, summary, annual, cash_unit, report_unit, direction, mode, policy_names)
    with st.expander("How the optimal allocation is calculated"):
        st.write(
            "For each path or historical year, let U be the unhedged outcome and P the outcome when "
            "all cash follows a policy. Assigning share h gives U + h(P − U). The calculation minimizes "
            "sample variance across those paired outcomes, subject to 0% ≤ h ≤ 100%. "
            "The installment percentage in the sidebar determines when the assigned exposure is hedged."
        )
        st.latex(r"h^*=\min\!\left(1,\max\!\left(0,-\frac{\operatorname{Cov}(U,P-U)}{\operatorname{Var}(P-U)}\right)\right)")


def main():
    st.set_page_config(page_title="EUR/PLN hedge comparison", layout="wide")
    st.title("EUR/PLN hedge comparison")
    st.write("Compare the page 170 timing rule, month-end hedging, and leaving all cash flows unhedged until December.")
    with st.sidebar:
        st.header("Scenario")
        scenario_label = st.selectbox("Exchange-rate scenario", list(SCENARIOS))
        mode = SCENARIOS[scenario_label]
        direction_label = st.radio("Cash-flow exposure", list(DIRECTIONS))
        direction = DIRECTIONS[direction_label]
        st.subheader("Hedge policy")
        schedule = st.radio(
            "Hedge schedule", ["layered", "once"],
            format_func=lambda x: "One hedge per cash flow" if x == "once" else "Monthly installments until fully hedged",
        )
        percentage = st.slider(
            "Target per scheduled hedge (% of the cash flow)", 0, 100,
            100 if schedule == "once" else 25, key=f"percentage_{schedule}",
        )
        st.caption(
            "January's hedge is scheduled in February." if schedule == "once" else
            "At 25%, January's four installments are scheduled in February–May."
        )
        year_end_catchup = st.checkbox(
            "Hedge all remaining January–November cash before settlement",
            value=True, key="year_end_catchup",
            help="Both policies hedge the remaining January–November amount on the second-to-last "
                 "December trading day. December's own cash flow always converts at spot.",
        )
        delay_label = st.radio("Maximum deferral", ["Two weeks", "One calendar month"])
        delay = "2w" if delay_label == "Two weeks" else "1m"
        band = st.number_input("Signal band (relative basis points)", 0.0, 1000.0, 5.0, 1.0)
        spread = st.number_input("One-way execution spread (basis points)", 0.0, 1000.0, 0.0, 1.0)
        with st.expander("Use your own data"):
            st.caption("Upload either file to replace its default source. Use ISO dates (YYYY-MM-DD).")
            fx_file = st.file_uploader("Daily PLN per EUR · date,spot", type="csv", key="fx_csv")
            refresh_fx = st.checkbox("Refresh default FX history online", disabled=fx_file is not None)
            rate_file = st.file_uploader(
                "Dated PLN and EUR rates (%)", type="csv", key="yield_csv",
                help="Use the date the rates became observable (YYYY-MM-DD). A rate of 4.5 means 4.5%.",
            )
            st.caption("Flat annual effective yields use this CSV header:")
            st.code("date,pln_yield,eur_yield", language=None)
            st.caption("For dated market curves, use simple annual deposit rates (PLN ACT/365, EUR ACT/360):")
            st.code("date,pln_1m,pln_3m,pln_6m,pln_12m,eur_1m,eur_3m,eur_6m,eur_12m", language=None)
    try:
        fx, rates, source_meta = read_inputs(file_contents(fx_file), file_contents(rate_file), bool(refresh_fx and fx_file is None))
        defaults = default_parameters(fx, rates)
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Unable to use these data. {exc}")
        return

    inputs = {"start_spot": float(defaults["spot"]),
              "pln_yield": float(defaults["pln_yield"]), "eur_yield": float(defaults["eur_yield"])}
    sigma = defaults["annual_vol"]
    n_paths, seed = 5_000, 42
    historical_rates = "observed"
    with st.sidebar:
        if mode == "historical":
            st.subheader("Historical forward pricing")
            historical_rates = st.radio(
                "Rates used to price each forward", ["observed", "policy", "government", "fixed"],
                format_func=lambda x: {
                    "observed": "Uploaded rate history" if rate_file is not None else "Historical market rates by maturity",
                    "policy": "Central bank rates in force on the trade date",
                    "government": "Historical one-year government yields (online)",
                    "fixed": "Fixed rates for every year",
                }[x],
            )
            if historical_rates == "observed" and rate_file is None:
                st.caption("Eurostat monthly PLN and EUR market rates at 1, 3, 6 and 12 months. "
                           "Each monthly average enters pricing after one full calendar month has elapsed.")
            elif historical_rates == "policy":
                st.caption("Use the NBP reference rate and ECB main refinancing rate effective on the trade date, held flat to December.")
            elif historical_rates == "government":
                st.caption("Fetch Polish and German one-year yields online. Each trade uses the most recent available prior observation.")
            elif historical_rates == "observed":
                st.caption("Each trade uses the most recent available uploaded observation strictly before the trade date.")
        else:
            st.subheader("Simulated FX paths")
            inputs["start_spot"] = st.number_input(
                "Starting spot (PLN per EUR)", min_value=0.0000001,
                value=inputs["start_spot"], step=0.01, format="%.4f",
            )
            if mode == "gaussian":
                vol_pct = st.number_input(
                    "Annual log volatility (%)", min_value=0.0,
                    value=100 * sigma if sigma is not None else None, step=0.5, format="%.3f",
                    placeholder="Enter an annual volatility",
                )
                if sigma is None:
                    st.caption("This FX history is too short for the 15-year calibration. Enter an annual volatility.")
                else:
                    st.caption(
                        f"Historical default {100 * sigma:.3f}% · "
                        f"{defaults['vol_start']} to {defaults['vol_end']}"
                    )
                sigma = vol_pct / 100 if vol_pct is not None else None
            n_paths = st.select_slider("Simulated paths", [100, 500, 1000, 2500, 5000, 10000, 20000], value=5000)
            seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42)
        if mode != "historical" or historical_rates == "fixed":
            inputs["pln_yield"] = st.number_input(
                "PLN effective annual rate (%)", min_value=-99.999999,
                value=inputs["pln_yield"], step=0.1, format="%.3f",
            )
            inputs["eur_yield"] = st.number_input(
                "EUR effective annual rate (%)", min_value=-99.999999,
                value=inputs["eur_yield"], step=0.1, format="%.3f",
            )
        if mode != "historical":
            forward = inputs["start_spot"] * (1 + inputs["pln_yield"] / 100) / (1 + inputs["eur_yield"] / 100)
            st.metric("Indicative one-year forward", f"{forward:.4f} PLN/EUR")
            st.caption(
                f"Default spot dated {defaults['valuation_date']} and default rates dated "
                f"{defaults['rate_date']}. The displayed forward is at mid and uses the current "
                "scenario inputs. Each December hedge uses its own remaining tenor."
            )

    cash_unit, report_unit = ("PLN", "EUR") if direction == "pln_receipts" else ("EUR", "PLN")
    cashflows = flow_editor(cash_unit, direction)
    if not np.isfinite(cashflows).all() or np.any(cashflows < 0):
        st.error("Enter a nonnegative amount for each month.")
        return
    fraction = percentage / 100
    st.write(
        f"**{cashflows.sum():,.0f} {cash_unit} annual cash flow** · "
        + (f"One {percentage}% target per January–November flow" if schedule == "once" else
           f"{percentage}% monthly installments starting the month after each flow")
        + " · December cash converts at spot. The month-end policy starts "
        "with January's cash flow at the end of February. All contracts settle on the "
        "last trading day of December."
    )
    st.caption(
        "Year-end catch-up is on. Both policies cover all January–November cash by the "
        "second-to-last December trading day. Only December's cash converts at spot."
        if year_end_catchup else
        "Strict month-end schedule. Late-year flows can reach December settlement partly "
        "or fully exposed to the spot rate."
    )
    if mode == "gaussian" and sigma is None:
        st.info("Enter annual log volatility in the sidebar to calculate the Gaussian scenario.")
        return
    options = dict(mode=mode, cashflows=tuple(float(x) for x in cashflows), direction=direction,
                   hedge_fraction=fraction, delay=delay, band_bps=band, spread_bps=spread,
                   **inputs, annual_vol=sigma, n_paths=int(n_paths), seed=int(seed),
                   sim_year=int(defaults["sim_year"]), historical_rates=historical_rates, schedule=schedule,
                   year_end_catchup=year_end_catchup)
    try:
        with st.spinner("Calculating matched policy outcomes…"):
            outcomes, audit, result_meta = calculate(fx, rates, options)
    except (ValueError, KeyError, OSError) as exc:
        st.error(str(exc))
        if mode == "historical" and historical_rates != "fixed":
            st.info(
                "Choose a rate source with coverage for the FX history, or upload your own dated rates. "
                "Selecting fixed rates runs a separate scenario using the values you enter."
            )
        show_assumptions(direction, cash_unit, report_unit, fraction, schedule, year_end_catchup, delay, band, spread,
                         mode, sigma, source_meta, {}, inputs, n_paths, seed)
        return
    comparison_tab, hedge_tab = st.tabs(["Policy comparison", "Hedge ratios"])
    with comparison_tab:
        show_results(outcomes, audit, result_meta, report_unit, direction, mode, year_end_catchup)
    with hedge_tab:
        show_optimum(outcomes, cashflows, cash_unit, report_unit, direction, mode, year_end_catchup)
    show_assumptions(direction, cash_unit, report_unit, fraction, schedule, year_end_catchup, delay, band, spread,
                     mode, sigma, source_meta, result_meta, inputs, n_paths, seed)


# Eurostat IRT_ST_M: PL and EA, IRT_M1/M3/M6/M12, monthly money-market rates (% p.a.).
# https://ec.europa.eu/eurostat/databrowser/view/irt_st_m/default/table?lang=en
# Monthly averages 2010-01 through 2026-10, snapshot retrieved 2026-11-26.
MARKET_CURVES_CSV = """month,pln_1m,pln_3m,pln_6m,pln_12m,eur_1m,eur_3m,eur_6m,eur_12m
2010-01,3.64,4.24,4.36,4.52,0.437,0.67975,0.97725,1.23215
2010-02,3.62,4.17,4.26,4.47,0.42145,0.6617,0.96455,1.2252
2010-03,3.62,4.13,4.23,4.44,0.40617,0.64496,0.95243,1.21513
2010-04,3.67,3.69,3.58,3.8,0.40465,0.6447,0.9552,1.22515
2010-05,3.62,3.85,4.01,4.26,0.423,0.68652,0.98224,1.24933
2010-06,3.63,3.86,4.0,4.24,0.44636,0.72759,1.01177,1.28132
2010-07,3.62,3.84,3.99,4.24,0.58332,0.84882,1.10423,1.37336
2010-08,3.6,3.82,4.0,4.24,0.63995,0.8955,1.14718,1.42105
2010-09,3.42,3.82,4.02,4.25,0.61814,0.8805,1.13727,1.42045
2010-10,3.61,3.83,4.04,4.26,0.78424,0.99767,1.2239,1.49538
2010-11,3.61,3.86,4.05,4.27,0.83386,1.04205,1.26905,1.5405
2010-12,3.64,3.92,4.14,4.35,0.81065,1.0217,1.2507,1.52609
2011-01,3.74,4.01,4.25,4.44,0.79281,1.01724,1.25357,1.54995
2011-02,3.87,4.11,4.34,4.51,0.89365,1.08675,1.35225,1.714
2011-03,3.9,4.18,4.4,4.57,0.90313,1.17552,1.48326,1.92409
2011-04,4.06,4.27,4.48,4.64,1.12711,1.32121,1.62137,2.08558
2011-05,4.27,4.4,4.58,4.73,1.24336,1.42509,1.70682,2.14714
2011-06,4.54,4.61,4.7,4.82,1.279,1.48859,1.74927,2.14414
2011-07,4.62,4.7,4.77,4.85,1.42181,1.59762,1.81824,2.18267
2011-08,4.61,4.72,4.83,4.87,1.37343,1.55209,1.75478,2.09691
2011-09,4.63,4.75,4.85,4.88,1.34718,1.53645,1.73595,2.06691
2011-10,4.66,4.8,4.88,4.9,1.36348,1.57586,1.7761,2.11014
2011-11,4.73,4.94,4.95,4.95,1.22659,1.48468,1.70645,2.04391
2011-12,4.76,4.98,4.99,4.99,1.14295,1.4261,1.67071,2.00352
2012-01,4.75,4.99,5.01,5.01,0.83618,1.22223,1.50473,1.83664
2012-02,4.75,4.97,4.99,4.99,0.626,1.04829,1.34538,1.67833
2012-03,4.72,4.95,4.97,4.97,0.46741,0.85845,1.16368,1.4985
2012-04,4.71,4.94,4.97,4.97,0.40884,0.74432,1.04021,1.36784
2012-05,4.86,5.05,5.08,5.08,0.39382,0.68491,0.97036,1.26605
2012-06,4.92,5.12,5.14,5.14,0.37995,0.65886,0.93452,1.21905
2012-07,4.91,5.13,5.15,5.16,0.21941,0.49705,0.77941,1.06082
2012-08,4.91,5.11,5.13,5.13,0.13157,0.33239,0.60609,0.87661
2012-09,4.9,4.95,4.96,4.98,0.11875,0.2463,0.4839,0.73985
2012-10,4.83,4.82,4.83,4.83,0.11122,0.20791,0.41183,0.65009
2012-11,4.64,4.62,4.61,4.6,0.1085,0.192,0.36041,0.58791
2012-12,4.34,4.26,4.22,4.17,0.11061,0.18547,0.32383,0.54928
2013-01,4.12,4.03,3.99,3.88,0.11255,0.20491,0.34427,0.57532
2013-02,3.87,3.8,3.76,3.67,0.12025,0.22335,0.36185,0.59415
2013-03,3.48,3.48,3.47,3.48,0.11795,0.20605,0.3288,0.54495
2013-04,3.32,3.29,3.27,3.27,0.1179,0.2089,0.32429,0.52843
2013-05,3.08,2.86,2.81,2.79,0.11232,0.20123,0.29945,0.48377
2013-06,2.74,2.69,2.82,2.68,0.12055,0.21025,0.32085,0.50705
2013-07,2.65,2.7,2.71,2.73,0.125,0.22143,0.33583,0.52539
2013-08,2.62,2.7,2.72,2.75,0.12782,0.22586,0.34223,0.54232
2013-09,2.61,2.69,2.72,2.76,0.12814,0.22324,0.34005,0.54343
2013-10,2.6,2.67,2.71,2.75,0.12826,0.22583,0.34152,0.54096
2013-11,2.6,2.65,2.7,2.75,0.13205,0.22338,0.3269,0.50638
2013-12,2.6,2.67,2.7,2.75,0.21435,0.2735,0.37055,0.5425
2014-01,2.61,2.7,2.72,2.75,0.22418,0.29205,0.396,0.56214
2014-02,2.61,2.71,2.74,2.78,0.2239,0.2881,0.3866,0.549
2014-03,2.61,2.71,2.74,2.79,0.23181,0.30533,0.4069,0.57729
2014-04,2.62,2.72,2.74,2.79,0.25255,0.32975,0.4299,0.60395
2014-05,2.62,2.72,2.74,2.8,0.25924,0.32457,0.41686,0.59229
2014-06,2.61,2.69,2.71,2.75,0.1529,0.24143,0.33319,0.51271
2014-07,2.6,2.68,2.69,2.73,0.09583,0.20504,0.30461,0.48752
2014-08,2.6,2.65,2.67,2.7,0.08548,0.19157,0.29176,0.4691
2014-09,2.52,2.45,2.44,2.43,0.01768,0.09709,0.19968,0.36232
2014-10,2.14,2.07,2.06,2.05,0.00791,0.08261,0.18365,0.33761
2014-11,2.05,2.03,2.02,2.01,0.00975,0.0809,0.18175,0.3347
2014-12,2.08,2.06,2.05,2.04,0.02305,0.0809,0.17648,0.32852
2015-01,2.05,2.03,2.02,2.01,0.00514,0.06267,0.15229,0.29819
2015-02,1.97,1.92,1.91,1.9,0.0004,0.0482,0.12585,0.25505
2015-03,1.66,1.67,1.67,1.7,-0.01032,0.02718,0.09686,0.21236
2015-04,1.64,1.65,1.66,1.68,-0.02895,0.0047,0.0728,0.18045
2015-05,1.64,1.67,1.72,1.76,-0.04942,-0.01037,0.05705,0.16542
2015-06,1.66,1.7,1.78,1.81,-0.06336,-0.01391,0.04918,0.16305
2015-07,1.66,1.72,1.79,1.83,-0.0713,-0.01874,0.04874,0.16726
2015-08,1.66,1.72,1.8,1.83,-0.08786,-0.02767,0.0439,0.16129
2015-09,1.66,1.72,1.8,1.83,-0.10518,-0.03705,0.03527,0.15359
2015-10,1.67,1.73,1.81,1.84,-0.11564,-0.05359,0.01968,0.12791
2015-11,1.67,1.73,1.8,1.82,-0.14033,-0.08762,-0.01543,0.0789
2015-12,1.66,1.72,1.78,1.8,-0.1895,-0.12632,-0.03977,0.05891
2016-01,1.62,1.71,1.76,1.78,-0.22265,-0.14615,-0.0608,0.04235
2016-02,1.58,1.69,1.74,1.76,-0.24605,-0.18357,-0.11529,-0.00786
2016-03,1.56,1.67,1.74,1.76,-0.30757,-0.22852,-0.13357,-0.01238
2016-04,1.56,1.67,1.74,1.76,-0.34119,-0.24919,-0.13795,-0.00976
2016-05,1.58,1.67,1.74,1.76,-0.348,-0.25718,-0.1445,-0.01259
2016-06,1.63,1.69,1.76,1.78,-0.35591,-0.26791,-0.16241,-0.028
2016-07,1.65,1.71,1.79,1.81,-0.36881,-0.29452,-0.18819,-0.05605
2016-08,1.65,1.71,1.79,1.81,-0.3697,-0.29822,-0.18865,-0.0483
2016-09,1.65,1.71,1.79,1.81,-0.37136,-0.30164,-0.19914,-0.05709
2016-10,1.65,1.72,1.8,1.83,-0.37143,-0.309,-0.2071,-0.06876
2016-11,1.66,1.73,1.81,1.85,-0.37277,-0.31268,-0.21482,-0.07405
2016-12,1.66,1.73,1.81,1.85,-0.37038,-0.31576,-0.2179,-0.08043
2017-01,1.66,1.73,1.81,1.85,-0.37127,-0.32555,-0.23555,-0.0945
2017-02,1.66,1.73,1.81,1.85,-0.3718,-0.3286,-0.24055,-0.1059
2017-03,1.66,1.73,1.81,1.85,-0.37239,-0.32935,-0.24074,-0.10961
2017-04,1.66,1.73,1.81,1.85,-0.37244,-0.33039,-0.2455,-0.119
2017-05,1.66,1.73,1.81,1.85,-0.37318,-0.32945,-0.25086,-0.12723
2017-06,1.66,1.73,1.81,1.85,-0.37282,-0.33,-0.26682,-0.14909
2017-07,1.66,1.73,1.81,1.85,-0.37281,-0.33043,-0.27262,-0.1541
2017-08,1.66,1.73,1.81,1.85,-0.37196,-0.32909,-0.27165,-0.15635
2017-09,1.66,1.73,1.81,1.85,-0.3721,-0.32938,-0.27262,-0.16829
2017-10,1.66,1.73,1.81,1.85,-0.37232,-0.32955,-0.27391,-0.17977
2017-11,1.66,1.73,1.81,1.85,-0.37164,-0.329,-0.27427,-0.18895
2017-12,1.65,1.72,1.81,1.85,-0.36874,-0.32789,-0.27137,-0.18974
2018-01,1.65,1.72,1.81,1.85,-0.36882,-0.32845,-0.27418,-0.18868
2018-02,1.65,1.72,1.81,1.85,-0.3696,-0.3285,-0.2749,-0.19125
2018-03,1.64,1.71,1.79,1.83,-0.37062,-0.3279,-0.27133,-0.191
2018-04,1.64,1.7,1.78,1.82,-0.37165,-0.32845,-0.2702,-0.1897
2018-05,1.64,1.7,1.78,1.82,-0.37082,-0.32523,-0.27,-0.188
2018-06,1.64,1.7,1.78,1.82,-0.37033,-0.32205,-0.26852,-0.18148
2018-07,1.64,1.7,1.78,1.82,-0.36968,-0.32073,-0.26918,-0.17955
2018-08,1.64,1.7,1.78,1.84,-0.36957,-0.319,-0.26678,-0.16926
2018-09,1.64,1.72,1.79,1.87,-0.37095,-0.31885,-0.26825,-0.1662
2018-10,1.64,1.72,1.79,1.87,-0.3697,-0.3177,-0.26404,-0.15383
2018-11,1.64,1.72,1.79,1.87,-0.36882,-0.31636,-0.25659,-0.14741
2018-12,1.64,1.72,1.79,1.87,-0.36705,-0.31189,-0.24084,-0.12868
2019-01,1.64,1.72,1.79,1.87,-0.36636,-0.308,-0.23636,-0.11586
2019-02,1.64,1.72,1.79,1.87,-0.3677,-0.30845,-0.2318,-0.1084
2019-03,1.64,1.72,1.79,1.87,-0.36748,-0.30919,-0.23024,-0.10876
2019-04,1.64,1.72,1.79,1.87,-0.36725,-0.31045,-0.23075,-0.1122
2019-05,1.64,1.72,1.79,1.87,-0.369,-0.31186,-0.23736,-0.13391
2019-06,1.64,1.72,1.79,1.86,-0.3848,-0.3289,-0.279,-0.19025
2019-07,1.64,1.72,1.79,1.86,-0.39513,-0.36487,-0.34739,-0.2833
2019-08,1.64,1.72,1.79,1.85,-0.40959,-0.40768,-0.40495,-0.35627
2019-09,1.64,1.72,1.79,1.85,-0.44824,-0.41757,-0.39386,-0.33914
2019-10,1.63,1.72,1.79,1.84,-0.45548,-0.41287,-0.36157,-0.30422
2019-11,1.63,1.71,1.79,1.84,-0.45,-0.40129,-0.33738,-0.27195
2019-12,1.63,1.7,1.79,1.84,-0.4529,-0.3947,-0.33555,-0.2614
2020-01,1.63,1.71,1.79,1.84,-0.45686,-0.39114,-0.33009,-0.25341
2020-02,1.63,1.71,1.79,1.84,-0.47315,-0.40885,-0.35525,-0.2881
2020-03,1.4,1.45,1.49,1.53,-0.47768,-0.41664,-0.36486,-0.26564
2020-04,0.79,0.83,0.84,0.89,-0.4281,-0.25405,-0.192,-0.1082
2020-05,0.62,0.66,0.67,0.72,-0.46435,-0.27195,-0.14305,-0.08135
2020-06,0.24,0.27,0.29,0.3,-0.49286,-0.376,-0.22268,-0.14745
2020-07,0.23,0.25,0.28,0.29,-0.50883,-0.44413,-0.34613,-0.27887
2020-08,0.2,0.23,0.27,0.27,-0.51733,-0.47971,-0.4329,-0.3589
2020-09,0.2,0.23,0.26,0.27,-0.52173,-0.49136,-0.46264,-0.41459
2020-10,0.2,0.22,0.25,0.26,-0.53845,-0.50914,-0.494,-0.466
2020-11,0.2,0.22,0.25,0.25,-0.54143,-0.52086,-0.509,-0.48062
2020-12,0.2,0.21,0.25,0.25,-0.56068,-0.53809,-0.51864,-0.49664
2021-01,0.19,0.21,0.25,0.25,-0.56205,-0.54715,-0.5285,-0.5047
2021-02,0.19,0.21,0.25,0.25,-0.5534,-0.5411,-0.5205,-0.50085
2021-03,0.19,0.21,0.25,0.25,-0.55343,-0.53909,-0.516,-0.48674
2021-04,0.18,0.21,0.24,0.25,-0.55505,-0.53815,-0.5158,-0.4835
2021-05,0.18,0.21,0.25,0.26,-0.5579,-0.5401,-0.513,-0.48062
2021-06,0.18,0.21,0.25,0.26,-0.55459,-0.54291,-0.51495,-0.48386
2021-07,0.18,0.21,0.25,0.26,-0.55623,-0.54482,-0.51595,-0.49073
2021-08,0.18,0.21,0.25,0.33,-0.56014,-0.54759,-0.52736,-0.49823
2021-09,0.18,0.24,0.29,0.44,-0.55764,-0.54505,-0.52241,-0.49232
2021-10,0.5,0.61,0.76,1.03,-0.56043,-0.54981,-0.52724,-0.47676
2021-11,1.33,1.57,1.9,2.26,-0.56536,-0.56736,-0.534,-0.487
2021-12,1.98,2.35,2.63,2.86,-0.59626,-0.58196,-0.54461,-0.50196
2022-01,2.43,2.79,3.17,3.43,-0.56486,-0.56014,-0.52733,-0.47667
2022-02,2.87,3.33,3.78,3.99,-0.55305,-0.5315,-0.47615,-0.33525
2022-03,3.69,4.27,4.62,4.74,-0.54443,-0.49543,-0.41765,-0.23739
2022-04,4.75,5.48,5.72,5.9,-0.53732,-0.44789,-0.31089,0.01305
2022-05,5.59,6.42,6.65,6.79,-0.54573,-0.38573,-0.14364,0.28659
2022-06,6.29,6.85,7.12,7.25,-0.52473,-0.23923,0.16168,0.85223
2022-07,6.7,7.03,7.33,7.47,-0.30643,0.03662,0.46648,0.99152
2022-08,6.85,7.04,7.3,7.44,0.02243,0.39474,0.83704,1.24948
2022-09,7.04,7.16,7.34,7.47,0.57386,1.01086,1.59609,2.233
2022-10,7.08,7.34,7.51,7.62,0.91519,1.42767,1.99738,2.62938
2022-11,7.03,7.43,7.63,7.76,1.42314,1.82523,2.32123,2.82832
2022-12,6.98,7.11,7.25,7.38,1.72433,2.06348,2.56033,3.01814
2023-01,6.9,6.95,7.03,7.13,1.98118,2.34486,2.85786,3.33705
2023-02,6.89,6.93,7.0,7.1,2.3688,2.64035,3.13535,3.5336
2023-03,6.86,6.92,6.98,7.08,2.70991,2.91057,3.26743,3.64661
2023-04,6.86,6.9,6.95,7.06,2.96089,3.17944,3.51561,3.75656
2023-05,6.85,6.9,6.95,7.06,3.15241,3.37173,3.68177,3.86223
2023-06,6.85,6.9,6.95,7.01,3.34255,3.53586,3.82527,4.00727
2023-07,6.86,6.81,6.76,6.75,3.469,3.67181,3.94248,4.14905
2023-08,6.81,6.69,6.55,6.51,3.62704,3.7803,3.94448,4.07296
2023-09,6.27,5.99,5.84,5.78,3.75771,3.88005,4.03033,4.14919
2023-10,5.87,5.68,5.58,5.56,3.86095,3.96764,4.11482,4.16
2023-11,5.81,5.77,5.73,5.69,3.84068,3.97164,4.06468,4.02209
2023-12,5.83,5.85,5.82,5.79,3.85763,3.93489,3.92695,3.67911
2024-01,5.85,5.87,5.84,5.81,3.869,3.92527,3.89168,3.60923
2024-02,5.83,5.86,5.85,5.84,3.86848,3.92319,3.90052,3.67052
2024-03,5.85,5.86,5.86,5.86,3.85295,3.9231,3.8948,3.71785
2024-04,5.84,5.86,5.87,5.86,3.85252,3.88533,3.83824,3.70333
2024-05,5.86,5.85,5.86,5.86,3.81591,3.81268,3.78664,3.68014
2024-06,5.84,5.85,5.86,5.86,3.6347,3.7245,3.7149,3.6496
2024-07,5.84,5.86,5.87,5.86,3.61752,3.68483,3.64357,3.52591
2024-08,5.84,5.85,5.86,5.85,3.59673,3.54759,3.42491,3.16614
2024-09,5.83,5.85,5.85,5.81,3.4381,3.43371,3.25781,2.93605
2024-10,5.82,5.85,5.85,5.74,3.20548,3.16661,3.00161,2.6913
2024-11,5.82,5.85,5.82,5.66,3.06643,3.00676,2.78752,2.50557
2024-12,5.83,5.85,5.79,5.6,2.89005,2.8246,2.63215,2.43605
2025-01,5.82,5.85,5.8,5.62,2.79186,2.70427,2.61441,2.52523
2025-02,5.83,5.87,5.82,5.62,2.60575,2.52495,2.4596,2.40695
2025-03,5.86,5.85,5.79,5.58,2.401,2.44243,2.38538,2.39838
2025-04,5.77,5.59,5.33,5.11,2.24265,2.2494,2.2017,2.1434
2025-05,5.43,5.26,5.02,4.84,2.0939,2.08724,2.11619,2.08057
2025-06,5.34,5.22,5.04,4.84,1.92852,1.98362,2.04967,2.0809
2025-07,5.15,5.02,4.85,4.66,1.89226,1.98635,2.05535,2.07909
2025-08,5.01,4.88,4.73,4.56,1.89,2.0211,2.08443,2.1139
2025-09,4.84,4.75,4.61,4.28,1.89723,2.02727,2.10232,2.1715
2025-10,4.65,4.56,4.41,4.3,1.90626,2.03443,2.10683,2.1873
2025-11,4.37,4.28,4.17,4.14,1.9056,2.04165,2.131,2.2169
2025-12,4.14,4.06,3.93,3.89,1.91548,2.04843,2.13948,2.26714
2026-01,4.04,3.93,3.83,3.76,1.96076,2.02773,2.13714,2.2451
2026-02,3.98,3.86,3.75,3.66,1.95225,2.0113,2.1435,2.2207
2026-03,3.84,3.82,3.82,3.83,1.93277,2.10941,2.32195,2.56514
2026-04,3.78,3.85,3.88,3.91,1.9712,2.1753,2.45375,2.7468
2026-05,3.79,3.85,3.93,3.98,1.95655,2.2261,2.5364,2.8042
2026-06,3.78,3.85,3.92,3.97,2.14632,2.339,2.5955,2.79759
2026-07,3.77,3.82,3.85,3.87,2.21404,2.42539,2.64674,2.85509
2026-08,3.8,3.83,3.84,3.85,2.22095,2.51314,2.71333,2.95367
2026-09,3.77,3.84,3.88,3.91,2.41436,2.63505,2.92159,3.247
2026-10,3.8,3.88,3.95,4.0,2.43582,2.63468,2.97468,3.23859
"""


# NBP reference rate effective dates: https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml
# ECB main refinancing operations: FM.B.U2.EUR.4F.KR.MRR_FR.LEV (ECB Data Portal).
# The paired snapshot contains an opening level, then a row whenever either policy rate changes.
POLICY_COVERAGE_END = pd.Timestamp("2026-11-24")
POLICY_CHANGES_CSV = """date,pln_yield,eur_yield
2010-01-01,3.5,1.0
2011-01-20,3.75,1.0
2011-04-06,4.0,1.0
2011-04-13,4.0,1.25
2011-05-12,4.25,1.25
2011-06-09,4.5,1.25
2011-07-13,4.5,1.5
2011-11-09,4.5,1.25
2011-12-14,4.5,1.0
2012-05-10,4.75,1.0
2012-07-11,4.75,0.75
2012-11-08,4.5,0.75
2012-12-06,4.25,0.75
2013-01-10,4.0,0.75
2013-02-07,3.75,0.75
2013-03-07,3.25,0.75
2013-05-08,3.25,0.5
2013-05-09,3.0,0.5
2013-06-06,2.75,0.5
2013-07-04,2.5,0.5
2013-11-13,2.5,0.25
2014-06-11,2.5,0.15
2014-09-10,2.5,0.05
2014-10-09,2.0,0.05
2015-03-05,1.5,0.05
2016-03-16,1.5,0.0
2020-03-18,1.0,0.0
2020-04-09,0.5,0.0
2020-05-29,0.1,0.0
2021-10-07,0.5,0.0
2021-11-04,1.25,0.0
2021-12-09,1.75,0.0
2022-01-05,2.25,0.0
2022-02-09,2.75,0.0
2022-03-09,3.5,0.0
2022-04-07,4.5,0.0
2022-05-06,5.25,0.0
2022-06-09,6.0,0.0
2022-07-08,6.5,0.0
2022-07-27,6.5,0.5
2022-09-08,6.75,0.5
2022-09-14,6.75,1.25
2022-11-02,6.75,2.0
2022-12-21,6.75,2.5
2023-02-08,6.75,3.0
2023-03-22,6.75,3.5
2023-05-10,6.75,3.75
2023-06-21,6.75,4.0
2023-08-02,6.75,4.25
2023-09-07,6.0,4.25
2023-09-20,6.0,4.5
2023-10-05,5.75,4.5
2024-06-12,5.75,4.25
2024-09-18,5.75,3.65
2024-10-23,5.75,3.4
2024-12-18,5.75,3.15
2025-02-05,5.75,2.9
2025-03-12,5.75,2.65
2025-04-23,5.75,2.4
2025-05-08,5.25,2.4
2025-06-11,5.25,2.15
2025-07-03,5.0,2.15
2025-09-04,4.75,2.15
2025-10-09,4.5,2.15
2025-11-06,4.25,2.15
2025-12-04,4.0,2.15
2026-03-05,3.75,2.15
2026-06-17,3.75,2.4
2026-09-16,3.75,2.65
"""


# ECB series EXR.D.PLN.EUR.SP00.A, daily PLN per EUR, 2010-11-01 to 2026-11-24.
# This embedded CSV lets the two simulation modes run offline.
ECB_SPOT_CSV = """date,spot
2010-11-01,3.9701
2010-11-02,3.9548
2010-11-03,3.9327
2010-11-04,3.8958
2010-11-05,3.915
2010-11-08,3.9193
2010-11-09,3.9219
2010-11-10,3.8843
2010-11-11,3.9329
2010-11-12,3.9289
2010-11-15,3.93
2010-11-16,3.9372
2010-11-17,3.964
2010-11-18,3.9375
2010-11-19,3.9345
2010-11-22,3.934
2010-11-23,3.9543
2010-11-24,3.9743
2010-11-25,3.9789
2010-11-26,4.0275
2010-11-29,4.0476
2010-11-30,4.0692
2010-12-01,4.0202
2010-12-02,4.0055
2010-12-03,3.9952
2010-12-06,4.019
2010-12-07,4.0117
2010-12-08,4.0633
2010-12-09,4.0305
2010-12-10,4.0296
2010-12-13,4.0156
2010-12-14,3.99
2010-12-15,3.9783
2010-12-16,3.979
2010-12-17,3.9798
2010-12-20,3.9966
2010-12-21,3.994
2010-12-22,3.9928
2010-12-23,3.9798
2010-12-24,3.9655
2010-12-27,3.9763
2010-12-28,3.9823
2010-12-29,3.9667
2010-12-30,3.965
2010-12-31,3.975
2011-01-03,3.9578
2011-01-04,3.9403
2011-01-05,3.8973
2011-01-06,3.8553
2011-01-07,3.8768
2011-01-10,3.9047
2011-01-11,3.8882
2011-01-12,3.8425
2011-01-13,3.8656
2011-01-14,3.88
2011-01-17,3.873
2011-01-18,3.8707
2011-01-19,3.8658
2011-01-20,3.9132
2011-01-21,3.8748
2011-01-24,3.877
2011-01-25,3.8755
2011-01-26,3.873
2011-01-27,3.8985
2011-01-28,3.9161
2011-01-31,3.9362
2011-02-01,3.9138
2011-02-02,3.9128
2011-02-03,3.9147
2011-02-04,3.9008
2011-02-07,3.874
2011-02-08,3.8856
2011-02-09,3.8943
2011-02-10,3.9337
2011-02-11,3.9193
2011-02-14,3.9436
2011-02-15,3.9287
2011-02-16,3.9093
2011-02-17,3.9075
2011-02-18,3.9145
2011-02-21,3.9277
2011-02-22,3.9655
2011-02-23,3.9583
2011-02-24,3.9975
2011-02-25,3.9709
2011-02-28,3.9548
2011-03-01,3.9725
2011-03-02,3.9845
2011-03-03,3.9826
2011-03-04,3.99
2011-03-07,3.9732
2011-03-08,3.9848
2011-03-09,3.9801
2011-03-10,3.9915
2011-03-11,4.0412
2011-03-14,4.0315
2011-03-15,4.0584
2011-03-16,4.0625
2011-03-17,4.079
2011-03-18,4.0588
2011-03-21,4.0436
2011-03-22,4.0338
2011-03-23,4.041
2011-03-24,4.0197
2011-03-25,4.0095
2011-03-28,4.0007
2011-03-29,3.9955
2011-03-30,3.988
2011-03-31,4.0106
2011-04-01,4.0398
2011-04-04,4.0325
2011-04-05,4.014
2011-04-06,3.9858
2011-04-07,3.9755
2011-04-08,3.956
2011-04-11,3.9775
2011-04-12,3.9733
2011-04-13,3.9584
2011-04-14,3.9526
2011-04-15,3.9467
2011-04-18,3.9683
2011-04-19,3.9785
2011-04-20,3.9723
2011-04-21,3.9493
2011-04-26,3.934
2011-04-27,3.9294
2011-04-28,3.9393
2011-04-29,3.9356
2011-05-02,3.9297
2011-05-03,3.9443
2011-05-04,3.9366
2011-05-05,3.9475
2011-05-06,3.9515
2011-05-09,3.9383
2011-05-10,3.9315
2011-05-11,3.9013
2011-05-12,3.915
2011-05-13,3.9173
2011-05-16,3.9303
2011-05-17,3.9288
2011-05-18,3.9263
2011-05-19,3.9236
2011-05-20,3.9266
2011-05-23,3.9485
2011-05-24,3.9473
2011-05-25,3.9678
2011-05-26,3.9675
2011-05-27,3.978
2011-05-30,3.975
2011-05-31,3.9558
2011-06-01,3.957
2011-06-02,3.9663
2011-06-03,3.957
2011-06-06,3.9612
2011-06-07,3.9487
2011-06-08,3.9572
2011-06-09,3.9558
2011-06-10,3.9354
2011-06-13,3.9332
2011-06-14,3.9302
2011-06-15,3.941
2011-06-16,3.9841
2011-06-17,3.9729
2011-06-20,3.9898
2011-06-21,3.9762
2011-06-22,3.9838
2011-06-23,3.9991
2011-06-24,3.9919
2011-06-27,4.0024
2011-06-28,4.0132
2011-06-29,3.9987
2011-06-30,3.9903
2011-07-01,3.9678
2011-07-04,3.9425
2011-07-05,3.95
2011-07-06,3.9544
2011-07-07,3.9465
2011-07-08,3.9401
2011-07-11,3.9813
2011-07-12,4.0315
2011-07-13,4.0445
2011-07-14,4.031
2011-07-15,4.0348
2011-07-18,4.044
2011-07-19,4.0283
2011-07-20,3.9917
2011-07-21,3.9955
2011-07-22,3.9815
2011-07-25,3.9984
2011-07-26,4.0035
2011-07-27,4.0058
2011-07-28,4.015
2011-07-29,4.0086
2011-08-01,3.9828
2011-08-02,4.0295
2011-08-03,4.028
2011-08-04,4.0313
2011-08-05,4.0455
2011-08-08,4.0514
2011-08-09,4.0956
2011-08-10,4.1006
2011-08-11,4.1759
2011-08-12,4.1518
2011-08-15,4.1583
2011-08-16,4.1595
2011-08-17,4.1338
2011-08-18,4.1526
2011-08-19,4.1724
2011-08-22,4.183
2011-08-23,4.1499
2011-08-24,4.1566
2011-08-25,4.1528
2011-08-26,4.1751
2011-08-29,4.1562
2011-08-30,4.1579
2011-08-31,4.1481
2011-09-01,4.1442
2011-09-02,4.1821
2011-09-05,4.2056
2011-09-06,4.2325
2011-09-07,4.2186
2011-09-08,4.2374
2011-09-09,4.3033
2011-09-12,4.3232
2011-09-13,4.3339
2011-09-14,4.358
2011-09-15,4.3535
2011-09-16,4.3445
2011-09-19,4.3645
2011-09-20,4.366
2011-09-21,4.4333
2011-09-22,4.4863
2011-09-23,4.513
2011-09-26,4.3888
2011-09-27,4.372
2011-09-28,4.4313
2011-09-29,4.4377
2011-09-30,4.405
2011-10-03,4.3815
2011-10-04,4.4069
2011-10-05,4.3953
2011-10-06,4.3768
2011-10-07,4.3762
2011-10-10,4.3185
2011-10-11,4.3284
2011-10-12,4.2941
2011-10-13,4.3168
2011-10-14,4.31
2011-10-17,4.2927
2011-10-18,4.3684
2011-10-19,4.3351
2011-10-20,4.3515
2011-10-21,4.3935
2011-10-24,4.3739
2011-10-25,4.3815
2011-10-26,4.3718
2011-10-27,4.342
2011-10-28,4.325
2011-10-31,4.3447
2011-11-01,4.4774
2011-11-02,4.3725
2011-11-03,4.3463
2011-11-04,4.3611
2011-11-07,4.3634
2011-11-08,4.3545
2011-11-09,4.397
2011-11-10,4.3825
2011-11-11,4.4235
2011-11-14,4.3956
2011-11-15,4.4145
2011-11-16,4.4276
2011-11-17,4.4388
2011-11-18,4.42
2011-11-21,4.4469
2011-11-22,4.4637
2011-11-23,4.4715
2011-11-24,4.4905
2011-11-25,4.5143
2011-11-28,4.5153
2011-11-29,4.527
2011-11-30,4.508
2011-12-01,4.505
2011-12-02,4.4726
2011-12-05,4.4687
2011-12-06,4.4548
2011-12-07,4.4715
2011-12-08,4.4763
2011-12-09,4.5149
2011-12-12,4.5395
2011-12-13,4.5608
2011-12-14,4.5607
2011-12-15,4.535
2011-12-16,4.4891
2011-12-19,4.488
2011-12-20,4.459
2011-12-21,4.4568
2011-12-22,4.4352
2011-12-23,4.4397
2011-12-27,4.404
2011-12-28,4.3947
2011-12-29,4.4411
2011-12-30,4.458
2012-01-02,4.4733
2012-01-03,4.4744
2012-01-04,4.4895
2012-01-05,4.5054
2012-01-06,4.4973
2012-01-09,4.4821
2012-01-10,4.4626
2012-01-11,4.4609
2012-01-12,4.4392
2012-01-13,4.406
2012-01-16,4.3927
2012-01-17,4.3697
2012-01-18,4.3478
2012-01-19,4.325
2012-01-20,4.3196
2012-01-23,4.2853
2012-01-24,4.2984
2012-01-25,4.2966
2012-01-26,4.247
2012-01-27,4.2207
2012-01-30,4.2545
2012-01-31,4.2243
2012-02-01,4.2005
2012-02-02,4.1997
2012-02-03,4.1932
2012-02-06,4.177
2012-02-07,4.1783
2012-02-08,4.1773
2012-02-09,4.1971
2012-02-10,4.2188
2012-02-13,4.1942
2012-02-14,4.191
2012-02-15,4.1875
2012-02-16,4.236
2012-02-17,4.1816
2012-02-20,4.168
2012-02-21,4.1837
2012-02-22,4.1866
2012-02-23,4.1803
2012-02-24,4.1665
2012-02-27,4.1738
2012-02-28,4.141
2012-02-29,4.1212
2012-03-01,4.1152
2012-03-02,4.1074
2012-03-05,4.126
2012-03-06,4.1495
2012-03-07,4.1644
2012-03-08,4.1135
2012-03-09,4.0992
2012-03-12,4.1095
2012-03-13,4.1243
2012-03-14,4.1545
2012-03-15,4.1393
2012-03-16,4.13
2012-03-19,4.1255
2012-03-20,4.127
2012-03-21,4.1525
2012-03-22,4.1645
2012-03-23,4.1682
2012-03-26,4.1376
2012-03-27,4.1367
2012-03-28,4.1568
2012-03-29,4.1612
2012-03-30,4.1522
2012-04-02,4.1445
2012-04-03,4.1357
2012-04-04,4.1548
2012-04-05,4.1707
2012-04-10,4.1706
2012-04-11,4.1937
2012-04-12,4.1802
2012-04-13,4.1808
2012-04-16,4.194
2012-04-17,4.1905
2012-04-18,4.1804
2012-04-19,4.187
2012-04-20,4.192
2012-04-23,4.2045
2012-04-24,4.2003
2012-04-25,4.1738
2012-04-26,4.182
2012-04-27,4.1788
2012-04-30,4.1709
2012-05-02,4.1693
2012-05-03,4.1629
2012-05-04,4.1848
2012-05-07,4.1958
2012-05-08,4.1905
2012-05-09,4.2154
2012-05-10,4.2292
2012-05-11,4.2434
2012-05-14,4.302
2012-05-15,4.3225
2012-05-16,4.35
2012-05-17,4.3484
2012-05-18,4.336
2012-05-21,4.333
2012-05-22,4.3296
2012-05-23,4.367
2012-05-24,4.3483
2012-05-25,4.353
2012-05-28,4.3394
2012-05-29,4.361
2012-05-30,4.3873
2012-05-31,4.3915
2012-06-01,4.4125
2012-06-04,4.399
2012-06-05,4.3816
2012-06-06,4.3498
2012-06-07,4.27
2012-06-08,4.296
2012-06-11,4.2953
2012-06-12,4.3283
2012-06-13,4.324
2012-06-14,4.315
2012-06-15,4.2984
2012-06-18,4.2807
2012-06-19,4.253
2012-06-20,4.2393
2012-06-21,4.2537
2012-06-22,4.2563
2012-06-25,4.2548
2012-06-26,4.248
2012-06-27,4.2515
2012-06-28,4.2872
2012-06-29,4.2488
2012-07-02,4.2205
2012-07-03,4.21
2012-07-04,4.2066
2012-07-05,4.206
2012-07-06,4.2139
2012-07-09,4.2333
2012-07-10,4.1982
2012-07-11,4.1697
2012-07-12,4.2081
2012-07-13,4.2113
2012-07-16,4.1844
2012-07-17,4.1705
2012-07-18,4.1795
2012-07-19,4.1584
2012-07-20,4.1626
2012-07-23,4.1968
2012-07-24,4.2133
2012-07-25,4.1936
2012-07-26,4.1506
2012-07-27,4.1275
2012-07-30,4.1207
2012-07-31,4.105
2012-08-01,4.1058
2012-08-02,4.1031
2012-08-03,4.0892
2012-08-06,4.0345
2012-08-07,4.0506
2012-08-08,4.0905
2012-08-09,4.0678
2012-08-10,4.0961
2012-08-13,4.082
2012-08-14,4.0833
2012-08-15,4.0875
2012-08-16,4.0771
2012-08-17,4.0806
2012-08-20,4.0697
2012-08-21,4.0624
2012-08-22,4.0761
2012-08-23,4.0808
2012-08-24,4.1022
2012-08-27,4.078
2012-08-28,4.1012
2012-08-29,4.1555
2012-08-30,4.1968
2012-08-31,4.1765
2012-09-03,4.1933
2012-09-04,4.1973
2012-09-05,4.2009
2012-09-06,4.1594
2012-09-07,4.1224
2012-09-10,4.1129
2012-09-11,4.1082
2012-09-12,4.0927
2012-09-13,4.1059
2012-09-14,4.0602
2012-09-17,4.1034
2012-09-18,4.1139
2012-09-19,4.1353
2012-09-20,4.1645
2012-09-21,4.1297
2012-09-24,4.154
2012-09-25,4.138
2012-09-26,4.1511
2012-09-27,4.1436
2012-09-28,4.1038
2012-10-01,4.1057
2012-10-02,4.1031
2012-10-03,4.0909
2012-10-04,4.0866
2012-10-05,4.0735
2012-10-08,4.078
2012-10-09,4.0777
2012-10-10,4.0825
2012-10-11,4.091
2012-10-12,4.0978
2012-10-15,4.0868
2012-10-16,4.092
2012-10-17,4.1029
2012-10-18,4.1027
2012-10-19,4.107
2012-10-22,4.1058
2012-10-23,4.1342
2012-10-24,4.1357
2012-10-25,4.14
2012-10-26,4.1503
2012-10-29,4.1489
2012-10-30,4.1305
2012-10-31,4.139
2012-11-01,4.127
2012-11-02,4.1088
2012-11-05,4.1226
2012-11-06,4.121
2012-11-07,4.1131
2012-11-08,4.1645
2012-11-09,4.1714
2012-11-12,4.1706
2012-11-13,4.1785
2012-11-14,4.1775
2012-11-15,4.16
2012-11-16,4.1615
2012-11-19,4.1458
2012-11-20,4.1278
2012-11-21,4.1249
2012-11-22,4.1092
2012-11-23,4.1192
2012-11-26,4.1032
2012-11-27,4.0905
2012-11-28,4.1076
2012-11-29,4.0968
2012-11-30,4.1052
2012-12-03,4.1116
2012-12-04,4.1291
2012-12-05,4.1198
2012-12-06,4.134
2012-12-07,4.1312
2012-12-10,4.1235
2012-12-11,4.0921
2012-12-12,4.0969
2012-12-13,4.0933
2012-12-14,4.0861
2012-12-17,4.089
2012-12-18,4.0928
2012-12-19,4.0736
2012-12-20,4.073
2012-12-21,4.0645
2012-12-24,4.0816
2012-12-27,4.0688
2012-12-28,4.0809
2012-12-31,4.074
2013-01-02,4.0727
2013-01-03,4.0884
2013-01-04,4.1165
2013-01-07,4.1193
2013-01-08,4.1245
2013-01-09,4.109
2013-01-10,4.0888
2013-01-11,4.1
2013-01-14,4.12
2013-01-15,4.12
2013-01-16,4.1312
2013-01-17,4.1203
2013-01-18,4.1455
2013-01-21,4.1699
2013-01-22,4.1737
2013-01-23,4.1638
2013-01-24,4.1909
2013-01-25,4.1769
2013-01-28,4.1989
2013-01-29,4.209
2013-01-30,4.1988
2013-01-31,4.1945
2013-02-01,4.1792
2013-02-04,4.164
2013-02-05,4.1815
2013-02-06,4.1868
2013-02-07,4.1885
2013-02-08,4.1592
2013-02-11,4.1537
2013-02-12,4.176
2013-02-13,4.1718
2013-02-14,4.1765
2013-02-15,4.1906
2013-02-18,4.1891
2013-02-19,4.1679
2013-02-20,4.1569
2013-02-21,4.1735
2013-02-22,4.1539
2013-02-25,4.147
2013-02-26,4.1678
2013-02-27,4.1648
2013-02-28,4.1515
2013-03-01,4.1477
2013-03-04,4.1398
2013-03-05,4.131
2013-03-06,4.1495
2013-03-07,4.1512
2013-03-08,4.138
2013-03-11,4.1472
2013-03-12,4.145
2013-03-13,4.1385
2013-03-14,4.1549
2013-03-15,4.1505
2013-03-18,4.1565
2013-03-19,4.1548
2013-03-20,4.1609
2013-03-21,4.1828
2013-03-22,4.1803
2013-03-25,4.1607
2013-03-26,4.1775
2013-03-27,4.1818
2013-03-28,4.1804
2013-04-02,4.1846
2013-04-03,4.1908
2013-04-04,4.1887
2013-04-05,4.1738
2013-04-08,4.1283
2013-04-09,4.1304
2013-04-10,4.116
2013-04-11,4.1124
2013-04-12,4.104
2013-04-15,4.1132
2013-04-16,4.1161
2013-04-17,4.1145
2013-04-18,4.1147
2013-04-19,4.1003
2013-04-22,4.1079
2013-04-23,4.1342
2013-04-24,4.1367
2013-04-25,4.1467
2013-04-26,4.159
2013-04-29,4.1309
2013-04-30,4.1504
2013-05-02,4.1465
2013-05-03,4.1407
2013-05-06,4.1515
2013-05-07,4.152
2013-05-08,4.1361
2013-05-09,4.1262
2013-05-10,4.1435
2013-05-13,4.1555
2013-05-14,4.1665
2013-05-15,4.1827
2013-05-16,4.1827
2013-05-17,4.1704
2013-05-20,4.1821
2013-05-21,4.1898
2013-05-22,4.1784
2013-05-23,4.2092
2013-05-24,4.1982
2013-05-27,4.1935
2013-05-28,4.1805
2013-05-29,4.225
2013-05-30,4.2667
2013-05-31,4.2792
2013-06-03,4.2666
2013-06-04,4.2365
2013-06-05,4.2452
2013-06-06,4.2759
2013-06-07,4.3058
2013-06-10,4.2532
2013-06-11,4.2789
2013-06-12,4.2532
2013-06-13,4.2645
2013-06-14,4.2347
2013-06-17,4.2296
2013-06-18,4.2554
2013-06-19,4.2598
2013-06-20,4.3183
2013-06-21,4.3289
2013-06-24,4.35
2013-06-25,4.319
2013-06-26,4.3445
2013-06-27,4.3203
2013-06-28,4.3376
2013-07-01,4.3329
2013-07-02,4.3329
2013-07-03,4.3365
2013-07-04,4.3083
2013-07-05,4.2847
2013-07-08,4.2959
2013-07-09,4.3263
2013-07-10,4.3325
2013-07-11,4.3349
2013-07-12,4.3193
2013-07-15,4.2855
2013-07-16,4.256
2013-07-17,4.261
2013-07-18,4.249
2013-07-19,4.2468
2013-07-22,4.2175
2013-07-23,4.2147
2013-07-24,4.2075
2013-07-25,4.2433
2013-07-26,4.2375
2013-07-29,4.2294
2013-07-30,4.2252
2013-07-31,4.237
2013-08-01,4.2346
2013-08-02,4.255
2013-08-05,4.2192
2013-08-06,4.2144
2013-08-07,4.217
2013-08-08,4.2066
2013-08-09,4.1877
2013-08-12,4.1888
2013-08-13,4.1913
2013-08-14,4.2037
2013-08-15,4.2062
2013-08-16,4.2281
2013-08-19,4.2503
2013-08-20,4.2267
2013-08-21,4.2403
2013-08-22,4.2416
2013-08-23,4.2323
2013-08-26,4.2285
2013-08-27,4.246
2013-08-28,4.293
2013-08-29,4.2825
2013-08-30,4.2633
2013-09-02,4.2548
2013-09-03,4.2707
2013-09-04,4.2663
2013-09-05,4.2867
2013-09-06,4.295
2013-09-09,4.273
2013-09-10,4.2595
2013-09-11,4.2333
2013-09-12,4.2131
2013-09-13,4.212
2013-09-16,4.1944
2013-09-17,4.2208
2013-09-18,4.2328
2013-09-19,4.1828
2013-09-20,4.2298
2013-09-23,4.2265
2013-09-24,4.2269
2013-09-25,4.2165
2013-09-26,4.228
2013-09-27,4.228
2013-09-30,4.2288
2013-10-01,4.2308
2013-10-02,4.219
2013-10-03,4.2053
2013-10-04,4.2045
2013-10-07,4.1951
2013-10-08,4.1931
2013-10-09,4.1995
2013-10-10,4.1958
2013-10-11,4.1932
2013-10-14,4.1828
2013-10-15,4.1811
2013-10-16,4.1765
2013-10-17,4.1744
2013-10-18,4.1808
2013-10-21,4.1824
2013-10-22,4.182
2013-10-23,4.1783
2013-10-24,4.1861
2013-10-25,4.1858
2013-10-28,4.1855
2013-10-29,4.1838
2013-10-30,4.1801
2013-10-31,4.1783
2013-11-01,4.1875
2013-11-04,4.1778
2013-11-05,4.1802
2013-11-06,4.1677
2013-11-07,4.1623
2013-11-08,4.1801
2013-11-11,4.2045
2013-11-12,4.2035
2013-11-13,4.202
2013-11-14,4.1855
2013-11-15,4.1829
2013-11-18,4.1756
2013-11-19,4.1829
2013-11-20,4.1867
2013-11-21,4.1923
2013-11-22,4.1978
2013-11-25,4.1916
2013-11-26,4.1987
2013-11-27,4.2036
2013-11-28,4.1942
2013-11-29,4.206
2013-12-02,4.1928
2013-12-03,4.2032
2013-12-04,4.1983
2013-12-05,4.1894
2013-12-06,4.1938
2013-12-09,4.1867
2013-12-10,4.1825
2013-12-11,4.1831
2013-12-12,4.1823
2013-12-13,4.1796
2013-12-16,4.1758
2013-12-17,4.179
2013-12-18,4.1783
2013-12-19,4.1739
2013-12-20,4.1653
2013-12-23,4.1597
2013-12-24,4.1422
2013-12-27,4.1515
2013-12-30,4.1487
2013-12-31,4.1543
2014-01-02,4.1693
2014-01-03,4.1658
2014-01-06,4.1764
2014-01-07,4.1776
2014-01-08,4.1767
2014-01-09,4.1755
2014-01-10,4.1749
2014-01-13,4.1542
2014-01-14,4.1507
2014-01-15,4.1575
2014-01-16,4.1692
2014-01-17,4.1645
2014-01-20,4.1578
2014-01-21,4.1651
2014-01-22,4.165
2014-01-23,4.1679
2014-01-24,4.208
2014-01-27,4.1974
2014-01-28,4.1913
2014-01-29,4.213
2014-01-30,4.2312
2014-01-31,4.2488
2014-02-03,4.2242
2014-02-04,4.2045
2014-02-05,4.1944
2014-02-06,4.188
2014-02-07,4.1875
2014-02-10,4.1803
2014-02-11,4.1825
2014-02-12,4.1658
2014-02-13,4.1675
2014-02-14,4.1495
2014-02-17,4.1471
2014-02-18,4.1538
2014-02-19,4.1734
2014-02-20,4.1801
2014-02-21,4.1661
2014-02-24,4.1578
2014-02-25,4.1525
2014-02-26,4.1611
2014-02-27,4.1779
2014-02-28,4.1676
2014-03-03,4.1928
2014-03-04,4.1865
2014-03-05,4.1795
2014-03-06,4.1845
2014-03-07,4.192
2014-03-10,4.2012
2014-03-11,4.217
2014-03-12,4.2272
2014-03-13,4.2231
2014-03-14,4.2423
2014-03-17,4.222
2014-03-18,4.2092
2014-03-19,4.1982
2014-03-20,4.2026
2014-03-21,4.1975
2014-03-24,4.1925
2014-03-25,4.1983
2014-03-26,4.1803
2014-03-27,4.1795
2014-03-28,4.1739
2014-03-31,4.1719
2014-04-01,4.1779
2014-04-02,4.1768
2014-04-03,4.1727
2014-04-04,4.165
2014-04-07,4.1746
2014-04-08,4.1704
2014-04-09,4.1708
2014-04-10,4.1698
2014-04-11,4.1837
2014-04-14,4.1824
2014-04-15,4.1868
2014-04-16,4.1956
2014-04-17,4.1937
2014-04-22,4.1906
2014-04-23,4.1926
2014-04-24,4.1914
2014-04-25,4.2028
2014-04-28,4.2069
2014-04-29,4.1994
2014-04-30,4.2024
2014-05-02,4.2035
2014-05-05,4.2087
2014-05-06,4.2019
2014-05-07,4.1998
2014-05-08,4.1932
2014-05-09,4.1823
2014-05-12,4.1818
2014-05-13,4.1854
2014-05-14,4.1884
2014-05-15,4.1792
2014-05-16,4.1904
2014-05-19,4.1868
2014-05-20,4.1893
2014-05-21,4.1888
2014-05-22,4.1765
2014-05-23,4.1631
2014-05-26,4.155
2014-05-27,4.1637
2014-05-28,4.1564
2014-05-29,4.1446
2014-05-30,4.1411
2014-06-02,4.1393
2014-06-03,4.148
2014-06-04,4.1454
2014-06-05,4.1181
2014-06-06,4.1153
2014-06-09,4.0997
2014-06-10,4.1058
2014-06-11,4.1138
2014-06-12,4.104
2014-06-13,4.1265
2014-06-16,4.1425
2014-06-17,4.1371
2014-06-18,4.1444
2014-06-19,4.1326
2014-06-20,4.1606
2014-06-23,4.1498
2014-06-24,4.154
2014-06-25,4.1504
2014-06-26,4.1431
2014-06-27,4.1522
2014-06-30,4.1568
2014-07-01,4.1582
2014-07-02,4.1456
2014-07-03,4.1471
2014-07-04,4.1394
2014-07-07,4.1432
2014-07-08,4.1344
2014-07-09,4.1284
2014-07-10,4.142
2014-07-11,4.1427
2014-07-14,4.1388
2014-07-15,4.1408
2014-07-16,4.1373
2014-07-17,4.144
2014-07-18,4.1476
2014-07-21,4.1511
2014-07-22,4.1473
2014-07-23,4.134
2014-07-24,4.1408
2014-07-25,4.1435
2014-07-28,4.1437
2014-07-29,4.1516
2014-07-30,4.1516
2014-07-31,4.1691
2014-08-01,4.1888
2014-08-04,4.1716
2014-08-05,4.1747
2014-08-06,4.2045
2014-08-07,4.2038
2014-08-08,4.201
2014-08-11,4.1991
2014-08-12,4.206
2014-08-13,4.1945
2014-08-14,4.1799
2014-08-15,4.1805
2014-08-18,4.1883
2014-08-19,4.1893
2014-08-20,4.1828
2014-08-21,4.1891
2014-08-22,4.1863
2014-08-25,4.1785
2014-08-26,4.1828
2014-08-27,4.1967
2014-08-28,4.2145
2014-08-29,4.2171
2014-09-01,4.2082
2014-09-02,4.2141
2014-09-03,4.1905
2014-09-04,4.1854
2014-09-05,4.1932
2014-09-08,4.177
2014-09-09,4.203
2014-09-10,4.2032
2014-09-11,4.1986
2014-09-12,4.1977
2014-09-15,4.198
2014-09-16,4.1956
2014-09-17,4.1876
2014-09-18,4.193
2014-09-19,4.1877
2014-09-22,4.1816
2014-09-23,4.1754
2014-09-24,4.1695
2014-09-25,4.1771
2014-09-26,4.1805
2014-09-29,4.1827
2014-09-30,4.1776
2014-10-01,4.1815
2014-10-02,4.1761
2014-10-03,4.1789
2014-10-06,4.1844
2014-10-07,4.184
2014-10-08,4.1919
2014-10-09,4.1758
2014-10-10,4.1835
2014-10-13,4.1916
2014-10-14,4.2032
2014-10-15,4.2081
2014-10-16,4.2286
2014-10-17,4.2318
2014-10-20,4.2227
2014-10-21,4.2184
2014-10-22,4.2246
2014-10-23,4.229
2014-10-24,4.2235
2014-10-27,4.2236
2014-10-28,4.2318
2014-10-29,4.2252
2014-10-30,4.2168
2014-10-31,4.2177
2014-11-03,4.2188
2014-11-04,4.2248
2014-11-05,4.2336
2014-11-06,4.2214
2014-11-07,4.2284
2014-11-10,4.2147
2014-11-11,4.2235
2014-11-12,4.2192
2014-11-13,4.2282
2014-11-14,4.2295
2014-11-17,4.2238
2014-11-18,4.2141
2014-11-19,4.2184
2014-11-20,4.214
2014-11-21,4.2047
2014-11-24,4.1943
2014-11-25,4.1868
2014-11-26,4.1801
2014-11-27,4.1798
2014-11-28,4.1839
2014-12-01,4.1782
2014-12-02,4.1637
2014-12-03,4.1598
2014-12-04,4.1614
2014-12-05,4.1628
2014-12-08,4.1611
2014-12-09,4.1602
2014-12-10,4.1619
2014-12-11,4.1798
2014-12-12,4.1735
2014-12-15,4.1782
2014-12-16,4.2061
2014-12-17,4.2258
2014-12-18,4.2422
2014-12-19,4.2738
2014-12-22,4.2665
2014-12-23,4.2765
2014-12-24,4.3078
2014-12-29,4.3023
2014-12-30,4.3103
2014-12-31,4.2732
2015-01-02,4.3053
2015-01-05,4.2979
2015-01-06,4.3075
2015-01-07,4.3108
2015-01-08,4.2902
2015-01-09,4.2699
2015-01-12,4.2765
2015-01-13,4.2824
2015-01-14,4.2863
2015-01-15,4.3008
2015-01-16,4.3179
2015-01-19,4.3164
2015-01-20,4.3275
2015-01-21,4.3058
2015-01-22,4.2997
2015-01-23,4.2279
2015-01-26,4.2157
2015-01-27,4.23
2015-01-28,4.2365
2015-01-29,4.2324
2015-01-30,4.2075
2015-02-02,4.1771
2015-02-03,4.162
2015-02-04,4.1607
2015-02-05,4.1655
2015-02-06,4.165
2015-02-09,4.1822
2015-02-10,4.2008
2015-02-11,4.218
2015-02-12,4.1916
2015-02-13,4.1768
2015-02-16,4.1832
2015-02-17,4.1872
2015-02-18,4.1909
2015-02-19,4.1695
2015-02-20,4.1812
2015-02-23,4.1746
2015-02-24,4.1662
2015-02-25,4.1615
2015-02-26,4.1535
2015-02-27,4.1524
2015-03-02,4.1563
2015-03-03,4.1582
2015-03-04,4.1727
2015-03-05,4.1397
2015-03-06,4.1178
2015-03-09,4.1181
2015-03-10,4.1368
2015-03-11,4.1365
2015-03-12,4.1371
2015-03-13,4.1539
2015-03-16,4.1281
2015-03-17,4.1392
2015-03-18,4.1466
2015-03-19,4.1288
2015-03-20,4.1272
2015-03-23,4.1215
2015-03-24,4.1131
2015-03-25,4.0858
2015-03-26,4.0763
2015-03-27,4.0978
2015-03-30,4.0938
2015-03-31,4.0854
2015-04-01,4.0608
2015-04-02,4.0647
2015-04-07,4.0578
2015-04-08,4.0171
2015-04-09,4.0167
2015-04-10,4.0181
2015-04-13,4.0136
2015-04-14,4.0105
2015-04-15,4.0178
2015-04-16,4.031
2015-04-17,4.0187
2015-04-20,3.9891
2015-04-21,3.9683
2015-04-22,4.0022
2015-04-23,4.0185
2015-04-24,4.0131
2015-04-27,4.0037
2015-04-28,4.0015
2015-04-29,4.012
2015-04-30,4.025
2015-05-04,4.048
2015-05-05,4.0182
2015-05-06,4.0483
2015-05-07,4.0461
2015-05-08,4.0679
2015-05-11,4.0668
2015-05-12,4.1028
2015-05-13,4.0872
2015-05-14,4.0827
2015-05-15,4.0432
2015-05-18,4.0615
2015-05-19,4.0486
2015-05-20,4.0653
2015-05-21,4.0959
2015-05-22,4.1003
2015-05-25,4.1075
2015-05-26,4.1391
2015-05-27,4.1335
2015-05-28,4.1289
2015-05-29,4.1298
2015-06-01,4.1335
2015-06-02,4.1288
2015-06-03,4.1261
2015-06-04,4.1692
2015-06-05,4.1441
2015-06-08,4.1551
2015-06-09,4.1657
2015-06-10,4.15
2015-06-11,4.1405
2015-06-12,4.1497
2015-06-15,4.1573
2015-06-16,4.1448
2015-06-17,4.1576
2015-06-18,4.1667
2015-06-19,4.1729
2015-06-22,4.1625
2015-06-23,4.1697
2015-06-24,4.1671
2015-06-25,4.1727
2015-06-26,4.1757
2015-06-29,4.1893
2015-06-30,4.1911
2015-07-01,4.1901
2015-07-02,4.1927
2015-07-03,4.1955
2015-07-06,4.2005
2015-07-07,4.1949
2015-07-08,4.2237
2015-07-09,4.2132
2015-07-10,4.1756
2015-07-13,4.1469
2015-07-14,4.15
2015-07-15,4.129
2015-07-16,4.1084
2015-07-17,4.1053
2015-07-20,4.1173
2015-07-21,4.1138
2015-07-22,4.1208
2015-07-23,4.1229
2015-07-24,4.1286
2015-07-27,4.1285
2015-07-28,4.1216
2015-07-29,4.1392
2015-07-30,4.1426
2015-07-31,4.1435
2015-08-03,4.1498
2015-08-04,4.1498
2015-08-05,4.1697
2015-08-06,4.1846
2015-08-07,4.1853
2015-08-10,4.1968
2015-08-11,4.1973
2015-08-12,4.1889
2015-08-13,4.1882
2015-08-14,4.1862
2015-08-17,4.1708
2015-08-18,4.1626
2015-08-19,4.1731
2015-08-20,4.1904
2015-08-21,4.2038
2015-08-24,4.2428
2015-08-25,4.2242
2015-08-26,4.231
2015-08-27,4.2357
2015-08-28,4.2405
2015-08-31,4.2289
2015-09-01,4.2375
2015-09-02,4.2359
2015-09-03,4.233
2015-09-04,4.2281
2015-09-07,4.2298
2015-09-08,4.2268
2015-09-09,4.2067
2015-09-10,4.2125
2015-09-11,4.208
2015-09-14,4.2097
2015-09-15,4.2011
2015-09-16,4.197
2015-09-17,4.2036
2015-09-18,4.203
2015-09-21,4.1866
2015-09-22,4.197
2015-09-23,4.2036
2015-09-24,4.23
2015-09-25,4.2218
2015-09-28,4.2343
2015-09-29,4.2356
2015-09-30,4.2448
2015-10-01,4.2459
2015-10-02,4.2452
2015-10-05,4.2449
2015-10-06,4.2487
2015-10-07,4.23
2015-10-08,4.2321
2015-10-09,4.217
2015-10-12,4.2246
2015-10-13,4.2391
2015-10-14,4.2378
2015-10-15,4.2291
2015-10-16,4.2383
2015-10-19,4.2367
2015-10-20,4.2466
2015-10-21,4.2757
2015-10-22,4.2849
2015-10-23,4.2529
2015-10-26,4.2678
2015-10-27,4.2919
2015-10-28,4.2936
2015-10-29,4.2699
2015-10-30,4.2644
2015-11-02,4.2532
2015-11-03,4.254
2015-11-04,4.2377
2015-11-05,4.2392
2015-11-06,4.2544
2015-11-09,4.263
2015-11-10,4.2432
2015-11-11,4.2215
2015-11-12,4.227
2015-11-13,4.2429
2015-11-16,4.2408
2015-11-17,4.2475
2015-11-18,4.254
2015-11-19,4.25
2015-11-20,4.2411
2015-11-23,4.2403
2015-11-24,4.2607
2015-11-25,4.2603
2015-11-26,4.2724
2015-11-27,4.2631
2015-11-30,4.2721
2015-12-01,4.2679
2015-12-02,4.2728
2015-12-03,4.2859
2015-12-04,4.3131
2015-12-07,4.3133
2015-12-08,4.3357
2015-12-09,4.3404
2015-12-10,4.3389
2015-12-11,4.3456
2015-12-14,4.3658
2015-12-15,4.3537
2015-12-16,4.3186
2015-12-17,4.2959
2015-12-18,4.2806
2015-12-21,4.2402
2015-12-22,4.2377
2015-12-23,4.2469
2015-12-24,4.2412
2015-12-28,4.2455
2015-12-29,4.2364
2015-12-30,4.24
2015-12-31,4.2639
2016-01-04,4.2955
2016-01-05,4.3084
2016-01-06,4.3374
2016-01-07,4.3512
2016-01-08,4.3523
2016-01-11,4.3646
2016-01-12,4.3568
2016-01-13,4.3458
2016-01-14,4.373
2016-01-15,4.4129
2016-01-18,4.4606
2016-01-19,4.4384
2016-01-20,4.4702
2016-01-21,4.4943
2016-01-22,4.4587
2016-01-25,4.469
2016-01-26,4.4942
2016-01-27,4.471
2016-01-28,4.4534
2016-01-29,4.4398
2016-02-01,4.3952
2016-02-02,4.4055
2016-02-03,4.3982
2016-02-04,4.4152
2016-02-05,4.4134
2016-02-08,4.4352
2016-02-09,4.453
2016-02-10,4.4289
2016-02-11,4.4485
2016-02-12,4.4109
2016-02-15,4.3945
2016-02-16,4.4062
2016-02-17,4.3979
2016-02-18,4.3817
2016-02-19,4.3777
2016-02-22,4.3646
2016-02-23,4.3668
2016-02-24,4.368
2016-02-25,4.3581
2016-02-26,4.3631
2016-02-29,4.3543
2016-03-01,4.3468
2016-03-02,4.3297
2016-03-03,4.3339
2016-03-04,4.3313
2016-03-07,4.3279
2016-03-08,4.3275
2016-03-09,4.3149
2016-03-10,4.2921
2016-03-11,4.3099
2016-03-14,4.2837
2016-03-15,4.2889
2016-03-16,4.3023
2016-03-17,4.2895
2016-03-18,4.2625
2016-03-21,4.2581
2016-03-22,4.2612
2016-03-23,4.2563
2016-03-24,4.2788
2016-03-29,4.2498
2016-03-30,4.2538
2016-03-31,4.2576
2016-04-01,4.244
2016-04-04,4.234
2016-04-05,4.2448
2016-04-06,4.2636
2016-04-07,4.2833
2016-04-08,4.2953
2016-04-11,4.2793
2016-04-12,4.2923
2016-04-13,4.2869
2016-04-14,4.2992
2016-04-15,4.2967
2016-04-18,4.3059
2016-04-19,4.3027
2016-04-20,4.2875
2016-04-21,4.3085
2016-04-22,4.3506
2016-04-25,4.3911
2016-04-26,4.3799
2016-04-27,4.3918
2016-04-28,4.388
2016-04-29,4.3965
2016-05-02,4.3833
2016-05-03,4.3844
2016-05-04,4.3936
2016-05-05,4.4234
2016-05-06,4.4198
2016-05-09,4.4142
2016-05-10,4.4201
2016-05-11,4.4244
2016-05-12,4.4193
2016-05-13,4.3929
2016-05-16,4.3718
2016-05-17,4.3621
2016-05-18,4.3885
2016-05-19,4.3928
2016-05-20,4.4203
2016-05-23,4.4367
2016-05-24,4.4358
2016-05-25,4.4205
2016-05-26,4.4108
2016-05-27,4.3912
2016-05-30,4.394
2016-05-31,4.3865
2016-06-01,4.3978
2016-06-02,4.3903
2016-06-03,4.3861
2016-06-06,4.3776
2016-06-07,4.3486
2016-06-08,4.3228
2016-06-09,4.3321
2016-06-10,4.3591
2016-06-13,4.3884
2016-06-14,4.4332
2016-06-15,4.4119
2016-06-16,4.449
2016-06-17,4.4463
2016-06-20,4.3925
2016-06-21,4.3991
2016-06-22,4.3785
2016-06-23,4.3571
2016-06-24,4.455
2016-06-27,4.4619
2016-06-28,4.4413
2016-06-29,4.4261
2016-06-30,4.4362
2016-07-01,4.3966
2016-07-04,4.4304
2016-07-05,4.4457
2016-07-06,4.4562
2016-07-07,4.4275
2016-07-08,4.4278
2016-07-11,4.421
2016-07-12,4.4171
2016-07-13,4.4055
2016-07-14,4.4115
2016-07-15,4.4099
2016-07-18,4.3805
2016-07-19,4.3729
2016-07-20,4.3758
2016-07-21,4.3692
2016-07-22,4.3554
2016-07-25,4.3578
2016-07-26,4.3629
2016-07-27,4.3733
2016-07-28,4.3636
2016-07-29,4.363
2016-08-01,4.351
2016-08-02,4.3241
2016-08-03,4.313
2016-08-04,4.294
2016-08-05,4.2841
2016-08-08,4.2786
2016-08-09,4.2669
2016-08-10,4.2684
2016-08-11,4.2604
2016-08-12,4.2658
2016-08-15,4.2695
2016-08-16,4.2696
2016-08-17,4.2907
2016-08-18,4.2925
2016-08-19,4.2992
2016-08-22,4.3019
2016-08-23,4.3126
2016-08-24,4.3039
2016-08-25,4.3123
2016-08-26,4.3244
2016-08-29,4.3314
2016-08-30,4.3436
2016-08-31,4.3533
2016-09-01,4.3645
2016-09-02,4.3914
2016-09-05,4.3541
2016-09-06,4.3388
2016-09-07,4.3186
2016-09-08,4.3103
2016-09-09,4.3345
2016-09-12,4.3494
2016-09-13,4.3478
2016-09-14,4.342
2016-09-15,4.3254
2016-09-16,4.3167
2016-09-19,4.2925
2016-09-20,4.2975
2016-09-21,4.2915
2016-09-22,4.2852
2016-09-23,4.2869
2016-09-26,4.3045
2016-09-27,4.2887
2016-09-28,4.2893
2016-09-29,4.3065
2016-09-30,4.3192
2016-10-03,4.2933
2016-10-04,4.2936
2016-10-05,4.2941
2016-10-06,4.2928
2016-10-07,4.2813
2016-10-10,4.2821
2016-10-11,4.2712
2016-10-12,4.2974
2016-10-13,4.302
2016-10-14,4.302
2016-10-17,4.3281
2016-10-18,4.3184
2016-10-19,4.318
2016-10-20,4.3282
2016-10-21,4.3251
2016-10-24,4.3166
2016-10-25,4.3042
2016-10-26,4.3245
2016-10-27,4.3292
2016-10-28,4.3307
2016-10-31,4.3278
2016-11-01,4.3085
2016-11-02,4.3255
2016-11-03,4.3186
2016-11-04,4.3188
2016-11-07,4.3301
2016-11-08,4.3283
2016-11-09,4.3388
2016-11-10,4.3588
2016-11-11,4.4081
2016-11-14,4.406
2016-11-15,4.4245
2016-11-16,4.4433
2016-11-17,4.4448
2016-11-18,4.4429
2016-11-21,4.4307
2016-11-22,4.4205
2016-11-23,4.4154
2016-11-24,4.4239
2016-11-25,4.419
2016-11-28,4.4194
2016-11-29,4.4299
2016-11-30,4.4483
2016-12-01,4.4676
2016-12-02,4.4892
2016-12-05,4.5005
2016-12-06,4.478
2016-12-07,4.428
2016-12-08,4.454
2016-12-09,4.4454
2016-12-12,4.453
2016-12-13,4.4442
2016-12-14,4.4402
2016-12-15,4.445
2016-12-16,4.42
2016-12-19,4.411
2016-12-20,4.4095
2016-12-21,4.4082
2016-12-22,4.4096
2016-12-23,4.4098
2016-12-27,4.409
2016-12-28,4.4028
2016-12-29,4.4141
2016-12-30,4.4103
2017-01-02,4.4123
2017-01-03,4.3934
2017-01-04,4.3778
2017-01-05,4.368
2017-01-06,4.3552
2017-01-09,4.3777
2017-01-10,4.3721
2017-01-11,4.3703
2017-01-12,4.3765
2017-01-13,4.3739
2017-01-16,4.3729
2017-01-17,4.3693
2017-01-18,4.3673
2017-01-19,4.3728
2017-01-20,4.3699
2017-01-23,4.3719
2017-01-24,4.3725
2017-01-25,4.3641
2017-01-26,4.3435
2017-01-27,4.34
2017-01-30,4.331
2017-01-31,4.3239
2017-02-01,4.3186
2017-02-02,4.3071
2017-02-03,4.2972
2017-02-06,4.2767
2017-02-07,4.3035
2017-02-08,4.3118
2017-02-09,4.3096
2017-02-10,4.2953
2017-02-13,4.3137
2017-02-14,4.3012
2017-02-15,4.3015
2017-02-16,4.3176
2017-02-17,4.3305
2017-02-20,4.3246
2017-02-21,4.3064
2017-02-22,4.2987
2017-02-23,4.308
2017-02-24,4.3107
2017-02-27,4.3148
2017-02-28,4.3148
2017-03-01,4.2958
2017-03-02,4.2941
2017-03-03,4.3007
2017-03-06,4.3091
2017-03-07,4.3075
2017-03-08,4.313
2017-03-09,4.3171
2017-03-10,4.3259
2017-03-13,4.3376
2017-03-14,4.3221
2017-03-15,4.3206
2017-03-16,4.2974
2017-03-17,4.2982
2017-03-20,4.282
2017-03-21,4.2628
2017-03-22,4.2816
2017-03-23,4.2685
2017-03-24,4.2695
2017-03-27,4.2641
2017-03-28,4.247
2017-03-29,4.2389
2017-03-30,4.2233
2017-03-31,4.2265
2017-04-03,4.2279
2017-04-04,4.2571
2017-04-05,4.2315
2017-04-06,4.2271
2017-04-07,4.2249
2017-04-10,4.2296
2017-04-11,4.2414
2017-04-12,4.2485
2017-04-13,4.245
2017-04-18,4.2363
2017-04-19,4.2358
2017-04-20,4.2588
2017-04-21,4.2727
2017-04-24,4.242
2017-04-25,4.2273
2017-04-26,4.2241
2017-04-27,4.2216
2017-04-28,4.2224
2017-05-02,4.2088
2017-05-03,4.1913
2017-05-04,4.2107
2017-05-05,4.2173
2017-05-08,4.2011
2017-05-09,4.2223
2017-05-10,4.218
2017-05-11,4.2242
2017-05-12,4.217
2017-05-15,4.2089
2017-05-16,4.1984
2017-05-17,4.1867
2017-05-18,4.2232
2017-05-19,4.2005
2017-05-22,4.1927
2017-05-23,4.198
2017-05-24,4.1867
2017-05-25,4.1793
2017-05-26,4.183
2017-05-29,4.1797
2017-05-30,4.176
2017-05-31,4.1712
2017-06-01,4.183
2017-06-02,4.1925
2017-06-05,4.1838
2017-06-06,4.1992
2017-06-07,4.1913
2017-06-08,4.2065
2017-06-09,4.1903
2017-06-12,4.1899
2017-06-13,4.1936
2017-06-14,4.1967
2017-06-15,4.2122
2017-06-16,4.2184
2017-06-19,4.2125
2017-06-20,4.2306
2017-06-21,4.2368
2017-06-22,4.2402
2017-06-23,4.232
2017-06-26,4.2108
2017-06-27,4.2151
2017-06-28,4.2375
2017-06-29,4.2489
2017-06-30,4.2259
2017-07-03,4.2355
2017-07-04,4.2426
2017-07-05,4.2463
2017-07-06,4.2478
2017-07-07,4.2322
2017-07-10,4.234
2017-07-11,4.246
2017-07-12,4.2444
2017-07-13,4.2332
2017-07-14,4.2172
2017-07-17,4.208
2017-07-18,4.2029
2017-07-19,4.2127
2017-07-20,4.21
2017-07-21,4.2364
2017-07-24,4.2408
2017-07-25,4.2571
2017-07-26,4.2613
2017-07-27,4.2541
2017-07-28,4.2493
2017-07-31,4.2493
2017-08-01,4.2563
2017-08-02,4.254
2017-08-03,4.2526
2017-08-04,4.241
2017-08-07,4.2518
2017-08-08,4.2554
2017-08-09,4.267
2017-08-10,4.2717
2017-08-11,4.2888
2017-08-14,4.2816
2017-08-15,4.2791
2017-08-16,4.2749
2017-08-17,4.2575
2017-08-18,4.2807
2017-08-21,4.2738
2017-08-22,4.2818
2017-08-23,4.2848
2017-08-24,4.2754
2017-08-25,4.2598
2017-08-28,4.2581
2017-08-29,4.2676
2017-08-30,4.2598
2017-08-31,4.2582
2017-09-01,4.2406
2017-09-04,4.2406
2017-09-05,4.2361
2017-09-06,4.2415
2017-09-07,4.2433
2017-09-08,4.2534
2017-09-11,4.244
2017-09-12,4.2549
2017-09-13,4.2798
2017-09-14,4.2843
2017-09-15,4.2761
2017-09-18,4.2884
2017-09-19,4.2785
2017-09-20,4.2774
2017-09-21,4.2805
2017-09-22,4.2672
2017-09-25,4.2681
2017-09-26,4.2824
2017-09-27,4.3027
2017-09-28,4.3122
2017-09-29,4.3042
2017-10-02,4.3132
2017-10-03,4.3102
2017-10-04,4.2971
2017-10-05,4.3
2017-10-06,4.3039
2017-10-09,4.3077
2017-10-10,4.2931
2017-10-11,4.2851
2017-10-12,4.273
2017-10-13,4.2555
2017-10-16,4.2378
2017-10-17,4.2298
2017-10-18,4.2277
2017-10-19,4.2392
2017-10-20,4.2363
2017-10-23,4.2266
2017-10-24,4.2403
2017-10-25,4.2355
2017-10-26,4.235
2017-10-27,4.2462
2017-10-30,4.2432
2017-10-31,4.244
2017-11-01,4.2338
2017-11-02,4.2376
2017-11-03,4.2383
2017-11-06,4.2408
2017-11-07,4.2396
2017-11-08,4.2341
2017-11-09,4.2346
2017-11-10,4.2308
2017-11-13,4.2349
2017-11-14,4.2381
2017-11-15,4.2473
2017-11-16,4.234
2017-11-17,4.2342
2017-11-20,4.2311
2017-11-21,4.229
2017-11-22,4.2155
2017-11-23,4.2124
2017-11-24,4.2109
2017-11-27,4.2087
2017-11-28,4.2069
2017-11-29,4.2006
2017-11-30,4.1955
2017-12-01,4.2129
2017-12-04,4.2026
2017-12-05,4.202
2017-12-06,4.2142
2017-12-07,4.2129
2017-12-08,4.202
2017-12-11,4.2033
2017-12-12,4.2044
2017-12-13,4.2146
2017-12-14,4.222
2017-12-15,4.2167
2017-12-18,4.207
2017-12-19,4.1993
2017-12-20,4.2041
2017-12-21,4.1998
2017-12-22,4.1975
2017-12-27,4.1884
2017-12-28,4.1808
2017-12-29,4.177
2018-01-02,4.1633
2018-01-03,4.1652
2018-01-04,4.1555
2018-01-05,4.1554
2018-01-08,4.1663
2018-01-09,4.1795
2018-01-10,4.1814
2018-01-11,4.1785
2018-01-12,4.1722
2018-01-15,4.1686
2018-01-16,4.1713
2018-01-17,4.1713
2018-01-18,4.1665
2018-01-19,4.1716
2018-01-22,4.1707
2018-01-23,4.1696
2018-01-24,4.1563
2018-01-25,4.1469
2018-01-26,4.1422
2018-01-29,4.1436
2018-01-30,4.1449
2018-01-31,4.1503
2018-02-01,4.1543
2018-02-02,4.1541
2018-02-05,4.1585
2018-02-06,4.1644
2018-02-07,4.1595
2018-02-08,4.1695
2018-02-09,4.1903
2018-02-12,4.1748
2018-02-13,4.1781
2018-02-14,4.1719
2018-02-15,4.1532
2018-02-16,4.1597
2018-02-19,4.146
2018-02-20,4.1467
2018-02-21,4.1589
2018-02-22,4.1819
2018-02-23,4.1698
2018-02-26,4.1697
2018-02-27,4.167
2018-02-28,4.1781
2018-03-01,4.1853
2018-03-02,4.1944
2018-03-05,4.1909
2018-03-06,4.1815
2018-03-07,4.1939
2018-03-08,4.2042
2018-03-09,4.2018
2018-03-12,4.2013
2018-03-13,4.2119
2018-03-14,4.2023
2018-03-15,4.2122
2018-03-16,4.2173
2018-03-19,4.2171
2018-03-20,4.2277
2018-03-21,4.2364
2018-03-22,4.2217
2018-03-23,4.2262
2018-03-26,4.23
2018-03-27,4.2196
2018-03-28,4.2072
2018-03-29,4.2106
2018-04-03,4.2087
2018-04-04,4.2035
2018-04-05,4.2041
2018-04-06,4.1983
2018-04-09,4.1952
2018-04-10,4.193
2018-04-11,4.1915
2018-04-12,4.1844
2018-04-13,4.1763
2018-04-16,4.1627
2018-04-17,4.1602
2018-04-18,4.1697
2018-04-19,4.1664
2018-04-20,4.1677
2018-04-23,4.1893
2018-04-24,4.1996
2018-04-25,4.2241
2018-04-26,4.236
2018-04-27,4.2161
2018-04-30,4.2264
2018-05-02,4.2755
2018-05-03,4.2628
2018-05-04,4.2543
2018-05-07,4.2537
2018-05-08,4.2792
2018-05-09,4.2656
2018-05-10,4.2443
2018-05-11,4.2584
2018-05-14,4.2629
2018-05-15,4.2837
2018-05-16,4.2889
2018-05-17,4.2899
2018-05-18,4.295
2018-05-21,4.3138
2018-05-22,4.2862
2018-05-23,4.3088
2018-05-24,4.3031
2018-05-25,4.309
2018-05-28,4.3053
2018-05-29,4.311
2018-05-30,4.3135
2018-05-31,4.3058
2018-06-01,4.3162
2018-06-04,4.2848
2018-06-05,4.2886
2018-06-06,4.2753
2018-06-07,4.2603
2018-06-08,4.2793
2018-06-11,4.2669
2018-06-12,4.2754
2018-06-13,4.2766
2018-06-14,4.2671
2018-06-15,4.2807
2018-06-18,4.2876
2018-06-19,4.3143
2018-06-20,4.3152
2018-06-21,4.3256
2018-06-22,4.3148
2018-06-25,4.337
2018-06-26,4.341
2018-06-27,4.3363
2018-06-28,4.3631
2018-06-29,4.3732
2018-07-02,4.3893
2018-07-03,4.3915
2018-07-04,4.379
2018-07-05,4.3577
2018-07-06,4.3675
2018-07-09,4.3204
2018-07-10,4.3208
2018-07-11,4.3214
2018-07-12,4.3272
2018-07-13,4.321
2018-07-16,4.3013
2018-07-17,4.3007
2018-07-18,4.3103
2018-07-19,4.328
2018-07-20,4.3229
2018-07-23,4.32
2018-07-24,4.3179
2018-07-25,4.2996
2018-07-26,4.2835
2018-07-27,4.2918
2018-07-30,4.2773
2018-07-31,4.277
2018-08-01,4.272
2018-08-02,4.282
2018-08-03,4.2612
2018-08-06,4.2731
2018-08-07,4.2523
2018-08-08,4.2597
2018-08-09,4.2739
2018-08-10,4.2911
2018-08-13,4.2965
2018-08-14,4.2993
2018-08-15,4.3284
2018-08-16,4.3142
2018-08-17,4.3115
2018-08-20,4.3037
2018-08-21,4.3081
2018-08-22,4.2821
2018-08-23,4.2782
2018-08-24,4.2788
2018-08-27,4.271
2018-08-28,4.2723
2018-08-29,4.2838
2018-08-30,4.2882
2018-08-31,4.2913
2018-09-03,4.2952
2018-09-04,4.3046
2018-09-05,4.3249
2018-09-06,4.3183
2018-09-07,4.3081
2018-09-10,4.3169
2018-09-11,4.3094
2018-09-12,4.3099
2018-09-13,4.3098
2018-09-14,4.3049
2018-09-17,4.2955
2018-09-18,4.3008
2018-09-19,4.2951
2018-09-20,4.2925
2018-09-21,4.2946
2018-09-24,4.3009
2018-09-25,4.293
2018-09-26,4.2826
2018-09-27,4.2779
2018-09-28,4.2774
2018-10-01,4.2796
2018-10-02,4.2907
2018-10-03,4.2956
2018-10-04,4.3071
2018-10-05,4.3045
2018-10-08,4.3081
2018-10-09,4.3155
2018-10-10,4.3096
2018-10-11,4.3167
2018-10-12,4.2936
2018-10-15,4.2942
2018-10-16,4.289
2018-10-17,4.294
2018-10-18,4.2989
2018-10-19,4.3017
2018-10-22,4.2905
2018-10-23,4.296
2018-10-24,4.3053
2018-10-25,4.3094
2018-10-26,4.3177
2018-10-29,4.3182
2018-10-30,4.3307
2018-10-31,4.3392
2018-11-01,4.3269
2018-11-02,4.3202
2018-11-05,4.3126
2018-11-06,4.3088
2018-11-07,4.294
2018-11-08,4.2915
2018-11-09,4.288
2018-11-12,4.2968
2018-11-13,4.2992
2018-11-14,4.2911
2018-11-15,4.2933
2018-11-16,4.3154
2018-11-19,4.3293
2018-11-20,4.3177
2018-11-21,4.2995
2018-11-22,4.3004
2018-11-23,4.2949
2018-11-26,4.292
2018-11-27,4.2902
2018-11-28,4.2953
2018-11-29,4.2916
2018-11-30,4.29
2018-12-03,4.2809
2018-12-04,4.2823
2018-12-05,4.2826
2018-12-06,4.2881
2018-12-07,4.2895
2018-12-10,4.2921
2018-12-11,4.2983
2018-12-12,4.2986
2018-12-13,4.2925
2018-12-14,4.2974
2018-12-17,4.2835
2018-12-18,4.2848
2018-12-19,4.2833
2018-12-20,4.2876
2018-12-21,4.2851
2018-12-24,4.2848
2018-12-27,4.2945
2018-12-28,4.3028
2018-12-31,4.3014
2019-01-02,4.2964
2019-01-03,4.2975
2019-01-04,4.2957
2019-01-07,4.2908
2019-01-08,4.3055
2019-01-09,4.2968
2019-01-10,4.2959
2019-01-11,4.2973
2019-01-14,4.2925
2019-01-15,4.293
2019-01-16,4.2877
2019-01-17,4.2835
2019-01-18,4.2931
2019-01-21,4.2911
2019-01-22,4.2827
2019-01-23,4.2887
2019-01-24,4.2881
2019-01-25,4.291
2019-01-28,4.2861
2019-01-29,4.2976
2019-01-30,4.2905
2019-01-31,4.2736
2019-02-01,4.2739
2019-02-04,4.2816
2019-02-05,4.2872
2019-02-06,4.2923
2019-02-07,4.3022
2019-02-08,4.3064
2019-02-11,4.3158
2019-02-12,4.3259
2019-02-13,4.3285
2019-02-14,4.3382
2019-02-15,4.3307
2019-02-18,4.3292
2019-02-19,4.3347
2019-02-20,4.3445
2019-02-21,4.3346
2019-02-22,4.337
2019-02-25,4.336
2019-02-26,4.3282
2019-02-27,4.3146
2019-02-28,4.3089
2019-03-01,4.3096
2019-03-04,4.2989
2019-03-05,4.3016
2019-03-06,4.2996
2019-03-07,4.2991
2019-03-08,4.3002
2019-03-11,4.2994
2019-03-12,4.2994
2019-03-13,4.2993
2019-03-14,4.3032
2019-03-15,4.3049
2019-03-18,4.2979
2019-03-19,4.2879
2019-03-20,4.2834
2019-03-21,4.2807
2019-03-22,4.2913
2019-03-25,4.2953
2019-03-26,4.2934
2019-03-27,4.2936
2019-03-28,4.295
2019-03-29,4.3006
2019-04-01,4.299
2019-04-02,4.2953
2019-04-03,4.293
2019-04-04,4.2892
2019-04-05,4.2897
2019-04-08,4.2887
2019-04-09,4.287
2019-04-10,4.2833
2019-04-11,4.283
2019-04-12,4.2796
2019-04-15,4.2742
2019-04-16,4.2744
2019-04-17,4.2731
2019-04-18,4.2786
2019-04-23,4.2864
2019-04-24,4.2896
2019-04-25,4.295
2019-04-26,4.2878
2019-04-29,4.2944
2019-04-30,4.2868
2019-05-02,4.28
2019-05-03,4.2853
2019-05-06,4.2832
2019-05-07,4.2862
2019-05-08,4.2912
2019-05-09,4.2934
2019-05-10,4.296
2019-05-13,4.3048
2019-05-14,4.3061
2019-05-15,4.3094
2019-05-16,4.2962
2019-05-17,4.3031
2019-05-20,4.2968
2019-05-21,4.306
2019-05-22,4.3038
2019-05-23,4.3064
2019-05-24,4.2974
2019-05-27,4.2935
2019-05-28,4.2951
2019-05-29,4.2998
2019-05-30,4.2887
2019-05-31,4.2843
2019-06-03,4.2833
2019-06-04,4.2802
2019-06-05,4.2778
2019-06-06,4.2788
2019-06-07,4.2714
2019-06-10,4.2639
2019-06-11,4.2678
2019-06-12,4.2608
2019-06-13,4.2574
2019-06-14,4.2534
2019-06-17,4.2605
2019-06-18,4.2592
2019-06-19,4.267
2019-06-20,4.2556
2019-06-21,4.2584
2019-06-24,4.2545
2019-06-25,4.2563
2019-06-26,4.2627
2019-06-27,4.2515
2019-06-28,4.2496
2019-07-01,4.2427
2019-07-02,4.2429
2019-07-03,4.2428
2019-07-04,4.2439
2019-07-05,4.2449
2019-07-08,4.253
2019-07-09,4.263
2019-07-10,4.2738
2019-07-11,4.2681
2019-07-12,4.2675
2019-07-15,4.2622
2019-07-16,4.2555
2019-07-17,4.2641
2019-07-18,4.2596
2019-07-19,4.2571
2019-07-22,4.248
2019-07-23,4.2497
2019-07-24,4.2547
2019-07-25,4.2472
2019-07-26,4.267
2019-07-29,4.2806
2019-07-30,4.2912
2019-07-31,4.2912
2019-08-01,4.2979
2019-08-02,4.3014
2019-08-05,4.3142
2019-08-06,4.3119
2019-08-07,4.3174
2019-08-08,4.3227
2019-08-09,4.3191
2019-08-12,4.3275
2019-08-13,4.3372
2019-08-14,4.372
2019-08-15,4.3878
2019-08-16,4.3381
2019-08-19,4.3702
2019-08-20,4.3552
2019-08-21,4.3529
2019-08-22,4.3643
2019-08-23,4.355
2019-08-26,4.3622
2019-08-27,4.3696
2019-08-28,4.3874
2019-08-29,4.3809
2019-08-30,4.3812
2019-09-02,4.3691
2019-09-03,4.3663
2019-09-04,4.3395
2019-09-05,4.3399
2019-09-06,4.3394
2019-09-09,4.3335
2019-09-10,4.3339
2019-09-11,4.3335
2019-09-12,4.3304
2019-09-13,4.3248
2019-09-16,4.3241
2019-09-17,4.3416
2019-09-18,4.337
2019-09-19,4.3386
2019-09-20,4.3575
2019-09-23,4.382
2019-09-24,4.3813
2019-09-25,4.3915
2019-09-26,4.3871
2019-09-27,4.3865
2019-09-30,4.3782
2019-10-01,4.3774
2019-10-02,4.3698
2019-10-03,4.34
2019-10-04,4.3245
2019-10-07,4.3317
2019-10-08,4.3326
2019-10-09,4.3236
2019-10-10,4.3183
2019-10-11,4.3057
2019-10-14,4.2931
2019-10-15,4.2956
2019-10-16,4.2955
2019-10-17,4.2855
2019-10-18,4.2843
2019-10-21,4.2788
2019-10-22,4.2778
2019-10-23,4.2786
2019-10-24,4.277
2019-10-25,4.2777
2019-10-28,4.2715
2019-10-29,4.2693
2019-10-30,4.2629
2019-10-31,4.2581
2019-11-01,4.2535
2019-11-04,4.2588
2019-11-05,4.2643
2019-11-06,4.2693
2019-11-07,4.2653
2019-11-08,4.261
2019-11-11,4.2737
2019-11-12,4.2747
2019-11-13,4.2874
2019-11-14,4.2926
2019-11-15,4.2785
2019-11-18,4.2915
2019-11-19,4.2877
2019-11-20,4.2932
2019-11-21,4.297
2019-11-22,4.2977
2019-11-25,4.2965
2019-11-26,4.3015
2019-11-27,4.3109
2019-11-28,4.3212
2019-11-29,4.3185
2019-12-02,4.3001
2019-12-03,4.2845
2019-12-04,4.2792
2019-12-05,4.2753
2019-12-06,4.2772
2019-12-09,4.2837
2019-12-10,4.2903
2019-12-11,4.2869
2019-12-12,4.2836
2019-12-13,4.2726
2019-12-16,4.2657
2019-12-17,4.2595
2019-12-18,4.2706
2019-12-19,4.2617
2019-12-20,4.2593
2019-12-23,4.2609
2019-12-24,4.2598
2019-12-27,4.2671
2019-12-30,4.2567
2019-12-31,4.2568
2020-01-02,4.2544
2020-01-03,4.2493
2020-01-06,4.2415
2020-01-07,4.2457
2020-01-08,4.2429
2020-01-09,4.2422
2020-01-10,4.2462
2020-01-13,4.2348
2020-01-14,4.2219
2020-01-15,4.2265
2020-01-16,4.2324
2020-01-17,4.2367
2020-01-20,4.2433
2020-01-21,4.2438
2020-01-22,4.2378
2020-01-23,4.2439
2020-01-24,4.2565
2020-01-27,4.2715
2020-01-28,4.2731
2020-01-29,4.2837
2020-01-30,4.2873
2020-01-31,4.3009
2020-02-03,4.2968
2020-02-04,4.2753
2020-02-05,4.2491
2020-02-06,4.2451
2020-02-07,4.2653
2020-02-10,4.2656
2020-02-11,4.2569
2020-02-12,4.2554
2020-02-13,4.2513
2020-02-14,4.249
2020-02-17,4.262
2020-02-18,4.2746
2020-02-19,4.2712
2020-02-20,4.2825
2020-02-21,4.2835
2020-02-24,4.2989
2020-02-25,4.3012
2020-02-26,4.3094
2020-02-27,4.3124
2020-02-28,4.3259
2020-03-02,4.3276
2020-03-03,4.3166
2020-03-04,4.2968
2020-03-05,4.3029
2020-03-06,4.3042
2020-03-09,4.3132
2020-03-10,4.326
2020-03-11,4.318
2020-03-12,4.3599
2020-03-13,4.357
2020-03-16,4.407
2020-03-17,4.4628
2020-03-18,4.501
2020-03-19,4.5604
2020-03-20,4.5315
2020-03-23,4.6056
2020-03-24,4.6146
2020-03-25,4.5817
2020-03-26,4.5748
2020-03-27,4.5306
2020-03-30,4.5509
2020-03-31,4.5506
2020-04-01,4.5815
2020-04-02,4.5697
2020-04-03,4.5765
2020-04-06,4.5643
2020-04-07,4.5356
2020-04-08,4.5442
2020-04-09,4.5586
2020-04-14,4.5473
2020-04-15,4.5381
2020-04-16,4.5288
2020-04-17,4.5185
2020-04-20,4.5272
2020-04-21,4.5291
2020-04-22,4.5349
2020-04-23,4.5379
2020-04-24,4.5284
2020-04-27,4.5287
2020-04-28,4.5468
2020-04-29,4.5442
2020-04-30,4.5336
2020-05-04,4.5668
2020-05-05,4.5316
2020-05-06,4.5394
2020-05-07,4.5467
2020-05-08,4.5482
2020-05-11,4.5591
2020-05-12,4.5449
2020-05-13,4.5636
2020-05-14,4.5666
2020-05-15,4.565
2020-05-18,4.5596
2020-05-19,4.551
2020-05-20,4.5441
2020-05-21,4.5298
2020-05-22,4.5209
2020-05-25,4.5031
2020-05-26,4.4506
2020-05-27,4.4372
2020-05-28,4.4242
2020-05-29,4.4495
2020-06-01,4.4278
2020-06-02,4.3993
2020-06-03,4.4135
2020-06-04,4.4372
2020-06-05,4.4425
2020-06-08,4.4333
2020-06-09,4.4464
2020-06-10,4.4524
2020-06-11,4.4729
2020-06-12,4.4484
2020-06-15,4.4352
2020-06-16,4.4272
2020-06-17,4.4467
2020-06-18,4.4647
2020-06-19,4.4516
2020-06-22,4.4493
2020-06-23,4.4371
2020-06-24,4.448
2020-06-25,4.4653
2020-06-26,4.4684
2020-06-29,4.4664
2020-06-30,4.456
2020-07-01,4.4583
2020-07-02,4.474
2020-07-03,4.4687
2020-07-06,4.4689
2020-07-07,4.4683
2020-07-08,4.4765
2020-07-09,4.4655
2020-07-10,4.4743
2020-07-13,4.4758
2020-07-14,4.4781
2020-07-15,4.4688
2020-07-16,4.4928
2020-07-17,4.4827
2020-07-20,4.4592
2020-07-21,4.4362
2020-07-22,4.4288
2020-07-23,4.4141
2020-07-24,4.4046
2020-07-27,4.4016
2020-07-28,4.4054
2020-07-29,4.4194
2020-07-30,4.408
2020-07-31,4.4034
2020-08-03,4.4201
2020-08-04,4.4055
2020-08-05,4.3935
2020-08-06,4.4073
2020-08-07,4.4081
2020-08-10,4.4021
2020-08-11,4.4018
2020-08-12,4.4033
2020-08-13,4.3968
2020-08-14,4.3984
2020-08-17,4.3966
2020-08-18,4.387
2020-08-19,4.3942
2020-08-20,4.3914
2020-08-21,4.3979
2020-08-24,4.3986
2020-08-25,4.3919
2020-08-26,4.4074
2020-08-27,4.3993
2020-08-28,4.3921
2020-08-31,4.3971
2020-09-01,4.3925
2020-09-02,4.4186
2020-09-03,4.4269
2020-09-04,4.4514
2020-09-07,4.4504
2020-09-08,4.4483
2020-09-09,4.4498
2020-09-10,4.4525
2020-09-11,4.4467
2020-09-14,4.4504
2020-09-15,4.4461
2020-09-16,4.4466
2020-09-17,4.4579
2020-09-18,4.4602
2020-09-21,4.4837
2020-09-22,4.4934
2020-09-23,4.5001
2020-09-24,4.5293
2020-09-25,4.5557
2020-09-28,4.5502
2020-09-29,4.5435
2020-09-30,4.5462
2020-10-01,4.4935
2020-10-02,4.4944
2020-10-05,4.498
2020-10-06,4.4872
2020-10-07,4.4862
2020-10-08,4.4843
2020-10-09,4.4702
2020-10-12,4.481
2020-10-13,4.4851
2020-10-14,4.5073
2020-10-15,4.5518
2020-10-16,4.5558
2020-10-19,4.5723
2020-10-20,4.5809
2020-10-21,4.574
2020-10-22,4.5783
2020-10-23,4.5823
2020-10-26,4.5799
2020-10-27,4.5842
2020-10-28,4.6202
2020-10-29,4.6225
2020-10-30,4.6222
2020-11-02,4.6018
2020-11-03,4.568
2020-11-04,4.5368
2020-11-05,4.5134
2020-11-06,4.5263
2020-11-09,4.467
2020-11-10,4.499
2020-11-11,4.4873
2020-11-12,4.4886
2020-11-13,4.4888
2020-11-16,4.4692
2020-11-17,4.4964
2020-11-18,4.4694
2020-11-19,4.4714
2020-11-20,4.4639
2020-11-23,4.4735
2020-11-24,4.4666
2020-11-25,4.4679
2020-11-26,4.4749
2020-11-27,4.4907
2020-11-30,4.471
2020-12-01,4.4788
2020-12-02,4.4783
2020-12-03,4.4733
2020-12-04,4.4769
2020-12-07,4.467
2020-12-08,4.4663
2020-12-09,4.4246
2020-12-10,4.4268
2020-12-11,4.4358
2020-12-14,4.4373
2020-12-15,4.4426
2020-12-16,4.4354
2020-12-17,4.4423
2020-12-18,4.4779
2020-12-21,4.5111
2020-12-22,4.5116
2020-12-23,4.5025
2020-12-24,4.5022
2020-12-28,4.4907
2020-12-29,4.5308
2020-12-30,4.5565
2020-12-31,4.5597
2021-01-04,4.5475
2021-01-05,4.5473
2021-01-06,4.516
2021-01-07,4.4998
2021-01-08,4.5113
2021-01-11,4.5294
2021-01-12,4.5248
2021-01-13,4.5253
2021-01-14,4.5379
2021-01-15,4.5375
2021-01-18,4.5389
2021-01-19,4.5352
2021-01-20,4.5322
2021-01-21,4.5284
2021-01-22,4.5385
2021-01-25,4.5409
2021-01-26,4.5465
2021-01-27,4.552
2021-01-28,4.5471
2021-01-29,4.5304
2021-02-01,4.508
2021-02-02,4.4953
2021-02-03,4.4917
2021-02-04,4.4941
2021-02-05,4.5023
2021-02-08,4.4824
2021-02-09,4.4761
2021-02-10,4.4819
2021-02-11,4.4975
2021-02-12,4.4999
2021-02-15,4.4864
2021-02-16,4.4942
2021-02-17,4.5012
2021-02-18,4.4888
2021-02-19,4.483
2021-02-22,4.4982
2021-02-23,4.5066
2021-02-24,4.5178
2021-02-25,4.5122
2021-02-26,4.5186
2021-03-01,4.5263
2021-03-02,4.5322
2021-03-03,4.5336
2021-03-04,4.5529
2021-03-05,4.5748
2021-03-08,4.6002
2021-03-09,4.5843
2021-03-10,4.5752
2021-03-11,4.576
2021-03-12,4.5867
2021-03-15,4.5914
2021-03-16,4.5933
2021-03-17,4.6136
2021-03-18,4.6253
2021-03-19,4.62
2021-03-22,4.6017
2021-03-23,4.6191
2021-03-24,4.628
2021-03-25,4.6399
2021-03-26,4.6493
2021-03-29,4.6457
2021-03-30,4.6582
2021-03-31,4.6508
2021-04-01,4.6089
2021-04-06,4.5914
2021-04-07,4.5756
2021-04-08,4.5513
2021-04-09,4.5392
2021-04-12,4.5246
2021-04-13,4.5668
2021-04-14,4.5537
2021-04-15,4.5554
2021-04-16,4.5509
2021-04-19,4.5473
2021-04-20,4.5525
2021-04-21,4.5561
2021-04-22,4.5567
2021-04-23,4.5623
2021-04-26,4.5571
2021-04-27,4.5644
2021-04-28,4.5859
2021-04-29,4.5654
2021-04-30,4.5635
2021-05-03,4.5518
2021-05-04,4.5584
2021-05-05,4.5778
2021-05-06,4.5868
2021-05-07,4.5754
2021-05-10,4.5631
2021-05-11,4.5522
2021-05-12,4.5438
2021-05-13,4.5467
2021-05-14,4.5218
2021-05-17,4.5428
2021-05-18,4.5243
2021-05-19,4.5188
2021-05-20,4.5153
2021-05-21,4.4892
2021-05-24,4.485
2021-05-25,4.4839
2021-05-26,4.4971
2021-05-27,4.4837
2021-05-28,4.4865
2021-05-31,4.4856
2021-06-01,4.4661
2021-06-02,4.4653
2021-06-03,4.452
2021-06-04,4.4703
2021-06-07,4.4662
2021-06-08,4.4698
2021-06-09,4.456
2021-06-10,4.4818
2021-06-11,4.4695
2021-06-14,4.5152
2021-06-15,4.5235
2021-06-16,4.5166
2021-06-17,4.5375
2021-06-18,4.5492
2021-06-21,4.5329
2021-06-22,4.5328
2021-06-23,4.5227
2021-06-24,4.5245
2021-06-25,4.5132
2021-06-28,4.507
2021-06-29,4.5193
2021-06-30,4.5201
2021-07-01,4.5164
2021-07-02,4.5264
2021-07-05,4.5059
2021-07-06,4.4941
2021-07-07,4.5192
2021-07-08,4.5504
2021-07-09,4.5497
2021-07-12,4.5542
2021-07-13,4.5687
2021-07-14,4.5736
2021-07-15,4.5748
2021-07-16,4.5867
2021-07-19,4.5885
2021-07-20,4.5954
2021-07-21,4.5987
2021-07-22,4.5661
2021-07-23,4.5691
2021-07-26,4.5888
2021-07-27,4.5906
2021-07-28,4.5965
2021-07-29,4.5782
2021-07-30,4.5636
2021-08-02,4.5562
2021-08-03,4.5562
2021-08-04,4.5438
2021-08-05,4.548
2021-08-06,4.5563
2021-08-09,4.5645
2021-08-10,4.5755
2021-08-11,4.5858
2021-08-12,4.5876
2021-08-13,4.574
2021-08-16,4.5653
2021-08-17,4.5605
2021-08-18,4.5634
2021-08-19,4.5832
2021-08-20,4.5895
2021-08-23,4.5868
2021-08-24,4.5792
2021-08-25,4.5779
2021-08-26,4.5779
2021-08-27,4.5827
2021-08-30,4.5678
2021-08-31,4.5296
2021-09-01,4.5078
2021-09-02,4.5069
2021-09-03,4.5088
2021-09-06,4.5208
2021-09-07,4.518
2021-09-08,4.5178
2021-09-09,4.5335
2021-09-10,4.5417
2021-09-13,4.5518
2021-09-14,4.5542
2021-09-15,4.5501
2021-09-16,4.5763
2021-09-17,4.5804
2021-09-20,4.5997
2021-09-21,4.6261
2021-09-22,4.6325
2021-09-23,4.6077
2021-09-24,4.6047
2021-09-27,4.5955
2021-09-28,4.6176
2021-09-29,4.626
2021-09-30,4.6197
2021-10-01,4.5826
2021-10-04,4.5685
2021-10-05,4.5998
2021-10-06,4.5925
2021-10-07,4.5458
2021-10-08,4.6182
2021-10-11,4.5913
2021-10-12,4.5805
2021-10-13,4.5772
2021-10-14,4.5693
2021-10-15,4.5644
2021-10-18,4.5787
2021-10-19,4.5729
2021-10-20,4.585
2021-10-21,4.603
2021-10-22,4.5975
2021-10-25,4.6187
2021-10-26,4.6009
2021-10-27,4.6207
2021-10-28,4.6192
2021-10-29,4.6215
2021-11-01,4.6225
2021-11-02,4.6078
2021-11-03,4.5875
2021-11-04,4.6067
2021-11-05,4.6037
2021-11-08,4.5956
2021-11-09,4.5926
2021-11-10,4.6121
2021-11-11,4.6296
2021-11-12,4.6428
2021-11-15,4.6374
2021-11-16,4.6545
2021-11-17,4.6571
2021-11-18,4.6565
2021-11-19,4.6818
2021-11-22,4.6984
2021-11-23,4.7119
2021-11-24,4.6802
2021-11-25,4.6661
2021-11-26,4.7117
2021-11-29,4.6962
2021-11-30,4.6639
2021-12-01,4.6283
2021-12-02,4.5953
2021-12-03,4.5923
2021-12-06,4.5924
2021-12-07,4.5942
2021-12-08,4.5962
2021-12-09,4.6045
2021-12-10,4.6123
2021-12-13,4.6221
2021-12-14,4.63
2021-12-15,4.6272
2021-12-16,4.6278
2021-12-17,4.6336
2021-12-20,4.6337
2021-12-21,4.6375
2021-12-22,4.632
2021-12-23,4.628
2021-12-24,4.619
2021-12-27,4.605
2021-12-28,4.6063
2021-12-29,4.6037
2021-12-30,4.596
2021-12-31,4.5969
2022-01-03,4.5895
2022-01-04,4.5667
2022-01-05,4.5666
2022-01-06,4.5614
2022-01-07,4.5496
2022-01-10,4.5334
2022-01-11,4.5438
2022-01-12,4.5359
2022-01-13,4.5361
2022-01-14,4.5414
2022-01-17,4.5256
2022-01-18,4.526
2022-01-19,4.5229
2022-01-20,4.5228
2022-01-21,4.5318
2022-01-24,4.5572
2022-01-25,4.5751
2022-01-26,4.5864
2022-01-27,4.5592
2022-01-28,4.5755
2022-01-31,4.5892
2022-02-01,4.5804
2022-02-02,4.5449
2022-02-03,4.5315
2022-02-04,4.5474
2022-02-07,4.5432
2022-02-08,4.5312
2022-02-09,4.5135
2022-02-10,4.4921
2022-02-11,4.5204
2022-02-14,4.54
2022-02-15,4.5036
2022-02-16,4.4961
2022-02-17,4.5065
2022-02-18,4.5201
2022-02-21,4.5351
2022-02-22,4.5447
2022-02-23,4.5481
2022-02-24,4.6554
2022-02-25,4.6369
2022-02-28,4.6835
2022-03-01,4.7947
2022-03-02,4.8021
2022-03-03,4.7691
2022-03-04,4.853
2022-03-07,4.9525
2022-03-08,4.9103
2022-03-09,4.8196
2022-03-10,4.8239
2022-03-11,4.782
2022-03-14,4.7218
2022-03-15,4.7355
2022-03-16,4.6765
2022-03-17,4.6889
2022-03-18,4.7135
2022-03-21,4.696
2022-03-22,4.6851
2022-03-23,4.7052
2022-03-24,4.7421
2022-03-25,4.7307
2022-03-28,4.718
2022-03-29,4.6594
2022-03-30,4.6679
2022-03-31,4.6531
2022-04-01,4.6401
2022-04-04,4.6375
2022-04-05,4.6265
2022-04-06,4.6328
2022-04-07,4.637
2022-04-08,4.6437
2022-04-11,4.6456
2022-04-12,4.6552
2022-04-13,4.6453
2022-04-14,4.6478
2022-04-19,4.6553
2022-04-20,4.6338
2022-04-21,4.63
2022-04-22,4.6336
2022-04-25,4.6398
2022-04-26,4.6466
2022-04-27,4.7043
2022-04-28,4.6891
2022-04-29,4.678
2022-05-02,4.685
2022-05-03,4.6925
2022-05-04,4.6875
2022-05-05,4.6673
2022-05-06,4.7028
2022-05-09,4.6985
2022-05-10,4.6763
2022-05-11,4.6575
2022-05-12,4.668
2022-05-13,4.6883
2022-05-16,4.6675
2022-05-17,4.6488
2022-05-18,4.6443
2022-05-19,4.6423
2022-05-20,4.6365
2022-05-23,4.621
2022-05-24,4.6015
2022-05-25,4.621
2022-05-26,4.6083
2022-05-27,4.5858
2022-05-30,4.5855
2022-05-31,4.5805
2022-06-01,4.5913
2022-06-02,4.5787
2022-06-03,4.5955
2022-06-06,4.5808
2022-06-07,4.5813
2022-06-08,4.5698
2022-06-09,4.5925
2022-06-10,4.6053
2022-06-13,4.6373
2022-06-14,4.6563
2022-06-15,4.669
2022-06-16,4.7138
2022-06-17,4.7003
2022-06-20,4.652
2022-06-21,4.6435
2022-06-22,4.6905
2022-06-23,4.7085
2022-06-24,4.7023
2022-06-27,4.699
2022-06-28,4.6905
2022-06-29,4.6869
2022-06-30,4.6904
2022-07-01,4.7168
2022-07-04,4.71
2022-07-05,4.7448
2022-07-06,4.771
2022-07-07,4.7721
2022-07-08,4.763
2022-07-11,4.7968
2022-07-12,4.819
2022-07-13,4.824
2022-07-14,4.8146
2022-07-15,4.7953
2022-07-18,4.776
2022-07-19,4.7598
2022-07-20,4.782
2022-07-21,4.761
2022-07-22,4.7508
2022-07-25,4.708
2022-07-26,4.742
2022-07-27,4.7978
2022-07-28,4.7908
2022-07-29,4.7375
2022-08-01,4.734
2022-08-02,4.7063
2022-08-03,4.691
2022-08-04,4.7233
2022-08-05,4.7085
2022-08-08,4.7043
2022-08-09,4.7085
2022-08-10,4.7063
2022-08-11,4.6828
2022-08-12,4.6773
2022-08-15,4.6858
2022-08-16,4.7043
2022-08-17,4.7078
2022-08-18,4.724
2022-08-19,4.751
2022-08-22,4.764
2022-08-23,4.7788
2022-08-24,4.7668
2022-08-25,4.7578
2022-08-26,4.7485
2022-08-29,4.745
2022-08-30,4.7323
2022-08-31,4.7283
2022-09-01,4.7128
2022-09-02,4.7063
2022-09-05,4.736
2022-09-06,4.7068
2022-09-07,4.729
2022-09-08,4.7155
2022-09-09,4.721
2022-09-12,4.6965
2022-09-13,4.705
2022-09-14,4.7163
2022-09-15,4.7273
2022-09-16,4.7143
2022-09-19,4.7058
2022-09-20,4.7208
2022-09-21,4.7505
2022-09-22,4.7592
2022-09-23,4.7543
2022-09-26,4.7608
2022-09-27,4.764
2022-09-28,4.8043
2022-09-29,4.857
2022-09-30,4.8483
2022-10-03,4.832
2022-10-04,4.8193
2022-10-05,4.791
2022-10-06,4.8505
2022-10-07,4.8595
2022-10-10,4.8655
2022-10-11,4.869
2022-10-12,4.8495
2022-10-13,4.8303
2022-10-14,4.8328
2022-10-17,4.8143
2022-10-18,4.804
2022-10-19,4.7878
2022-10-20,4.7728
2022-10-21,4.7885
2022-10-24,4.7908
2022-10-25,4.777
2022-10-26,4.7548
2022-10-27,4.7585
2022-10-28,4.7275
2022-10-31,4.7085
2022-11-01,4.7053
2022-11-02,4.7035
2022-11-03,4.709
2022-11-04,4.6825
2022-11-07,4.6865
2022-11-08,4.6918
2022-11-09,4.701
2022-11-10,4.706
2022-11-11,4.6765
2022-11-14,4.6898
2022-11-15,4.7073
2022-11-16,4.7065
2022-11-17,4.7153
2022-11-18,4.7033
2022-11-21,4.7075
2022-11-22,4.7125
2022-11-23,4.7033
2022-11-24,4.6958
2022-11-25,4.6875
2022-11-28,4.6938
2022-11-29,4.673
2022-11-30,4.6635
2022-12-01,4.6998
2022-12-02,4.684
2022-12-05,4.695
2022-12-06,4.6975
2022-12-07,4.7003
2022-12-08,4.6853
2022-12-09,4.6869
2022-12-12,4.6923
2022-12-13,4.6938
2022-12-14,4.681
2022-12-15,4.689
2022-12-16,4.6925
2022-12-19,4.6853
2022-12-20,4.6757
2022-12-21,4.6665
2022-12-22,4.6443
2022-12-23,4.6423
2022-12-27,4.6683
2022-12-28,4.7008
2022-12-29,4.6855
2022-12-30,4.6808
2023-01-02,4.6813
2023-01-03,4.6831
2023-01-04,4.6685
2023-01-05,4.67
2023-01-06,4.695
2023-01-09,4.6963
2023-01-10,4.695
2023-01-11,4.6819
2023-01-12,4.692
2023-01-13,4.6888
2023-01-16,4.6935
2023-01-17,4.6958
2023-01-18,4.6983
2023-01-19,4.7063
2023-01-20,4.71
2023-01-23,4.7113
2023-01-24,4.713
2023-01-25,4.7158
2023-01-26,4.7195
2023-01-27,4.7085
2023-01-30,4.7103
2023-01-31,4.709
2023-02-01,4.7075
2023-02-02,4.7015
2023-02-03,4.692
2023-02-06,4.7228
2023-02-07,4.7565
2023-02-08,4.7385
2023-02-09,4.7388
2023-02-10,4.781
2023-02-13,4.7875
2023-02-14,4.7623
2023-02-15,4.76
2023-02-16,4.7765
2023-02-17,4.7773
2023-02-20,4.7435
2023-02-21,4.7478
2023-02-22,4.7443
2023-02-23,4.7438
2023-02-24,4.7205
2023-02-27,4.716
2023-02-28,4.7125
2023-03-01,4.6745
2023-03-02,4.689
2023-03-03,4.708
2023-03-06,4.709
2023-03-07,4.6898
2023-03-08,4.6753
2023-03-09,4.6785
2023-03-10,4.6815
2023-03-13,4.6944
2023-03-14,4.6878
2023-03-15,4.7
2023-03-16,4.6963
2023-03-17,4.7045
2023-03-20,4.7045
2023-03-21,4.6998
2023-03-22,4.6835
2023-03-23,4.6868
2023-03-24,4.6865
2023-03-27,4.6855
2023-03-28,4.6815
2023-03-29,4.687
2023-03-30,4.6813
2023-03-31,4.67
2023-04-03,4.6765
2023-04-04,4.6718
2023-04-05,4.6818
2023-04-06,4.6863
2023-04-11,4.669
2023-04-12,4.6631
2023-04-13,4.632
2023-04-14,4.6435
2023-04-17,4.6288
2023-04-18,4.618
2023-04-19,4.6238
2023-04-20,4.611
2023-04-21,4.6108
2023-04-24,4.6055
2023-04-25,4.59
2023-04-26,4.5928
2023-04-27,4.59
2023-04-28,4.5815
2023-05-02,4.5758
2023-05-03,4.582
2023-05-04,4.5905
2023-05-05,4.577
2023-05-08,4.5693
2023-05-09,4.5723
2023-05-10,4.5363
2023-05-11,4.5238
2023-05-12,4.5343
2023-05-15,4.5078
2023-05-16,4.4865
2023-05-17,4.507
2023-05-18,4.543
2023-05-19,4.5205
2023-05-22,4.5133
2023-05-23,4.5018
2023-05-24,4.49
2023-05-25,4.511
2023-05-26,4.5354
2023-05-29,4.52
2023-05-30,4.5235
2023-05-31,4.5368
2023-06-01,4.541
2023-06-02,4.4975
2023-06-05,4.4735
2023-06-06,4.4915
2023-06-07,4.4695
2023-06-08,4.4788
2023-06-09,4.4605
2023-06-12,4.4395
2023-06-13,4.475
2023-06-14,4.4553
2023-06-15,4.4693
2023-06-16,4.4728
2023-06-19,4.457
2023-06-20,4.4463
2023-06-21,4.4328
2023-06-22,4.4253
2023-06-23,4.4238
2023-06-26,4.4363
2023-06-27,4.4283
2023-06-28,4.4763
2023-06-29,4.4485
2023-06-30,4.4388
2023-07-03,4.4385
2023-07-04,4.4215
2023-07-05,4.4558
2023-07-06,4.4753
2023-07-07,4.4838
2023-07-10,4.4508
2023-07-11,4.4563
2023-07-12,4.4473
2023-07-13,4.4323
2023-07-14,4.449
2023-07-17,4.4508
2023-07-18,4.4458
2023-07-19,4.4508
2023-07-20,4.4513
2023-07-21,4.4583
2023-07-24,4.446
2023-07-25,4.4263
2023-07-26,4.42
2023-07-27,4.4265
2023-07-28,4.4113
2023-07-31,4.407
2023-08-01,4.4303
2023-08-02,4.4475
2023-08-03,4.4598
2023-08-04,4.4485
2023-08-07,4.4138
2023-08-08,4.45
2023-08-09,4.467
2023-08-10,4.4548
2023-08-11,4.4368
2023-08-14,4.4395
2023-08-15,4.4888
2023-08-16,4.4528
2023-08-17,4.4718
2023-08-18,4.4698
2023-08-21,4.4785
2023-08-22,4.4643
2023-08-23,4.4773
2023-08-24,4.4768
2023-08-25,4.469
2023-08-28,4.4735
2023-08-29,4.472
2023-08-30,4.4728
2023-08-31,4.4673
2023-09-01,4.4685
2023-09-04,4.4663
2023-09-05,4.489
2023-09-06,4.501
2023-09-07,4.5883
2023-09-08,4.6213
2023-09-11,4.6438
2023-09-12,4.66
2023-09-13,4.6243
2023-09-14,4.6275
2023-09-15,4.6308
2023-09-18,4.6383
2023-09-19,4.6548
2023-09-20,4.6403
2023-09-21,4.627
2023-09-22,4.5925
2023-09-25,4.5893
2023-09-26,4.6093
2023-09-27,4.625
2023-09-28,4.6343
2023-09-29,4.6283
2023-10-02,4.6123
2023-10-03,4.623
2023-10-04,4.6278
2023-10-05,4.5988
2023-10-06,4.5983
2023-10-09,4.5798
2023-10-10,4.5565
2023-10-11,4.5158
2023-10-12,4.5291
2023-10-13,4.535
2023-10-16,4.4733
2023-10-17,4.4273
2023-10-18,4.441
2023-10-19,4.45
2023-10-20,4.4653
2023-10-23,4.459
2023-10-24,4.463
2023-10-25,4.4825
2023-10-26,4.4673
2023-10-27,4.4715
2023-10-30,4.4565
2023-10-31,4.4405
2023-11-01,4.4658
2023-11-02,4.4495
2023-11-03,4.45
2023-11-06,4.467
2023-11-07,4.4618
2023-11-08,4.4655
2023-11-09,4.449
2023-11-10,4.4243
2023-11-13,4.428
2023-11-14,4.4211
2023-11-15,4.397
2023-11-16,4.373
2023-11-17,4.377
2023-11-20,4.369
2023-11-21,4.3625
2023-11-22,4.3625
2023-11-23,4.3655
2023-11-24,4.3638
2023-11-27,4.359
2023-11-28,4.3473
2023-11-29,4.3355
2023-11-30,4.349
2023-12-01,4.343
2023-12-04,4.3393
2023-12-05,4.3225
2023-12-06,4.3403
2023-12-07,4.3313
2023-12-08,4.3318
2023-12-11,4.3303
2023-12-12,4.3393
2023-12-13,4.326
2023-12-14,4.3123
2023-12-15,4.309
2023-12-18,4.345
2023-12-19,4.3373
2023-12-20,4.3413
2023-12-21,4.3403
2023-12-22,4.342
2023-12-27,4.3308
2023-12-28,4.3355
2023-12-29,4.3395
2024-01-02,4.3708
2024-01-03,4.3638
2024-01-04,4.346
2024-01-05,4.3568
2024-01-08,4.3465
2024-01-09,4.3448
2024-01-10,4.341
2024-01-11,4.349
2024-01-12,4.3628
2024-01-15,4.3625
2024-01-16,4.387
2024-01-17,4.3993
2024-01-18,4.399
2024-01-19,4.3618
2024-01-22,4.364
2024-01-23,4.383
2024-01-24,4.3725
2024-01-25,4.3765
2024-01-26,4.3775
2024-01-29,4.366
2024-01-30,4.3618
2024-01-31,4.333
2024-02-01,4.3373
2024-02-02,4.3175
2024-02-05,4.3343
2024-02-06,4.3463
2024-02-07,4.3453
2024-02-08,4.3375
2024-02-09,4.3206
2024-02-12,4.3198
2024-02-13,4.3198
2024-02-14,4.3415
2024-02-15,4.344
2024-02-16,4.34
2024-02-19,4.3265
2024-02-20,4.3248
2024-02-21,4.3165
2024-02-22,4.3158
2024-02-23,4.3118
2024-02-26,4.3053
2024-02-27,4.3153
2024-02-28,4.2965
2024-02-29,4.3208
2024-03-01,4.318
2024-03-04,4.322
2024-03-05,4.3248
2024-03-06,4.3018
2024-03-07,4.302
2024-03-08,4.3065
2024-03-11,4.2805
2024-03-12,4.2865
2024-03-13,4.286
2024-03-14,4.2935
2024-03-15,4.2953
2024-03-18,4.3113
2024-03-19,4.321
2024-03-20,4.3248
2024-03-21,4.3028
2024-03-22,4.317
2024-03-25,4.3078
2024-03-26,4.3093
2024-03-27,4.3148
2024-03-28,4.3123
2024-04-02,4.2938
2024-04-03,4.2968
2024-04-04,4.2955
2024-04-05,4.2835
2024-04-08,4.2773
2024-04-09,4.2565
2024-04-10,4.2563
2024-04-11,4.257
2024-04-12,4.2603
2024-04-15,4.2938
2024-04-16,4.3435
2024-04-17,4.3508
2024-04-18,4.3255
2024-04-19,4.33
2024-04-22,4.3188
2024-04-23,4.3158
2024-04-24,4.3193
2024-04-25,4.3165
2024-04-26,4.3205
2024-04-29,4.3255
2024-04-30,4.3178
2024-05-02,4.3278
2024-05-03,4.329
2024-05-06,4.3185
2024-05-07,4.3138
2024-05-08,4.3075
2024-05-09,4.292
2024-05-10,4.2968
2024-05-13,4.286
2024-05-14,4.2695
2024-05-15,4.269
2024-05-16,4.2648
2024-05-17,4.265
2024-05-20,4.2533
2024-05-21,4.2553
2024-05-22,4.2686
2024-05-23,4.263
2024-05-24,4.2553
2024-05-27,4.2553
2024-05-28,4.2483
2024-05-29,4.264
2024-05-30,4.2848
2024-05-31,4.2645
2024-06-03,4.2815
2024-06-04,4.3005
2024-06-05,4.3045
2024-06-06,4.3015
2024-06-07,4.288
2024-06-10,4.324
2024-06-11,4.3413
2024-06-12,4.3385
2024-06-13,4.3473
2024-06-14,4.3733
2024-06-17,4.36
2024-06-18,4.3503
2024-06-19,4.33
2024-06-20,4.321
2024-06-21,4.322
2024-06-24,4.3003
2024-06-25,4.2995
2024-06-26,4.3075
2024-06-27,4.3188
2024-06-28,4.309
2024-07-01,4.3008
2024-07-02,4.325
2024-07-03,4.3018
2024-07-04,4.2918
2024-07-05,4.2798
2024-07-08,4.2778
2024-07-09,4.2643
2024-07-10,4.2555
2024-07-11,4.2545
2024-07-12,4.2548
2024-07-15,4.2495
2024-07-16,4.2683
2024-07-17,4.2948
2024-07-18,4.2918
2024-07-19,4.2853
2024-07-22,4.2788
2024-07-23,4.277
2024-07-24,4.289
2024-07-25,4.2968
2024-07-26,4.274
2024-07-29,4.2938
2024-07-30,4.2838
2024-07-31,4.2908
2024-08-01,4.2958
2024-08-02,4.289
2024-08-05,4.3128
2024-08-06,4.3105
2024-08-07,4.3173
2024-08-08,4.319
2024-08-09,4.3253
2024-08-12,4.3058
2024-08-13,4.289
2024-08-14,4.2975
2024-08-15,4.2815
2024-08-16,4.2703
2024-08-19,4.2688
2024-08-20,4.2643
2024-08-21,4.2858
2024-08-22,4.2815
2024-08-23,4.2775
2024-08-26,4.2815
2024-08-27,4.2783
2024-08-28,4.3003
2024-08-29,4.2898
2024-08-30,4.2763
2024-09-02,4.275
2024-09-03,4.2783
2024-09-04,4.2788
2024-09-05,4.2683
2024-09-06,4.28
2024-09-09,4.278
2024-09-10,4.2833
2024-09-11,4.2873
2024-09-12,4.2978
2024-09-13,4.2898
2024-09-16,4.2708
2024-09-17,4.2683
2024-09-18,4.27
2024-09-19,4.2658
2024-09-20,4.2763
2024-09-23,4.275
2024-09-24,4.2665
2024-09-25,4.2643
2024-09-26,4.271
2024-09-27,4.2738
2024-09-30,4.2788
2024-10-01,4.2853
2024-10-02,4.295
2024-10-03,4.3058
2024-10-04,4.3145
2024-10-07,4.3215
2024-10-08,4.3083
2024-10-09,4.2978
2024-10-10,4.3048
2024-10-11,4.2898
2024-10-14,4.2893
2024-10-15,4.2938
2024-10-16,4.3
2024-10-17,4.3113
2024-10-18,4.3085
2024-10-21,4.3175
2024-10-22,4.317
2024-10-23,4.3433
2024-10-24,4.3435
2024-10-25,4.3478
2024-10-28,4.35
2024-10-29,4.348
2024-10-30,4.3463
2024-10-31,4.3508
2024-11-01,4.3543
2024-11-04,4.3423
2024-11-05,4.358
2024-11-06,4.3493
2024-11-07,4.3253
2024-11-08,4.3255
2024-11-11,4.344
2024-11-12,4.3503
2024-11-13,4.3425
2024-11-14,4.3383
2024-11-15,4.32
2024-11-18,4.3278
2024-11-19,4.3303
2024-11-20,4.3308
2024-11-21,4.3418
2024-11-22,4.3373
2024-11-25,4.3185
2024-11-26,4.3105
2024-11-27,4.312
2024-11-28,4.31
2024-11-29,4.296
2024-12-02,4.2905
2024-12-03,4.2973
2024-12-04,4.2868
2024-12-05,4.2745
2024-12-06,4.267
2024-12-09,4.2593
2024-12-10,4.2678
2024-12-11,4.2643
2024-12-12,4.2758
2024-12-13,4.2643
2024-12-16,4.264
2024-12-17,4.2635
2024-12-18,4.2578
2024-12-19,4.25
2024-12-20,4.2623
2024-12-23,4.2753
2024-12-24,4.2715
2024-12-27,4.2753
2024-12-30,4.2655
2024-12-31,4.275
2025-01-02,4.2753
2025-01-03,4.2725
2025-01-06,4.2518
2025-01-07,4.2583
2025-01-08,4.2765
2025-01-09,4.2723
2025-01-10,4.266
2025-01-13,4.2715
2025-01-14,4.2683
2025-01-15,4.2613
2025-01-16,4.2688
2025-01-17,4.2585
2025-01-20,4.2568
2025-01-21,4.2538
2025-01-22,4.23
2025-01-23,4.211
2025-01-24,4.2138
2025-01-27,4.2193
2025-01-28,4.2078
2025-01-29,4.2135
2025-01-30,4.2063
2025-01-31,4.213
2025-02-03,4.2255
2025-02-04,4.2193
2025-02-05,4.1968
2025-02-06,4.2065
2025-02-07,4.1985
2025-02-10,4.1758
2025-02-11,4.1785
2025-02-12,4.1713
2025-02-13,4.1688
2025-02-14,4.1653
2025-02-17,4.1605
2025-02-18,4.166
2025-02-19,4.174
2025-02-20,4.1605
2025-02-21,4.1683
2025-02-24,4.1445
2025-02-25,4.1393
2025-02-26,4.144
2025-02-27,4.1308
2025-02-28,4.1503
2025-03-03,4.1708
2025-03-04,4.1593
2025-03-05,4.15
2025-03-06,4.177
2025-03-07,4.1788
2025-03-10,4.196
2025-03-11,4.2003
2025-03-12,4.1955
2025-03-13,4.1965
2025-03-14,4.1723
2025-03-17,4.1863
2025-03-18,4.175
2025-03-19,4.1885
2025-03-20,4.2003
2025-03-21,4.2023
2025-03-24,4.176
2025-03-25,4.1635
2025-03-26,4.1755
2025-03-27,4.1968
2025-03-28,4.1775
2025-03-31,4.184
2025-04-01,4.1863
2025-04-02,4.1775
2025-04-03,4.196
2025-04-04,4.2625
2025-04-07,4.294
2025-04-08,4.269
2025-04-09,4.2953
2025-04-10,4.2543
2025-04-11,4.2953
2025-04-14,4.286
2025-04-15,4.2844
2025-04-16,4.2933
2025-04-17,4.2743
2025-04-22,4.28
2025-04-23,4.2915
2025-04-24,4.2748
2025-04-25,4.2683
2025-04-28,4.2755
2025-04-29,4.2708
2025-04-30,4.2753
2025-05-02,4.275
2025-05-05,4.2733
2025-05-06,4.2748
2025-05-07,4.2753
2025-05-08,4.27
2025-05-09,4.2393
2025-05-12,4.2403
2025-05-13,4.2518
2025-05-14,4.2425
2025-05-15,4.2423
2025-05-16,4.26
2025-05-19,4.2688
2025-05-20,4.2488
2025-05-21,4.242
2025-05-22,4.246
2025-05-23,4.2603
2025-05-26,4.2445
2025-05-27,4.2475
2025-05-28,4.2415
2025-05-29,4.236
2025-05-30,4.2498
2025-06-02,4.2578
2025-06-03,4.2798
2025-06-04,4.283
2025-06-05,4.284
2025-06-06,4.287
2025-06-09,4.27
2025-06-10,4.2533
2025-06-11,4.2605
2025-06-12,4.269
2025-06-13,4.2745
2025-06-16,4.2643
2025-06-17,4.2778
2025-06-18,4.2733
2025-06-19,4.2773
2025-06-20,4.2656
2025-06-23,4.2755
2025-06-24,4.2538
2025-06-25,4.2478
2025-06-26,4.2468
2025-06-27,4.2378
2025-06-30,4.2423
2025-07-01,4.2443
2025-07-02,4.2493
2025-07-03,4.2485
2025-07-04,4.2458
2025-07-07,4.2493
2025-07-08,4.2438
2025-07-09,4.2395
2025-07-10,4.2475
2025-07-11,4.2575
2025-07-14,4.2585
2025-07-15,4.255
2025-07-16,4.2593
2025-07-17,4.2544
2025-07-18,4.2493
2025-07-21,4.2425
2025-07-22,4.2565
2025-07-23,4.2545
2025-07-24,4.2553
2025-07-25,4.2543
2025-07-28,4.2595
2025-07-29,4.2805
2025-07-30,4.2675
2025-07-31,4.2728
2025-08-01,4.2755
2025-08-04,4.2758
2025-08-05,4.2765
2025-08-06,4.278
2025-08-07,4.2593
2025-08-08,4.251
2025-08-11,4.2535
2025-08-12,4.2603
2025-08-13,4.26
2025-08-14,4.2605
2025-08-15,4.2588
2025-08-18,4.25
2025-08-19,4.2448
2025-08-20,4.2483
2025-08-21,4.2518
2025-08-22,4.2628
2025-08-25,4.2616
2025-08-26,4.2608
2025-08-27,4.2698
2025-08-28,4.2618
2025-08-29,4.2665
2025-09-01,4.2553
2025-09-02,4.2653
2025-09-03,4.2523
2025-09-04,4.251
2025-09-05,4.2505
2025-09-08,4.2495
2025-09-09,4.2558
2025-09-10,4.2645
2025-09-11,4.2653
2025-09-12,4.2575
2025-09-15,4.2505
2025-09-16,4.2513
2025-09-17,4.2525
2025-09-18,4.256
2025-09-19,4.263
2025-09-22,4.259
2025-09-23,4.2553
2025-09-24,4.2673
2025-09-25,4.2638
2025-09-26,4.2705
2025-09-29,4.2695
2025-09-30,4.2698
2025-10-01,4.2583
2025-10-02,4.2565
2025-10-03,4.2555
2025-10-06,4.2528
2025-10-07,4.2535
2025-10-08,4.2508
2025-10-09,4.2548
2025-10-10,4.2576
2025-10-13,4.2615
2025-10-14,4.2625
2025-10-15,4.256
2025-10-16,4.2513
2025-10-17,4.245
2025-10-20,4.2468
2025-10-21,4.2393
2025-10-22,4.2312
2025-10-23,4.2327
2025-10-24,4.2408
2025-10-27,4.235
2025-10-28,4.2375
2025-10-29,4.2418
2025-10-30,4.245
2025-10-31,4.256
2025-11-03,4.254
2025-11-04,4.257
2025-11-05,4.2583
2025-11-06,4.2528
2025-11-07,4.248
2025-11-10,4.234
2025-11-11,4.23
2025-11-12,4.228
2025-11-13,4.228
2025-11-14,4.2328
2025-11-17,4.2228
2025-11-18,4.244
2025-11-19,4.2283
2025-11-20,4.232
2025-11-21,4.2448
2025-11-24,4.2303
2025-11-25,4.2228
2025-11-26,4.2313
2025-11-27,4.2345
2025-11-28,4.239
2025-12-01,4.2283
2025-12-02,4.238
2025-12-03,4.2318
2025-12-04,4.2313
2025-12-05,4.2323
2025-12-08,4.2305
2025-12-09,4.2303
2025-12-10,4.2248
2025-12-11,4.227
2025-12-12,4.2225
2025-12-15,4.2215
2025-12-16,4.2208
2025-12-17,4.21
2025-12-18,4.2053
2025-12-19,4.2105
2025-12-22,4.2138
2025-12-23,4.232
2025-12-24,4.2135
2025-12-29,4.2295
2025-12-30,4.2263
2025-12-31,4.221
2026-01-02,4.2123
2026-01-05,4.2178
2026-01-06,4.2105
2026-01-07,4.2148
2026-01-08,4.2093
2026-01-09,4.2138
2026-01-12,4.2093
2026-01-13,4.2118
2026-01-14,4.218
2026-01-15,4.2093
2026-01-16,4.2248
2026-01-19,4.2253
2026-01-20,4.2273
2026-01-21,4.2275
2026-01-22,4.2063
2026-01-23,4.2035
2026-01-26,4.2085
2026-01-27,4.2023
2026-01-28,4.2028
2026-01-29,4.2033
2026-01-30,4.2073
2026-02-02,4.2173
2026-02-03,4.2205
2026-02-04,4.2235
2026-02-05,4.2205
2026-02-06,4.2185
2026-02-09,4.2118
2026-02-10,4.217
2026-02-11,4.2135
2026-02-12,4.2158
2026-02-13,4.215
2026-02-16,4.2105
2026-02-17,4.2138
2026-02-18,4.2165
2026-02-19,4.2215
2026-02-20,4.2238
2026-02-23,4.2173
2026-02-24,4.2203
2026-02-25,4.2233
2026-02-26,4.2225
2026-02-27,4.2243
2026-03-02,4.244
2026-03-03,4.2865
2026-03-04,4.2588
2026-03-05,4.2725
2026-03-06,4.2875
2026-03-09,4.2785
2026-03-10,4.2548
2026-03-11,4.2523
2026-03-12,4.2578
2026-03-13,4.2675
2026-03-16,4.2683
2026-03-17,4.2593
2026-03-18,4.2713
2026-03-19,4.2853
2026-03-20,4.2783
2026-03-23,4.2643
2026-03-24,4.2743
2026-03-25,4.2718
2026-03-26,4.2743
2026-03-27,4.2875
2026-03-30,4.2883
2026-03-31,4.289
2026-04-01,4.2793
2026-04-02,4.2855
2026-04-07,4.2753
2026-04-08,4.2538
2026-04-09,4.2558
2026-04-10,4.2435
2026-04-13,4.2543
2026-04-14,4.2425
2026-04-15,4.2413
2026-04-16,4.2423
2026-04-17,4.2328
2026-04-20,4.2335
2026-04-21,4.2295
2026-04-22,4.2468
2026-04-23,4.2428
2026-04-24,4.2403
2026-04-27,4.2443
2026-04-28,4.2478
2026-04-29,4.2518
2026-04-30,4.2605
2026-05-04,4.257
2026-05-05,4.2543
2026-05-06,4.233
2026-05-07,4.227
2026-05-08,4.2318
2026-05-11,4.2398
2026-05-12,4.248
2026-05-13,4.2513
2026-05-14,4.2378
2026-05-15,4.2465
2026-05-18,4.2443
2026-05-19,4.2435
2026-05-20,4.255
2026-05-21,4.2483
2026-05-22,4.242
2026-05-25,4.2345
2026-05-26,4.2315
2026-05-27,4.2365
2026-05-28,4.2335
2026-05-29,4.2275
2026-06-01,4.233
2026-06-02,4.2345
2026-06-03,4.2383
2026-06-04,4.2368
2026-06-05,4.2338
2026-06-08,4.2423
2026-06-09,4.2373
2026-06-10,4.249
2026-06-11,4.2518
2026-06-12,4.248
2026-06-15,4.246
2026-06-16,4.2429
2026-06-17,4.241
2026-06-18,4.258
2026-06-19,4.2615
2026-06-22,4.268
2026-06-23,4.2825
2026-06-24,4.2878
2026-06-25,4.2875
2026-06-26,4.2868
2026-06-29,4.2873
2026-06-30,4.2955
2026-07-01,4.2958
2026-07-02,4.29
2026-07-03,4.2848
2026-07-06,4.2883
2026-07-07,4.2928
2026-07-08,4.3095
2026-07-09,4.3085
2026-07-10,4.3475
2026-07-13,4.3238
2026-07-14,4.3383
2026-07-15,4.3258
2026-07-16,4.3285
2026-07-17,4.3473
2026-07-20,4.332
2026-07-21,4.3305
2026-07-22,4.33
2026-07-23,4.327
2026-07-24,4.3155
2026-07-27,4.322
2026-07-28,4.3265
2026-07-29,4.327
2026-07-30,4.3078
2026-07-31,4.3135
2026-08-03,4.3033
2026-08-04,4.3063
2026-08-05,4.306
2026-08-06,4.299
2026-08-07,4.2983
2026-08-10,4.2993
2026-08-11,4.2965
2026-08-12,4.3043
2026-08-13,4.3048
2026-08-14,4.3068
2026-08-17,4.3063
2026-08-18,4.319
2026-08-19,4.3245
2026-08-20,4.3188
2026-08-21,4.3078
2026-08-24,4.3078
2026-08-25,4.3055
2026-08-26,4.2955
2026-08-27,4.3268
2026-08-28,4.3365
2026-08-31,4.328
2026-09-01,4.3313
2026-09-02,4.3265
2026-09-03,4.3265
2026-09-04,4.3148
2026-09-07,4.3103
2026-09-08,4.3178
2026-09-09,4.3153
2026-09-10,4.322
2026-09-11,4.325
2026-09-14,4.3418
2026-09-15,4.34
2026-09-16,4.3473
2026-09-17,4.358
2026-09-18,4.3635
2026-09-21,4.353
2026-09-22,4.348
2026-09-23,4.3765
2026-09-24,4.3823
2026-09-25,4.3718
2026-09-28,4.373
2026-09-29,4.3653
2026-09-30,4.369
2026-10-01,4.3735
2026-10-02,4.3775
2026-10-05,4.3795
2026-10-06,4.365
2026-10-07,4.3825
2026-10-08,4.3753
2026-10-09,4.3835
2026-10-12,4.385
2026-10-13,4.3768
2026-10-14,4.3658
2026-10-15,4.37
2026-10-16,4.37
2026-10-19,4.3713
2026-10-20,4.3768
2026-10-21,4.3635
2026-10-22,4.3695
2026-10-23,4.3945
2026-10-26,4.4038
2026-10-27,4.402
2026-10-28,4.4045
2026-10-29,4.3938
2026-10-30,4.4038
2026-11-02,4.4185
2026-11-03,4.419
2026-11-04,4.399
2026-11-05,4.389
2026-11-06,4.3925
2026-11-09,4.3843
2026-11-10,4.387
2026-11-11,4.4038
2026-11-12,4.4088
2026-11-13,4.4113
2026-11-16,4.403
2026-11-17,4.4038
2026-11-18,4.4118
2026-11-19,4.417
2026-11-20,4.4583
2026-11-23,4.4588
2026-11-24,4.4599
"""


if __name__ == "__main__":
    main()

