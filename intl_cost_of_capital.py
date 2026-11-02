"""Streamlit cost-of-capital calculator.

Run: python -m streamlit run app.py
Rates inside the model are decimals; input widgets display percentages.
Sources are downloaded at run time. No historical yields are hardcoded.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
import re
import unicodedata
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
import requests
import streamlit as st

BETA_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls"
BETA_PAGE = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/Betas.html"
TAX_URL = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ctryprem.html"
RATINGS_URL = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ratings.html"
BONDS_URL = "https://tradingeconomics.com/bonds"
TE_API = "https://api.tradingeconomics.com/markets/bond"
DEFAULT_INDUSTRY = "Software (System & Application)"

# Country labels are matched exactly after normalization. No fuzzy tax matching.
COUNTRY_ALIASES = {
    "korea south": "south korea", "korea republic of": "south korea",
    "republic of korea": "south korea", "korea": "south korea",
    "turkiye": "turkey", "turkey turkiye": "turkey", "czechia": "czech republic",
    "united states of america": "united states", "usa": "united states",
    "russian federation": "russia", "viet nam": "vietnam",
    "slovak republic": "slovakia", "uk": "united kingdom",
}
CURRENCIES = {
    "United States": "USD", "United Kingdom": "GBP", "Japan": "JPY",
    "Australia": "AUD", "New Zealand": "NZD", "Canada": "CAD",
    "Brazil": "BRL", "Russia": "RUB", "India": "INR", "China": "CNY",
    "South Africa": "ZAR", "Switzerland": "CHF", "Mexico": "MXN",
    "South Korea": "KRW", "Korea": "KRW", "Turkey": "TRY", "Norway": "NOK",
    "Poland": "PLN", "Romania": "RON", "Sweden": "SEK", "Hungary": "HUF",
    "Iceland": "ISK", "Czech Republic": "CZK", "Denmark": "DKK",
    "Peru": "PEN", "Chile": "CLP", "Colombia": "COP", "Singapore": "SGD",
    "Taiwan": "TWD", "Thailand": "THB", "Vietnam": "VND",
    "Hong Kong": "HKD", "Indonesia": "IDR", "Israel": "ILS",
    "Malaysia": "MYR", "Pakistan": "PKR", "Philippines": "PHP",
    "Zambia": "ZMW", "Kenya": "KES", "Nigeria": "NGN",
}
for _country in (
    "Austria Belgium Croatia Cyprus Estonia Finland France Germany Greece Ireland "
    "Italy Latvia Lithuania Luxembourg Malta Netherlands Portugal Slovakia Slovenia Spain"
).split():
    CURRENCIES[_country] = "EUR"


class DataError(RuntimeError):
    """A source could not be retrieved or confidently interpreted."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalized(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def country_key(value) -> str:
    # Some source rows append update notices, e.g. Turkey (updated February 2026).
    key = normalized(re.sub(r"\s*\(updated[^)]*\)", "", str(value), flags=re.I))
    return COUNTRY_ALIASES.get(key, key)


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def number(value) -> float:
    """Parse an unambiguous displayed number; never replace missing values with zero."""
    if value is None:
        raise ValueError("Missing number")
    text = str(value).strip().replace(",", "").replace("%", "").replace("\u2212", "-")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ValueError(f"Not a number: {str(value)[:40]}")
    result = float(text)
    if not math.isfinite(result):
        raise ValueError("Non-finite number")
    return result


def fetch_bytes(url: str, params=None) -> bytes:
    """Bounded request; errors deliberately omit credential-bearing request URLs."""
    try:
        response = requests.get(
            url, params=params, timeout=(8, 20),
            headers={"User-Agent": "Mozilla/5.0 (compatible; CostOfCapitalCalculator/1.0)",
                     "Accept": "text/html,application/json,application/vnd.ms-excel,*/*"},
        )
        if response.status_code != 200:
            raise DataError(f"Source returned HTTP {response.status_code}.")
        if len(response.content) > 20_000_000:
            raise DataError("Source response exceeded the expected size.")
        return response.content
    except requests.RequestException as exc:
        raise DataError(f"Could not connect to source ({type(exc).__name__}).") from None


def parse_betas(content: bytes) -> tuple[pd.DataFrame, str]:
    """Locate the header by label, not row number, and retain full-precision betas."""
    sheets = pd.read_excel(BytesIO(content), sheet_name=None, header=None, engine="xlrd")
    for raw in sheets.values():
        for row_index, row in raw.iterrows():
            labels = [normalized(x) for x in row]
            if "industry name" not in labels or "unlevered beta" not in labels:
                continue
            columns = {label: index for index, label in enumerate(labels)}
            cash_col = next((i for i, label in enumerate(labels)
                             if label.startswith("unlevered beta") and "cash" in label), None)
            records = []
            for _, values in raw.iloc[row_index + 1:].iterrows():
                industry = clean_text(values.iloc[columns["industry name"]])
                if normalized(industry).startswith("total market") or industry in ("nan", ""):
                    continue
                try:
                    beta = number(values.iloc[columns["unlevered beta"]])
                    firms = int(number(values.iloc[columns["number of firms"]]))
                except (ValueError, KeyError):
                    continue
                cash_beta = None
                if cash_col is not None:
                    try:
                        cash_beta = number(values.iloc[cash_col])
                    except ValueError:
                        pass
                if firms > 0:
                    records.append({"Industry": industry, "Firms": firms,
                                    "Unlevered beta": beta,
                                    "Cash-adjusted unlevered beta": cash_beta})
            date = "Not supplied"
            for _, values in raw.iloc[:row_index].iterrows():
                if normalized(values.iloc[0]) == "date updated":
                    value = values.iloc[1]
                    if isinstance(value, (float, int)):
                        date = (pd.Timestamp("1899-12-30") + pd.to_timedelta(value, unit="D")).strftime("%Y-%m-%d")
                    else:
                        date = pd.to_datetime(value).strftime("%Y-%m-%d")
            frame = pd.DataFrame(records).drop_duplicates("Industry").sort_values("Industry")
            if len(frame) < 20:
                raise DataError("Too few industries found; the beta file may have changed.")
            return frame.reset_index(drop=True), date
    raise DataError("Could not identify the industry and unlevered-beta columns.")


def parse_taxes(content: bytes) -> tuple[pd.DataFrame, str]:
    soup = BeautifulSoup(content, "html.parser")
    for table in soup.find_all("table"):
        headers = None
        records = []
        for row in table.find_all("tr"):
            cells = [clean_text(c.get_text(" ", strip=True))
                     for c in row.find_all(["td", "th"], recursive=False)]
            labels = [normalized(c) for c in cells]
            if "country" in labels and "corporate tax rate" in labels:
                headers = {name: index for index, name in enumerate(labels)}
                continue
            if headers is None or len(cells) <= max(headers.values()):
                continue
            try:
                tax = number(cells[headers["corporate tax rate"]]) / 100
                country = cells[headers["country"]]
                if not 0 <= tax <= 1 or not country:
                    continue
                records.append({"Country": country, "Tax rate": tax, "Key": country_key(country)})
            except ValueError:
                continue
        if len(records) >= 20:
            text = soup.get_text(" ", strip=True)
            match = re.search(r"Last\s+updated\s*:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", text, re.I)
            date = match.group(1) if match else "Not supplied"
            return pd.DataFrame(records).drop_duplicates("Key").sort_values("Country"), date
    raise DataError("Could not identify Damodaran's country corporate-tax table.")


def parse_bbb_spread(content: bytes) -> tuple[float, str]:
    """Read the exact BBB row in the large non-financial firms' rating columns."""
    soup = BeautifulSoup(content, "html.parser")
    text = clean_text(soup.get_text(" ", strip=True))
    match = re.search(r"(?:Data\s+used\s+is\s+as\s+of|Last\s+updated\s*:)\s*"
                      r"([A-Za-z]+\s+(?:\d{1,2},?\s+)?\d{4})", text, re.I)
    date = match.group(1) if match else "Not supplied"
    for table in soup.find_all("table"):
        if not re.search(r"\bnon\s?financial\b", normalized(table.get_text(" ", strip=True))):
            continue
        rating_col = spread_col = None
        spreads = []
        for row in table.find_all("tr"):
            cells = [clean_text(c.get_text(" ", strip=True))
                     for c in row.find_all(["td", "th"], recursive=False)]
            labels = [normalized(c) for c in cells]
            # The first rating/spread pair belongs to large non-financial firms;
            # the separate financial-firm pair must not override it.
            if "rating is" in labels or "rating" in labels:
                rating_col = next(i for i, label in enumerate(labels)
                                  if label in ("rating", "rating is"))
                spread_col = next((i for i in range(rating_col + 1, len(labels))
                                   if labels[i] in ("spread", "spread is", "default spread")), None)
                continue
            if rating_col is None or spread_col is None or len(cells) <= max(rating_col, spread_col):
                continue
            if not re.fullmatch(r"(?:Baa2\s*/\s*)?BBB", cells[rating_col], re.I):
                continue
            try:
                if not cells[spread_col].endswith("%"):
                    raise ValueError("Spread units are not percentages")
                spread = number(cells[spread_col]) / 100
                if not 0 <= spread <= .5:
                    raise ValueError("Spread outside expected range")
            except ValueError as exc:
                raise DataError("Damodaran's BBB spread is missing or invalid.") from exc
            spreads.append(spread)
        if spreads:
            if len(set(spreads)) != 1:
                raise DataError("Damodaran's table contains conflicting BBB spreads.")
            return spreads[0], date
    raise DataError("Could not identify the BBB spread in Damodaran's non-financial-firm table.")


def ten_year_symbol(symbol: str) -> bool:
    return bool(re.search(r"(?<!\d)10(?:YR|Y|:|$)", symbol.upper()))


def parse_bonds_html(content: bytes) -> dict:
    """Use current price cells only, never forecasts, change columns or prose."""
    soup = BeautifulSoup(content, "html.parser")
    quotes = {}
    for row in soup.select("tr[data-symbol]"):
        symbol = row.get("data-symbol", "")
        anchor = row.select_one('a[href$="/government-bond-yield"]')
        price = row.select_one('[id="p"]')
        if not anchor or not price or not ten_year_symbol(symbol):
            continue
        try:
            value = number(price.get_text(strip=True)) / 100
        except ValueError:
            continue
        if not -0.1 < value < 2:
            continue
        name = clean_text(anchor.get_text(" ", strip=True))
        key = country_key(name)
        if key in quotes:  # Keep the first quote if a country appears in several groups.
            continue
        date = row.select_one('[id="date"]')
        quotes[key] = {
            "country": name, "rate": value, "symbol": symbol,
            "quote_date": clean_text(date.get_text()) if date else "Not supplied",
            "url": urljoin(BONDS_URL, anchor["href"]), "route": "Trading Economics website",
        }
    if "united states" not in quotes:
        raise DataError("No identifiable U.S. 10-year yield found on Trading Economics.")
    return quotes


def parse_bonds_api(payload) -> dict:
    if not isinstance(payload, list):
        raise DataError("Unexpected Trading Economics API response.")
    quotes = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("Country", "")
        symbol = str(item.get("Symbol", ""))
        is_ten_year = ten_year_symbol(symbol) or bool(re.search(r"\b10Y\b", str(item.get("Name", "")), re.I))
        if not name or not is_ten_year:
            continue
        # Reject series explicitly labelled as foreign-currency debt.
        if re.search(r"\b(?:USD|EUR|dollar|euro)\b", str(item.get("Name", "")), re.I):
            continue
        if str(item.get("unit", "percent")).lower() not in ("percent", "%", "percentage"):
            continue
        try:
            value = number(item.get("Last")) / 100
        except ValueError:
            continue
        if not -0.1 < value < 2:
            continue
        quotes[country_key(name)] = {
            "country": name, "rate": value, "symbol": symbol,
            "quote_date": str(item.get("Date") or item.get("LastUpdate") or "Not supplied"),
            "url": urljoin(BONDS_URL, item.get("URL") or "/bonds"),
            "route": "Trading Economics API",
        }
    if "united states" not in quotes:
        raise DataError("The API did not return a usable U.S. 10-year yield.")
    return quotes


@st.cache_data(ttl=86400, show_spinner=False)
def load_industries():
    frame, date = parse_betas(fetch_bytes(BETA_URL))
    return {"data": frame, "date": date, "fetched": now_utc(), "source": BETA_URL}


@st.cache_data(ttl=86400, show_spinner=False)
def load_taxes():
    frame, date = parse_taxes(fetch_bytes(TAX_URL))
    return {"data": frame, "date": date, "fetched": now_utc(), "source": TAX_URL}


@st.cache_data(ttl=900, show_spinner=False)
def load_bonds(api_key: str = ""):
    # The credential is hashed in the cache key but is never displayed or exported.
    note = ""
    if api_key:
        try:
            payload = json.loads(fetch_bytes(TE_API, {"c": api_key, "type": "10Y", "f": "json"}))
            return {"quotes": parse_bonds_api(payload), "fetched": now_utc(), "note": ""}
        except (DataError, ValueError):
            note = "The configured API was unavailable; quotes were retrieved from the public website."
    return {"quotes": parse_bonds_html(fetch_bytes(BONDS_URL)), "fetched": now_utc(), "note": note}


@st.cache_data(ttl=86400, show_spinner=False)
def load_bbb_spread():
    spread, date = parse_bbb_spread(fetch_bytes(RATINGS_URL))
    return {"spread": spread, "rating": "BBB", "date": date,
            "fetched": now_utc(), "source": RATINGS_URL}


def calculate(beta_u: float, debt_fraction: float, tax: float, erp: float,
              rf_us: float, rf_local: float, spread: float) -> dict:
    """Compute USD and selected-currency component costs, then each WACC.

    Hamada relevering assumes zero debt beta. The chosen tax jurisdiction and
    capital structure are held fixed across the two currency scenarios.
    Each borrowing cost is its own currency's 10-year yield plus the same spread.
    The local WACC is a financing scenario with interest deductible locally.
    """
    values = (beta_u, debt_fraction, tax, erp, rf_us, rf_local, spread)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("All inputs must be finite numbers.")
    if not 0 <= debt_fraction < 1:
        raise ValueError("Debt financing must be at least 0% and below 100%; 100% leaves no equity.")
    if not 0 <= tax <= 1 or erp < 0 or spread < 0:
        raise ValueError("Tax must be 0–100%; the equity premium and borrowing spread must be nonnegative.")
    if rf_us <= -1 or rf_local <= -1:
        raise ValueError("Risk-free rates must exceed -100%.")
    de = debt_fraction / (1 - debt_fraction)
    beta_e = beta_u * (1 + (1 - tax) * de)
    equity_us = rf_us + beta_e * erp
    debt_us = rf_us + spread
    factor = (1 + rf_local) / (1 + rf_us)
    equity_local = (1 + equity_us) * factor - 1
    debt_local = rf_local + spread
    wacc_us = (1 - debt_fraction) * equity_us + debt_fraction * debt_us * (1 - tax)
    wacc_local = (1 - debt_fraction) * equity_local + debt_fraction * debt_local * (1 - tax)
    return {"debt_equity": de, "beta_equity": beta_e, "conversion_factor": factor,
            "equity_us": equity_us, "debt_us": debt_us, "wacc_us": wacc_us,
            "equity_local": equity_local, "debt_local": debt_local, "wacc_local": wacc_local}


def rate_input(label: str, quote: dict | None, key: str) -> tuple[float | None, bool]:
    manual = st.checkbox(f"Enter {label} manually", value=quote is None, key=f"{key}_manual")
    if quote:
        st.caption(f"Source quote: {quote['rate']:.4%} · {quote['quote_date']}")
    if manual:
        value = st.number_input(label + " (%)", min_value=-9.99, max_value=199.99,
                                value=None, step=0.01, format="%.4f", key=f"{key}_value",
                                placeholder="Enter an annual 10-year yield")
        return (None if value is None else value / 100), True
    return (quote["rate"] if quote else None), False


def get_api_key() -> str:
    key = os.environ.get("TE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("TE_API_KEY", "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    return str(key)


def spread_input(benchmark: dict | None) -> tuple[float | None, bool]:
    """Follow source updates in automatic mode; preserve explicit manual overrides."""
    sourced_pct = benchmark["spread"] * 100 if benchmark else None
    manual = st.checkbox("Override BBB borrowing spread", value=benchmark is None,
                         key="spread_override")
    if benchmark:
        st.caption(f"Damodaran BBB spread: {benchmark['spread']:.2%} "
                   f"({benchmark['spread'] * 10000:.0f} basis points) · {benchmark['date']}")
    else:
        st.info("BBB spread unavailable. Enter a borrowing spread manually or refresh source data.")
    if not manual or "spread" not in st.session_state:
        st.session_state["spread"] = sourced_pct
    value = st.number_input("Borrowing spread over local 10-year yield (percentage points)",
                            min_value=0.0, max_value=50.0, value=None, step=0.01, format="%.2f",
                            key="spread", disabled=not manual,
                            placeholder="Enter a borrowing spread",
                            help="Uses Damodaran's BBB spread unless overridden. 1.00 means 100 basis points. "
                                 "The spread is added to the selected country's 10-year yield, "
                                 "and to the U.S. yield for the USD comparison.")
    return value, manual


def main():
    st.set_page_config(page_title="Cost of capital", page_icon="📊", layout="wide")
    st.title("Cost of capital")
    st.caption("U.S. industry betas, corporate taxes, BBB borrowing spreads, and 10-year government bond yields")
    with st.sidebar:
        st.header("Project assumptions")
        if st.button("Refresh source data", width="stretch"):
            load_industries.clear()
            load_taxes.clear()
            load_bonds.clear()
            load_bbb_spread.clear()
            st.rerun()

    api_key = get_api_key()
    results, failures = {}, {}
    loaders = {"industries": load_industries, "taxes": load_taxes,
               "bonds": lambda: load_bonds(api_key), "BBB spread": load_bbb_spread}
    with st.spinner("Retrieving source data…"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {name: pool.submit(loader) for name, loader in loaders.items()}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                    st.session_state[f"last_good_{name}"] = results[name]
                except Exception as exc:
                    failures[name] = str(exc) if isinstance(exc, DataError) else type(exc).__name__
                    if f"last_good_{name}" in st.session_state:
                        results[name] = st.session_state[f"last_good_{name}"]
    for name, error in failures.items():
        if name in results:
            st.warning(f"Could not refresh {name}. Using this session's last successful retrieval "
                       f"from {results[name]['fetched']}. {error}")
        else:
            st.warning(f"Could not retrieve {name}. {error}")
    if "industries" not in results:
        st.error("Industry data is required. Check the connection and use Refresh source data.")
        st.link_button("Open Damodaran's industry beta page", BETA_PAGE)
        st.stop()

    industry_data = results["industries"]
    industries = industry_data["data"]
    taxes = results.get("taxes")
    bbb_spread = results.get("BBB spread")
    bonds = results.get("bonds", {"quotes": {}, "fetched": "Unavailable", "note": ""})
    quotes = bonds["quotes"]
    if bonds["note"]:
        st.info(bonds["note"])
    names = industries["Industry"].tolist()
    with st.sidebar:
        default_index = names.index(DEFAULT_INDUSTRY) if DEFAULT_INDUSTRY in names else 0
        if DEFAULT_INDUSTRY not in names:
            st.info("The default software category is absent from the current data. Choose an industry.")
        industry = st.selectbox("U.S. industry", names, index=default_index, key="industry")
        beta_column = st.selectbox("Unlevered beta measure",
                                  ["Unlevered beta", "Cash-adjusted unlevered beta"], key="beta_measure",
                                  help="The standard measure includes cash. The adjusted measure isolates operating assets.")
        erp_pct = st.number_input("Annual equity risk premium (%)", 0.0, 50.0, 6.5, 0.1, key="erp")
        debt_pct = st.slider("Fraction financed with debt (%)", 0.0, 99.0, 30.0, 1.0, key="debt_fraction",
                             help="Target market-value debt share D/(D+E). 100% debt leaves no equity beta.")
        spread_pct, override_spread = spread_input(bbb_spread)
        available_countries = sorted(set(taxes["data"]["Country"]) if taxes else
                                     (set(CURRENCIES) | {q["country"] for q in quotes.values()}))
        country = st.selectbox("Project country / currency", available_countries,
                               index=available_countries.index("United States") if "United States" in available_countries else 0,
                               key="country")
        matched_tax = taxes["data"].loc[taxes["data"]["Key"] == country_key(country)] if taxes else pd.DataFrame()
        sourced_tax = float(matched_tax.iloc[0]["Tax rate"]) if not matched_tax.empty else None
        override_tax = st.checkbox("Override corporate tax rate", value=sourced_tax is None, key="tax_override")
        if sourced_tax is not None:
            st.caption(f"Damodaran corporate tax rate: {sourced_tax:.2%}")
        if override_tax:
            tax_value = st.number_input("Corporate tax rate (%)", 0.0, 100.0,
                                        value=None, step=0.5, key=f"tax_{country}",
                                        placeholder="Enter the applicable marginal rate")
            tax = None if tax_value is None else tax_value / 100
        else:
            tax = sourced_tax

    row = industries.loc[industries["Industry"] == industry].iloc[0]
    beta_u = row[beta_column]
    if pd.isna(beta_u):
        st.error("This beta measure is missing for the selected industry. Select the other measure.")
        st.stop()
    us_quote = quotes.get("united states")
    local_quote = quotes.get(country_key(country))
    is_us = country_key(country) == "united states"
    currency = next((code for name, code in CURRENCIES.items()
                     if country_key(name) == country_key(country)), "local currency")
    st.subheader("10-year yields used in the calculation")
    a, b = st.columns(2)
    with a:
        rf_us, manual_us = rate_input("U.S. 10-year yield", us_quote, "rf_us")
    with b:
        if is_us:
            rf_local, manual_local = rf_us, manual_us
            st.write("Selected country: United States (USD)")
            st.caption("The selected-country yield equals the U.S. yield. No currency conversion is needed.")
        else:
            if local_quote is None:
                st.info(f"No verified 10-year quote was retrieved for {country}. Enter its local-currency 10-year yield.")
            rf_local, manual_local = rate_input(f"{country} 10-year yield", local_quote, f"rf_{country}")

    if tax is None or rf_us is None or rf_local is None or spread_pct is None:
        st.info("Enter the missing tax rate, bond yield or borrowing spread to calculate results. Missing data is never assumed to be zero.")
        st.latex(r"\mathrm{WACC}=\frac{E}{D+E}R_e+\frac{D}{D+E}R_d(1-\tau)")
        st.stop()
    d, erp, spread = debt_pct / 100, erp_pct / 100, spread_pct / 100
    model = calculate(float(beta_u), d, tax, erp, rf_us, rf_local, spread)
    st.subheader(industry)
    st.caption(f"{int(row['Firms'])} U.S. firms. Beta data dated {industry_data['date']}. "
               f"Tax jurisdiction: {country}; the same tax rate is used in both currency scenarios.")
    a, b, c, e = st.columns(4)
    a.metric("U.S. 10-year yield", f"{rf_us:.3%}")
    b.metric(f"{country} 10-year yield", f"{rf_local:.3%}")
    c.metric("Corporate tax rate", f"{tax:.2%}")
    e.metric("Unlevered beta", f"{beta_u:.4f}")
    st.divider()
    if is_us:
        a, b = st.columns(2)
        a.metric("Equity cost of capital · USD", f"{model['equity_us']:.2%}")
        b.metric("WACC · USD", f"{model['wacc_us']:.2%}")
    else:
        a, b, c, e = st.columns(4)
        a.metric("Equity cost · USD", f"{model['equity_us']:.2%}")
        b.metric("WACC · USD", f"{model['wacc_us']:.2%}")
        c.metric(f"Equity cost · {currency}", f"{model['equity_local']:.2%}")
        e.metric(f"WACC · {currency}", f"{model['wacc_local']:.2%}")
    st.caption(f"Equity beta after leverage: {model['beta_equity']:.4f}. "
               f"D/E = {model['debt_equity']:.4f}. Pretax borrowing cost: {model['debt_us']:.3%} in USD" +
               (f" and {model['debt_local']:.3%} in {currency}." if not is_us else "."))

    st.subheader("WACC formula")
    st.latex(r"d=\frac{D}{D+E},\qquad\mathrm{WACC}=(1-d)R_e+dR_d(1-\tau)")
    st.latex(rf"\mathrm{{WACC}}_{{USD}}=(1-{d:.3f})\times {model['equity_us']:.5f}"
             rf"+{d:.3f}\times {model['debt_us']:.5f}\times(1-{tax:.4f})"
             rf"={model['wacc_us']*100:.3f}\%")
    if not is_us:
        st.latex(rf"\mathrm{{WACC}}^*=(1-{d:.3f})\times {model['equity_local']:.5f}"
                 rf"+{d:.3f}\times {model['debt_local']:.5f}\times(1-{tax:.4f})"
                 rf"={model['wacc_local']*100:.3f}\%")

    with st.expander("Equity beta, borrowing cost and currency conversion", expanded=True):
        st.latex(r"\beta_E=\beta_U\left[1+(1-\tau)\frac{d}{1-d}\right]")
        st.latex(r"R_e=R_f+\beta_E\,ERP,\qquad R_d=R_f+s")
        st.latex(r"1+R_e^*=(1+R_e)\frac{1+R_f^*}{1+R_f}")
        st.latex(r"R_d^*=R_f^*+s")
        st.latex(r"\mathrm{WACC}^*=(1-d)R_e^*+dR_d^*(1-\tau)")
        st.write("The star denotes the selected country's currency. Rf is the U.S. 10-year yield; "
                 "Rf* is the selected country's 10-year yield. All rates in the formulas are annual decimals. "
                 "The borrowing spread is added directly to the selected country's 10-year yield. "
                 "For the USD comparison, the same spread is added to the U.S. 10-year yield.")
        st.write("The local-currency WACC combines the converted equity cost with the local 10-year "
                 "yield plus the borrowing spread, applying the tax deduction to local interest. "
                 "The gross-return currency conversion applies only to the equity cost.")

    with st.expander("Assumptions and interpretation", expanded=True):
        st.markdown("""
- The selected U.S. industry's unlevered beta is the project's business-risk proxy in every country. It is relevered using the project's target debt share.
- Debt and equity weights are target weights. The industry sample's debt ratio is not used as the project's financing ratio.
- The selected country's Damodaran corporate tax rate proxies the marginal rate on interest deductions. An override allows a different tax jurisdiction or project-specific rate.
- Interest is fully deductible and tax benefits can be used immediately. There are no preferred shares, issuance fees, or financial-distress costs.
- The local pretax borrowing cost equals the selected country's 10-year yield plus the entered spread. The USD comparison adds that same spread to the U.S. yield. The spread is held fixed when the debt share changes; the model does not estimate how lenders would reprice higher leverage.
- The relevering formula assumes zero debt beta. With a positive borrowing spread, this is an approximation; the model does not separately estimate debt beta.
- Ten-year government yields are used as risk-free proxies. Their differences can include sovereign credit and liquidity risk as well as currency effects. Countries using the same currency can therefore produce different results under this convention.
- No additional country equity-risk premium is added.
- The same tax rate and financing weights are held fixed in the USD and local-currency calculations. All outputs are nominal annual rates. A 100% debt share is excluded because an equity cost cannot then be estimated by this formula.
""")

    source_rows = [
        {"Input": "U.S. industry betas", "Source": BETA_URL, "Source date": industry_data["date"],
         "Retrieved (UTC)": industry_data["fetched"], "Used": beta_column},
        {"Input": f"{country} corporate tax", "Source": TAX_URL,
         "Source date": taxes["date"] if taxes else "Unavailable",
         "Retrieved (UTC)": taxes["fetched"] if taxes else "Unavailable",
         "Used": f"{'Manual override' if override_tax else 'Source'}: {tax:.4%}"},
        {"Input": "BBB borrowing spread", "Source": RATINGS_URL,
         "Source date": bbb_spread["date"] if bbb_spread else "Unavailable",
         "Retrieved (UTC)": bbb_spread["fetched"] if bbb_spread else "Unavailable",
         "Used": f"{'Manual override' if override_spread else 'Damodaran BBB'}: {spread:.4%}"},
    ]
    for index, (name, quote, manual, rate) in enumerate([
            ("U.S. 10-year", us_quote, manual_us, rf_us),
            (f"{country} 10-year", local_quote, manual_local, rf_local)]):
        if is_us and index == 1:
            break
        source_rows.append({"Input": name, "Source": quote["url"] if quote else BONDS_URL,
                            "Source date": quote["quote_date"] if quote else "Unavailable",
                            "Retrieved (UTC)": bonds["fetched"],
                            "Used": f"{'Manual override' if manual else quote['route']}: {rate:.4%}"})
    with st.expander("Sources and data dates"):
        st.dataframe(pd.DataFrame(source_rows), hide_index=True, width="stretch",
                     column_config={"Source": st.column_config.LinkColumn("Source")})
        st.caption("Beta, tax and BBB spread downloads are cached for 24 hours; bond quotes for 15 minutes. "
                   "Quote dates are shown exactly as supplied and may omit a year or timezone. "
                   "Retrieval time is not the observation time. Refresh source data clears the caches.")
        if not api_key:
            st.caption("Public Trading Economics pages are used without an API key. "
                       "If access is blocked or the page structure changes, enter yields manually. "
                       "An optional TE_API_KEY environment variable or Streamlit secret enables the official API.")

    # A complete export makes the same assumptions reproducible for every industry.
    all_rows = []
    for _, item in industries.iterrows():
        if pd.isna(item[beta_column]):
            continue
        computed = calculate(float(item[beta_column]), d, tax, erp, rf_us, rf_local, spread)
        all_rows.append({"Industry": item["Industry"], "Firms": int(item["Firms"]),
                         "Unlevered beta used": item[beta_column], "Equity beta": computed["beta_equity"],
                         "Equity cost USD (%)": computed["equity_us"] * 100,
                         "WACC USD (%)": computed["wacc_us"] * 100,
                         "Equity cost local (%)": computed["equity_local"] * 100,
                         "WACC local (%)": computed["wacc_local"] * 100})
    with st.expander("Compare all U.S. industries"):
        st.dataframe(pd.DataFrame(all_rows), hide_index=True, width="stretch")
        st.caption("Every row uses the same country, tax, financing share, equity premium and borrowing spread.")
    export = {
        "generated_utc": now_utc(), "industry": industry, "country": country, "currency": currency,
        "beta_measure": beta_column, "beta_unlevered": float(beta_u), "debt_fraction": d,
        "tax_rate": tax, "equity_premium": erp, "rf_us": rf_us, "rf_local": rf_local,
        "borrowing_spread": spread, "results": model, "sources": source_rows,
        "borrowing_spread_overridden": override_spread, "bbb_spread_benchmark": bbb_spread,
        "source_refresh_failures": failures, "rate_units": "annual decimal",
        "local_wacc_convention": "Convert equity cost; local pretax debt cost equals local 10-year yield plus spread; apply local interest tax shield.",
    }
    a, b = st.columns(2)
    a.download_button("Download selected calculation (JSON)", json.dumps(export, indent=2),
                      "cost_of_capital.json", "application/json")
    csv = pd.DataFrame(all_rows).assign(
        Country=country, Currency=currency, **{"Tax rate (%)": tax * 100, "Debt share (%)": debt_pct,
        "ERP (%)": erp_pct, "Borrowing spread (pp)": spread_pct, "US 10Y (%)": rf_us * 100,
        "Borrowing spread overridden": override_spread,
        "BBB benchmark spread (pp)": bbb_spread["spread"] * 100 if bbb_spread else None,
        "BBB spread source date": bbb_spread["date"] if bbb_spread else "Unavailable",
        "BBB spread retrieval UTC": bbb_spread["fetched"] if bbb_spread else "Unavailable",
        "BBB spread source": RATINGS_URL,
        "Local 10Y (%)": rf_local * 100, "Beta measure": beta_column,
        "Beta source date": industry_data["date"], "Tax source date": taxes["date"] if taxes else "Manual",
        "Tax overridden": override_tax, "US yield overridden": manual_us,
        "Local yield overridden": manual_local, "US yield quote date": us_quote["quote_date"] if us_quote else "Manual",
        "Local yield quote date": local_quote["quote_date"] if local_quote else "Manual",
        "Bond retrieval UTC": bonds["fetched"], "Beta source": BETA_URL, "Tax source": TAX_URL,
        "Bond source": BONDS_URL, "Source refresh failures": "; ".join(failures)})
    b.download_button("Download all industries (CSV)", csv.to_csv(index=False),
                      "industry_costs_of_capital.csv", "text/csv")


if __name__ == "__main__":
    main()
