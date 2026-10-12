"""Two-country monthly real exchange rate app using the current IMF API.

Run with: streamlit run app.py
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Country names from the IMF country codelist, restricted to ISO country codes.
COUNTRY_NAMES = {'ABW': 'Aruba, Kingdom of the Netherlands',
 'AFG': 'Afghanistan, Islamic Republic of',
 'AGO': 'Angola',
 'AIA': 'Anguilla, United Kingdom-British Overseas Territory',
 'ALA': 'Åland Islands',
 'ALB': 'Albania',
 'AND': 'Andorra, Principality of',
 'ARE': 'United Arab Emirates',
 'ARG': 'Argentina',
 'ARM': 'Armenia, Republic of',
 'ASM': 'American Samoa',
 'ATF': 'French Southern Territories',
 'ATG': 'Antigua and Barbuda',
 'AUS': 'Australia',
 'AUT': 'Austria',
 'AZE': 'Azerbaijan, Republic of',
 'BDI': 'Burundi',
 'BEL': 'Belgium',
 'BEN': 'Benin',
 'BES': 'Bonaire, Saint Eustatius and Saba',
 'BFA': 'Burkina Faso',
 'BGD': 'Bangladesh',
 'BGR': 'Bulgaria',
 'BHR': 'Bahrain, Kingdom of',
 'BHS': 'Bahamas, The',
 'BIH': 'Bosnia and Herzegovina',
 'BLM': 'Saint Barthelemy',
 'BLR': 'Belarus, Republic of',
 'BLZ': 'Belize',
 'BMU': 'Bermuda',
 'BOL': 'Bolivia',
 'BRA': 'Brazil',
 'BRB': 'Barbados',
 'BRN': 'Brunei Darussalam',
 'BTN': 'Bhutan',
 'BVT': 'Bouvet Island',
 'BWA': 'Botswana',
 'CAF': 'Central African Republic',
 'CAN': 'Canada',
 'CCK': 'Cocos Islands',
 'CHE': 'Switzerland',
 'CHL': 'Chile',
 'CHN': "China, People's Republic of",
 'CIV': "Côte d'Ivoire",
 'CMR': 'Cameroon',
 'COD': 'Congo, Democratic Republic of the',
 'COG': 'Congo, Republic of',
 'COK': 'Cook Islands',
 'COL': 'Colombia',
 'COM': 'Comoros, Union of the',
 'CPV': 'Cabo Verde',
 'CRI': 'Costa Rica',
 'CUB': 'Cuba',
 'CUW': 'Curaçao, Kingdom of the Netherlands',
 'CXR': 'Christmas Island',
 'CYM': 'Cayman Islands',
 'CYP': 'Cyprus',
 'CZE': 'Czech Republic',
 'DEU': 'Germany',
 'DJI': 'Djibouti',
 'DMA': 'Dominica',
 'DNK': 'Denmark',
 'DOM': 'Dominican Republic',
 'DZA': 'Algeria',
 'ECU': 'Ecuador',
 'EGY': 'Egypt, Arab Republic of',
 'ERI': 'Eritrea, The State of',
 'ESH': 'Western Sahara',
 'ESP': 'Spain',
 'EST': 'Estonia, Republic of',
 'ETH': 'Ethiopia, The Federal Democratic Republic of',
 'FIN': 'Finland',
 'FJI': 'Fiji, Republic of',
 'FLK': 'Falkland Islands (Malvinas)',
 'FRA': 'France',
 'FRO': 'Faroe Islands',
 'FSM': 'Micronesia, Federated States of',
 'GAB': 'Gabon',
 'GBR': 'United Kingdom',
 'GEO': 'Georgia',
 'GGY': 'Guernsey',
 'GHA': 'Ghana',
 'GIB': 'Gibraltar',
 'GIN': 'Guinea',
 'GLP': 'Guadeloupe',
 'GMB': 'Gambia, The',
 'GNB': 'Guinea-Bissau',
 'GNQ': 'Equatorial Guinea, Republic of',
 'GRC': 'Greece',
 'GRD': 'Grenada',
 'GRL': 'Greenland',
 'GTM': 'Guatemala',
 'GUF': 'French Guiana',
 'GUM': 'Guam',
 'GUY': 'Guyana',
 'HKG': "Hong Kong Special Administrative Region, People's Republic of China",
 'HMD': 'Heard Island and McDonald Islands',
 'HND': 'Honduras',
 'HRV': 'Croatia, Republic of',
 'HTI': 'Haiti',
 'HUN': 'Hungary',
 'IDN': 'Indonesia',
 'IMN': 'Isle of Man',
 'IND': 'India',
 'IOT': 'British Indian Ocean Territory',
 'IRL': 'Ireland',
 'IRN': 'Iran, Islamic Republic of',
 'IRQ': 'Iraq',
 'ISL': 'Iceland',
 'ISR': 'Israel',
 'ITA': 'Italy',
 'JAM': 'Jamaica',
 'JEY': 'Jersey',
 'JOR': 'Jordan',
 'JPN': 'Japan',
 'KAZ': 'Kazakhstan, Republic of',
 'KEN': 'Kenya',
 'KGZ': 'Kyrgyz Republic',
 'KHM': 'Cambodia',
 'KIR': 'Kiribati',
 'KNA': 'St. Kitts and Nevis',
 'KOR': 'Korea, Republic of',
 'KWT': 'Kuwait',
 'LAO': "Lao People's Democratic Republic",
 'LBN': 'Lebanon',
 'LBR': 'Liberia',
 'LBY': 'Libya',
 'LCA': 'St. Lucia',
 'LIE': 'Liechtenstein, Principality of',
 'LKA': 'Sri Lanka',
 'LSO': 'Lesotho, Kingdom of',
 'LTU': 'Lithuania, Republic of',
 'LUX': 'Luxembourg',
 'LVA': 'Latvia, Republic of',
 'MAC': "Macao Special Administrative Region, People's Republic of China",
 'MAF': 'Saint Martin',
 'MAR': 'Morocco',
 'MCO': 'Monaco',
 'MDA': 'Moldova, Republic of',
 'MDG': 'Madagascar, Republic of',
 'MDV': 'Maldives',
 'MEX': 'Mexico',
 'MHL': 'Marshall Islands, Republic of the',
 'MKD': 'North Macedonia, Republic of',
 'MLI': 'Mali',
 'MLT': 'Malta',
 'MMR': 'Myanmar',
 'MNE': 'Montenegro',
 'MNG': 'Mongolia',
 'MNP': 'Northern Mariana Islands',
 'MOZ': 'Mozambique, Republic of',
 'MRT': 'Mauritania, Islamic Republic of',
 'MSR': 'Montserrat, United Kingdom-British Overseas Territory',
 'MTQ': 'Martinique',
 'MUS': 'Mauritius',
 'MWI': 'Malawi',
 'MYS': 'Malaysia',
 'MYT': 'Mayotte',
 'NAM': 'Namibia',
 'NCL': 'New Caledonia',
 'NER': 'Niger',
 'NFK': 'Norfolk Island',
 'NGA': 'Nigeria',
 'NIC': 'Nicaragua',
 'NIU': 'Niue',
 'NLD': 'Netherlands, The',
 'NOR': 'Norway',
 'NPL': 'Nepal',
 'NRU': 'Naoero, Republic of',
 'NZL': 'New Zealand',
 'OMN': 'Oman',
 'PAK': 'Pakistan',
 'PAN': 'Panama',
 'PCN': 'Pitcairn Islands',
 'PER': 'Peru',
 'PHL': 'Philippines',
 'PLW': 'Palau, Republic of',
 'PNG': 'Papua New Guinea',
 'POL': 'Poland, Republic of',
 'PRI': 'Puerto Rico',
 'PRK': "Korea, Democratic People's Republic of",
 'PRT': 'Portugal',
 'PRY': 'Paraguay',
 'PYF': 'French Polynesia',
 'QAT': 'Qatar',
 'REU': 'Réunion',
 'ROU': 'Romania',
 'RUS': 'Russian Federation',
 'RWA': 'Rwanda',
 'SAU': 'Saudi Arabia',
 'SDN': 'Sudan',
 'SEN': 'Senegal',
 'SGP': 'Singapore',
 'SGS': 'South Georgia and the South Sandwich Islands',
 'SHN': 'St. Helena',
 'SJM': 'Svalbard and Jan Mayen Islands',
 'SLB': 'Solomon Islands',
 'SLE': 'Sierra Leone',
 'SLV': 'El Salvador',
 'SMR': 'San Marino, Republic of',
 'SOM': 'Somalia',
 'SPM': 'St. Pierre and Miquelon',
 'SRB': 'Serbia, Republic of',
 'SSD': 'South Sudan, Republic of',
 'STP': 'São Tomé and Príncipe, Democratic Republic of',
 'SUR': 'Suriname',
 'SVK': 'Slovak Republic',
 'SVN': 'Slovenia, Republic of',
 'SWE': 'Sweden',
 'SWZ': 'Eswatini, Kingdom of',
 'SXM': 'Sint Maarten, Kingdom of the Netherlands',
 'SYC': 'Seychelles',
 'SYR': 'Syrian Arab Republic',
 'TCA': 'Turks and Caicos Islands',
 'TCD': 'Chad',
 'TGO': 'Togo',
 'THA': 'Thailand',
 'TJK': 'Tajikistan, Republic of',
 'TKL': 'Tokelau',
 'TKM': 'Turkmenistan',
 'TLS': 'Timor-Leste, Democratic Republic of',
 'TON': 'Tonga',
 'TTO': 'Trinidad and Tobago',
 'TUN': 'Tunisia',
 'TUR': 'Türkiye, Republic of',
 'TUV': 'Tuvalu',
 'TWN': 'Taiwan Province of China',
 'TZA': 'Tanzania, United Republic of',
 'UGA': 'Uganda',
 'UKR': 'Ukraine',
 'UMI': 'United States Minor Outlying Islands',
 'URY': 'Uruguay',
 'USA': 'United States',
 'UZB': 'Uzbekistan, Republic of',
 'VAT': 'Holy See',
 'VCT': 'St. Vincent and the Grenadines',
 'VEN': 'Venezuela, República Bolivariana de',
 'VGB': 'British Virgin Islands',
 'VIR': 'United States Virgin Islands',
 'VNM': 'Vietnam',
 'VUT': 'Vanuatu',
 'WLF': 'Wallis and Futuna Islands',
 'WSM': 'Samoa',
 'YEM': 'Yemen, Republic of',
 'ZAF': 'South Africa',
 'ZMB': 'Zambia',
 'ZWE': 'Zimbabwe'}


API_BASE = "https://api.imf.org/external/sdmx/3.0"
START_MONTH = "1999-01"
# Use the euro-area ER series only from the country's euro adoption onward.
# Do not splice legacy currency units onto euros.
EURO_ADOPTION = {
    "AUT": "1999-01", "BEL": "1999-01", "DEU": "1999-01",
    "ESP": "1999-01", "FIN": "1999-01", "FRA": "1999-01",
    "IRL": "1999-01", "ITA": "1999-01", "LUX": "1999-01",
    "NLD": "1999-01", "PRT": "1999-01", "GRC": "2001-01",
    "SVN": "2007-01", "CYP": "2008-01", "MLT": "2008-01",
    "SVK": "2009-01", "EST": "2011-01", "LVA": "2014-01",
    "LTU": "2015-01", "HRV": "2023-01", "BGR": "2026-01",
}


class DataError(Exception):
    """A data or network issue that can be shown safely to the user."""


def get_api_key():
    # Most installations work without a key. A deployment may optionally
    # provide an IMF subscription key in Secrets; it is never sent to GitHub.
    try:
        return str(st.secrets.get("IMF_API_KEY", ""))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ""


def request_json(url, api_key=""):
    """Bounded retries for transient errors; never silently replace data."""
    headers = {"Accept": "application/vnd.sdmx.data+json;version=2.0.0"}
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key
    for attempt in range(3):
        try:
            response = requests.get(
                url, headers=headers,
                params={"startPeriod": START_MONTH}, timeout=(10, 40),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 2:
                raise DataError(
                    "The IMF download timed out or the connection failed after "
                    "three attempts. Try Refresh data, or use a saved export below."
                ) from exc
            time.sleep(2 ** attempt)
            continue
        except requests.RequestException as exc:
            raise DataError(f"Network request to IMF failed ({type(exc).__name__}).") from exc
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise DataError(f"IMF temporarily unavailable (HTTP {response.status_code}).")
        if response.status_code in (401, 403, 404):
            raise DataError(
                f"IMF returned HTTP {response.status_code}. The series may be "
                "unavailable, or this network may require API access credentials. "
                "An optional IMF subscription key can be stored as IMF_API_KEY "
                "in Streamlit Secrets."
            )
        if not response.ok:
            raise DataError(f"IMF download failed (HTTP {response.status_code}).")
        try:
            payload = response.json()
        except ValueError as exc:
            raise DataError("IMF returned an invalid or non-JSON response.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise DataError("IMF returned an unexpected response structure.")
        return payload
    raise DataError("IMF download failed.")


def month_timestamp(text):
    """SDMX monthly periods can be YYYY-Mmm or YYYY-mm."""
    match = re.fullmatch(r"(\d{4})-M?(0[1-9]|1[0-2])(?:-01)?", str(text))
    if not match:
        raise DataError(f"Unexpected monthly date in the IMF response: {text!s}")
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def resolve_attribute(definition, value):
    if value is None:
        return None
    values = definition.get("values", [])
    if isinstance(value, int) and values:
        if value < 0 or value >= len(values):
            raise DataError("Invalid attribute index in IMF data.")
        entry = values[value]
        return entry.get("id", entry.get("value"))
    return value


def parse_series(payload, expected):
    """Validate one exact SDMX series and decode its time dimension.

    Use keyPosition rather than assuming array order. Missing months remain
    missing, and observation statuses are retained for the exported data.
    """
    data = payload["data"]
    datasets, structures = data.get("dataSets", []), data.get("structures", [])
    if len(datasets) != 1 or not structures:
        raise DataError("IMF returned no data or more than one dataset.")
    dataset = datasets[0]
    try:
        structure = structures[dataset.get("structure", 0)]
    except (IndexError, TypeError) as exc:
        raise DataError("Invalid IMF structure reference.") from exc
    series = dataset.get("series", {})
    if len(series) == 0:
        raise DataError("No monthly observations are available for this IMF series.")
    if len(series) != 1:
        raise DataError("IMF returned multiple series; refusing to combine them.")
    key, record = next(iter(series.items()))
    dimensions = sorted(structure["dimensions"]["series"], key=lambda d: d["keyPosition"])
    try:
        indexes = [int(part) for part in key.split(":")]
        if len(indexes) != len(dimensions):
            raise DataError("Invalid IMF series key.")
        selected = {
            d["id"]: d["values"][i]["id"] for d, i in zip(dimensions, indexes)
        }
    except (ValueError, IndexError, KeyError) as exc:
        raise DataError("Cannot decode IMF series dimensions.") from exc
    if selected != expected:
        raise DataError("The downloaded series does not match the requested country and indicator.")
    observation_dimensions = structure["dimensions"]["observation"]
    if len(observation_dimensions) != 1 or observation_dimensions[0]["id"] != "TIME_PERIOD":
        raise DataError("Unsupported IMF observation dimensions.")
    times = observation_dimensions[0]["values"]
    attributes = structure.get("attributes", {})
    series_attributes = {
        d["id"]: resolve_attribute(d, value)
        for d, value in zip(attributes.get("series", []), record.get("attributes", []))
    }
    try:
        scale = float(series_attributes.get("SCALE") or 0)
    except (TypeError, ValueError) as exc:
        raise DataError("Unsupported IMF value scale.") from exc
    rows = []
    for index, obs in record.get("observations", {}).items():
        try:
            entry = times[int(index)]
            date = month_timestamp(entry.get("id", entry.get("value")))
            value = float(obs[0]) * 10 ** scale if obs[0] is not None else float("nan")
        except (IndexError, ValueError, TypeError, OverflowError) as exc:
            raise DataError("Cannot decode an IMF observation.") from exc
        if not math.isfinite(value) or value <= 0:
            continue
        attrs = {
            d["id"]: resolve_attribute(d, v)
            for d, v in zip(attributes.get("observation", []), obs[1:])
        }
        rows.append({"month": date, "value": value, "status": attrs.get("STATUS") or ""})
    if not rows:
        raise DataError("The IMF series contains no positive monthly values.")
    frame = pd.DataFrame(rows).set_index("month").sort_index()
    if frame.index.has_duplicates:
        raise DataError("Duplicate months in the IMF response.")
    now_month = pd.Timestamp(datetime.now(timezone.utc).date()).to_period("M").to_timestamp()
    frame = frame.loc[(frame.index >= pd.Timestamp(START_MONTH)) & (frame.index < now_month)]
    return frame, series_attributes


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_series(flow, country, api_key=""):
    if flow == "CPI":
        expected = {"COUNTRY": country, "INDEX_TYPE": "CPI", "COICOP_1999": "_T",
                    "TYPE_OF_TRANSFORMATION": "IX", "FREQUENCY": "M"}
        key = f"{country}.CPI._T.IX.M"
    else:
        expected = {"COUNTRY": country, "INDICATOR": "XDC_USD",
                    "TYPE_OF_TRANSFORMATION": "PA_RT", "FREQUENCY": "M"}
        key = f"{country}.XDC_USD.PA_RT.M"
    url = f"{API_BASE}/data/dataflow/IMF.STA/{flow}/+/{key}"
    frame, attributes = parse_series(request_json(url, api_key), expected)
    return {"frame": frame, "attributes": attributes, "url": url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def fx_country(country):
    return "G163" if country in EURO_ADOPTION else country


def estimate_us_october(frame):
    result = frame.copy()
    result["estimated"] = False
    oct_month, sep, nov = map(pd.Timestamp, ["2025-10-01", "2025-09-01", "2025-11-01"])
    observed = oct_month in result.index and pd.notna(result.loc[oct_month, "value"])
    if not observed and sep in result.index and nov in result.index:
        neighbors = result.loc[[sep, nov], "value"]
        if neighbors.notna().all() and (neighbors > 0).all():
            result.loc[oct_month, ["value", "status", "estimated"]] = [
                float(neighbors.mean()), "estimated by app", True]
    return result.sort_index()


def prepare_inputs(numerator, denominator, results, estimate_october=True):
    """Get common months without filling gaps or mixing old/new currencies."""
    cpis = {}
    for country in (numerator, denominator):
        frame = results[("CPI", country)]["frame"].copy()
        frame["estimated"] = False
        if country == "USA" and estimate_october:
            frame = estimate_us_october(frame)
        cpis[country] = frame
    parts = []
    for role, country in [("numerator", numerator), ("denominator", denominator)]:
        parts.append(cpis[country].rename(columns={
            "value": f"cpi_{role}", "status": f"cpi_{role}_status",
            "estimated": f"cpi_{role}_estimated"}))
        if country != "USA":
            parts.append(results[("ER", fx_country(country))]["frame"].rename(columns={
                "value": f"fx_{role}_per_usd", "status": f"fx_{role}_status"}))
    frame = pd.concat(parts, axis=1)
    for role, country in [("numerator", numerator), ("denominator", denominator)]:
        if country == "USA":
            frame[f"fx_{role}_per_usd"] = 1.0  # Definition of USD/USD, not estimated data.
            frame[f"fx_{role}_status"] = "currency identity"
    start = max([START_MONTH] + [EURO_ADOPTION[c] for c in (numerator, denominator) if c in EURO_ADOPTION])
    frame = frame.loc[frame.index >= pd.Timestamp(start)].sort_index()
    required = ["cpi_numerator", "cpi_denominator", "fx_numerator_per_usd", "fx_denominator_per_usd"]
    keep = frame[required].notna().all(axis=1) & (frame[required] > 0).all(axis=1)
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise DataError("These countries have no common valid monthly sample from January 1999 onward.")
    # Both q rates are local-currency units per USD. Their ratio is
    # denominator-currency units per numerator-currency unit.
    frame["S_denominator_per_numerator"] = frame["fx_denominator_per_usd"] / frame["fx_numerator_per_usd"]
    frame["RER"] = frame["S_denominator_per_numerator"] * frame["cpi_numerator"] / frame["cpi_denominator"]
    return frame


def calculate_indexes(frame, base_month):
    """Always rebase the full common sample before applying chart dates."""
    base = pd.Timestamp(base_month)
    if base not in frame.index:
        raise DataError("Choose a base month with all required observations.")
    result = frame.copy()
    result["real_index"] = 100 * (result["RER"] / result.loc[base, "RER"])
    result["nominal_index"] = 100 * (result["S_denominator_per_numerator"] / result.loc[base, "S_denominator_per_numerator"])
    return result


def make_figure(frame, start, end, numerator_name, denominator_name, base_month):
    # Reindex only the display frame so Plotly leaves gaps at missing months.
    display = frame.loc[pd.Timestamp(start):pd.Timestamp(end)].reindex(
        pd.date_range(start, end, freq="MS"))
    fig = go.Figure()
    for column, label, color in [("real_index", "Real exchange rate", "#b32232"),
                                 ("nominal_index", "Nominal exchange rate", "#175e9d")]:
        fig.add_trace(go.Scatter(x=display.index, y=display[column], name=label,
                                mode="lines", line={"color": color, "width": 2.5},
                                connectgaps=False, hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=100, line_color="#888888", line_dash="dash")
    fig.update_layout(template="plotly_white", height=510,
                      title={"text": f"{numerator_name} relative to {denominator_name}", "font": {"color": "#b32232", "size": 22}},
                      yaxis_title=f"Index ({pd.Timestamp(base_month):%b %Y} = 100)",
                      xaxis_title=None, hovermode="x unified", legend={"orientation": "h", "y": 1.12},
                      margin={"l": 55, "r": 20, "t": 105, "b": 50})
    return fig


def render_graph(frame, numerator, denominator, source_label):
    numerator_name, denominator_name = COUNTRY_NAMES[numerator], COUNTRY_NAMES[denominator]
    st.subheader(f"{numerator_name} / {denominator_name}")
    st.latex(r"\mathrm{RER}_t=S_t\frac{P_t^{\mathrm{numerator}}}{P_t^{\mathrm{denominator}}}")
    st.write(f"S is {denominator_name}'s currency units per unit of {numerator_name}'s currency. "
             f"A rise means **real appreciation of {numerator_name} relative to {denominator_name}**.")
    for country in (numerator, denominator):
        if country in EURO_ADOPTION:
            st.caption(f"{COUNTRY_NAMES[country]}: euro exchange rate from {EURO_ADOPTION[country]}; "
                       "months before euro adoption are excluded.")
    months = [date.strftime("%Y-%m") for date in frame.index]
    pair = f"{numerator}_{denominator}_{source_label}"
    base_month = st.selectbox("Base month (index = 100)", months, index=0, key=f"base_{pair}")
    indexed = calculate_indexes(frame, base_month)
    if len(months) > 1:
        start, end = st.select_slider("Months shown on the graph", months,
                                      value=(months[0], months[-1]), key=f"range_{pair}")
    else:
        start = end = months[0]
    st.plotly_chart(make_figure(indexed, start, end, numerator_name, denominator_name, base_month),
                    width="stretch")
    estimated = indexed["cpi_numerator_estimated"].any() or indexed["cpi_denominator_estimated"].any()
    if estimated:
        st.info("October 2025 U.S. CPI is estimated as the average of September and November. "
                "The estimate is flagged in the table and CSV.")
    grid = pd.date_range(indexed.index.min(), indexed.index.max(), freq="MS")
    st.caption(f"Common sample: {months[0]} to {months[-1]} · {len(indexed):,} valid months · "
               f"{len(grid) - len(indexed):,} missing common months · {source_label}.")
    st.caption("An index above 100 means appreciation since the base month; it does not establish overvaluation. "
               "Countries' CPI baskets differ. Data are current revised observations, subject to IMF revisions.")
    with st.expander("Calculation and source details"):
        st.write("The IMF CPI series is the monthly national all-items index (CPI._T.IX.M). "
                 "Exchange rates are monthly period averages in local currency per U.S. dollar (XDC_USD.PA_RT.M). "
                 "S = (denominator currency per dollar) / (numerator currency per dollar). "
                 "For USD, the rate is exactly 1 by definition. CPI reference years may differ: rebasing "
                 "the final index removes their constant scale factors. The ratio of two monthly-average "
                 "USD rates is an approximation to the monthly average of the daily bilateral rate. "
                 "Other currency redenominations or methodological breaks may require extra treatment.")
        st.markdown("[IMF CPI](https://data.imf.org/en/datasets/IMF.STA:CPI) · "
                    "[IMF exchange rates](https://data.imf.org/Datasets/ER)")
    exported = indexed.reset_index(names="month")
    exported["month"] = exported["month"].dt.strftime("%Y-%m")
    exported["numerator_country"] = numerator
    exported["denominator_country"] = denominator
    exported["base_month"] = base_month
    st.download_button("Download data (CSV)", exported.to_csv(index=False).encode("utf-8"),
                       file_name=f"rer_{numerator}_{denominator}.csv", mime="text/csv", key=f"export_{pair}")
    with st.expander("Data table"):
        st.dataframe(exported, width="stretch", hide_index=True)


def read_saved_export(file):
    """Only accept the app's own fully identified export for offline plotting."""
    frame = pd.read_csv(file, keep_default_na=False)
    required = {"month", "numerator_country", "denominator_country", "cpi_numerator",
                "cpi_denominator", "fx_numerator_per_usd", "fx_denominator_per_usd",
                "cpi_numerator_estimated", "cpi_denominator_estimated", "RER", "S_denominator_per_numerator"}
    if frame.empty or not required.issubset(frame.columns):
        raise DataError("Upload a CSV previously exported by this app.")
    countries = []
    for column in ["numerator_country", "denominator_country"]:
        values = frame[column].unique()
        if len(values) != 1 or values[0] not in COUNTRY_NAMES:
            raise DataError("Invalid or mixed country identifiers in the saved CSV.")
        countries.append(values[0])
    if countries[0] == countries[1]:
        raise DataError("Choose two different countries.")
    frame["month"] = frame["month"].map(month_timestamp)
    frame = frame.set_index("month").sort_index()
    if frame.index.has_duplicates:
        raise DataError("Duplicate months in saved CSV.")
    for col in ["cpi_numerator", "cpi_denominator", "fx_numerator_per_usd", "fx_denominator_per_usd"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any() or not frame[col].map(math.isfinite).all() or (frame[col] <= 0).any():
            raise DataError("Invalid input values in saved CSV.")
    for col in ["cpi_numerator_estimated", "cpi_denominator_estimated"]:
        labels = frame[col].astype(str).str.lower()
        if not labels.isin(["true", "false"]).all():
            raise DataError("Invalid estimate flags in saved CSV.")
        frame[col] = labels.eq("true")
    # Recalculate from raw inputs instead of trusting stored indexes.
    frame["S_denominator_per_numerator"] = frame["fx_denominator_per_usd"] / frame["fx_numerator_per_usd"]
    frame["RER"] = frame["S_denominator_per_numerator"] * frame["cpi_numerator"] / frame["cpi_denominator"]
    cutoff = pd.Timestamp(datetime.now(timezone.utc).date()).to_period("M").to_timestamp()
    start = max([START_MONTH] + [EURO_ADOPTION[c] for c in countries if c in EURO_ADOPTION])
    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < cutoff)]
    if frame.empty:
        raise DataError("The CSV has no valid completed months in this app's sample.")
    return frame, countries[0], countries[1]


def main():
    st.set_page_config(page_title="Two-country real exchange rate", layout="wide")
    st.title("Two-country real exchange rate")
    st.write("Choose the two countries in the CPI ratio. The app downloads monthly IMF data and plots real and nominal exchange rate indexes.")
    mode = st.radio("Data source", ["Live IMF data", "Saved CSV export"], horizontal=True)
    if mode == "Saved CSV export":
        uploaded = st.file_uploader("CSV previously downloaded from this app", type="csv")
        if uploaded is not None:
            try:
                frame, numerator, denominator = read_saved_export(uploaded)
                st.caption("Source: uploaded saved export; this is not a live update.")
                render_graph(frame, numerator, denominator, "uploaded saved export")
            except (DataError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                st.error(str(exc))
        return
    options = sorted(COUNTRY_NAMES, key=lambda c: COUNTRY_NAMES[c])
    with st.form("countries"):
        left, right = st.columns(2)
        numerator = left.selectbox("Country in the numerator (P numerator)", options,
                                   index=options.index("GBR"), format_func=lambda c: COUNTRY_NAMES[c])
        denominator = right.selectbox("Country in the denominator (P denominator)", options,
                                       index=options.index("USA"), format_func=lambda c: COUNTRY_NAMES[c])
        estimate_october = st.checkbox("Estimate missing October 2025 U.S. CPI using September and November", value=True)
        submitted = st.form_submit_button("Fetch data and plot", type="primary")
    if submitted:
        st.session_state["active_pair"] = (numerator, denominator, estimate_october)
    if st.button("Refresh data"):
        fetch_series.clear()
    if "active_pair" in st.session_state:
        numerator, denominator, estimate_october = st.session_state["active_pair"]
        if numerator == denominator:
            st.error("Please choose two different countries.")
        else:
            api_key = get_api_key()
            keys = {( "CPI", numerator), ("CPI", denominator)}
            keys |= {("ER", fx_country(c)) for c in (numerator, denominator) if c != "USA"}
            results, errors = {}, []
            with st.spinner("Downloading monthly IMF data…"):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(fetch_series, flow, country, api_key): (flow, country)
                               for flow, country in sorted(keys)}
                    for future in as_completed(futures):
                        flow, country = futures[future]
                        try:
                            results[(flow, country)] = future.result()
                        except Exception as exc:
                            message = str(exc) if isinstance(exc, DataError) else "Unexpected response; try again later."
                            errors.append(f"{COUNTRY_NAMES.get(country, 'Euro area')} {flow}: {message}")
            if errors:
                for error in errors:
                    st.error(error)
                st.info("Availability varies by country. Switch Data source to Saved CSV export to reload a previous download.")
            else:
                try:
                    frame = prepare_inputs(numerator, denominator, results, estimate_october)
                    render_graph(frame, numerator, denominator, "IMF live downloads (cached for one hour)")
                    with st.expander("Download times and exact series"):
                        for key in sorted(results):
                            item = results[key]
                            st.write(f"{key[0]} / {key[1]} · downloaded {item['downloaded_at']} (UTC)")
                            st.code(item["url"], language=None)
                except DataError as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    main()
