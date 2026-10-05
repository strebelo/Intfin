"""Python 3.12: monthly GBP real exchange rate, with auditable inputs."""
import csv
import io
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TE = "https://tradingeconomics.com/united-kingdom/consumer-price-index-cpi"
ONS = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23"
ONS_CSV = "https://www.ons.gov.uk/generator?format=csv&uri=" + ONS.split(".uk")[1]
IDS = {"S": "EXUSUK", "US": "CPIAUCNS", "UK": "GBRCPIALLMINMEI"}
MONTHS = dict(zip("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), range(1, 13)))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@st.cache_data(ttl=3600, show_spinner=False)
def download(url, series_id=None):
    """Cache successes and failures; Refresh data clears both."""
    try:
        response = requests.get(url, params={"id": series_id} if series_id else None,
                                timeout=30, headers={"User-Agent": "GBP-RER-Teaching-App/1.0"})
        response.raise_for_status()
        return response.content.decode("utf-8-sig"), utc_now(), None
    except (requests.RequestException, UnicodeError) as exc:
        return None, utc_now(), str(exc)


def monthly_series(rows):
    """Strict dates/numbers. Only explicit annual and quarterly rows are skipped."""
    parsed = []
    for date_text, value_text in rows:
        date_text = str(date_text).strip().upper()
        if re.fullmatch(r"\d{4}(?: Q[1-4])?", date_text):
            continue
        if re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", date_text):
            date = datetime.fromisoformat(date_text + "-01" if len(date_text) == 7 else date_text)
        elif re.fullmatch(r"\d{4} [A-Z]{3}", date_text):
            year, month = date_text.split()
            if month not in MONTHS:
                raise ValueError(f"Invalid month: {date_text}")
            date = datetime(int(year), MONTHS[month], 1)
        else:
            raise ValueError(f"Invalid date: {date_text!r}. Use YYYY-MM-DD, YYYY-MM or YYYY MON.")
        text = str(value_text).strip()
        value = float("nan") if text in ("", ".") else float(text)
        if not math.isnan(value) and not math.isfinite(value):
            raise ValueError(f"Nonfinite numeric value at {date_text}")
        if text.lower() in ("nan", "+nan", "-nan"):
            raise ValueError(f"Use a blank or '.' for missing values at {date_text}")
        parsed.append((pd.Timestamp(date).to_period("M").to_timestamp(), value))
    if not parsed:
        raise ValueError("No monthly observations found.")
    frame = pd.DataFrame(parsed, columns=["month", "value"])
    if frame["month"].duplicated().any():
        raise ValueError("Duplicate months found; upload monthly data, not daily data.")
    series = frame.set_index("month")["value"].sort_index()
    if series.dropna().empty:
        raise ValueError("No numeric monthly observations found.")
    return series


def parse_csv(text, series_id=None):
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("\ufeff")))
            if r and any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("Empty CSV.")
    header = [cell.strip().lower() for cell in rows[0]]
    if series_id:
        date_col = next((i for i, name in enumerate(header) if name in ("date", "observation_date")), None)
        if date_col is None or series_id.lower() not in header:
            raise ValueError(f"Expected DATE/observation_date and {series_id} columns from FRED.")
        value_col = header.index(series_id.lower())
        data = rows[1:]
    elif header[0] == "title":
        metadata = {r[0].strip().upper(): " ".join(r[1:]).strip().upper() for r in rows if r}
        if metadata.get("CDID") != "D7BT" or "CPI INDEX 00: ALL ITEMS 2015=100" not in metadata.get("TITLE", ""):
            raise ValueError("This is not the ONS D7BT all-items CPI (2015=100) CSV.")
        date_col, value_col = 0, 1
        data = [r for r in rows if r and re.match(r"^\d{4}", r[0].strip())]
    else:
        if "date" not in header or "cpi" not in header:
            raise ValueError("Supplementary CSV needs date,cpi columns, or the original ONS D7BT CSV.")
        date_col, value_col = header.index("date"), header.index("cpi")
        data = rows[1:]
    pairs = []
    for row in data:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) <= max(date_col, value_col):
            raise ValueError("CSV row has missing columns.")
        pairs.append((row[date_col], row[value_col]))
    return monthly_series(pairs)


class Tables(HTMLParser):
    """Read HTML table cells using only the standard library."""
    def __init__(self, text):
        super().__init__()
        self.tables, self.table, self.row, self.cell = [], None, None, None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def load_supplement(current_month):
    """Accept explicit historical actuals only; otherwise use official ONS D7BT."""
    notes = []
    text, retrieved, error = download(TE)
    if text:
        # TE's summary/previous/forecast boxes are deliberately never parsed.
        if re.search(r"2015\s*=\s*100\s*,?\s*NSA", text, re.I):
            for table in Tables(text).tables:
                if not table or [s.lower() for s in table[0]] != ["date", "actual"]:
                    continue
                try:
                    series = monthly_series(table[1:])
                    series = series.loc[series.index < current_month].dropna()
                    expected = pd.date_range("1999-01-01", series.index.max(), freq="MS")
                    if not series.empty and len(expected) > 12 and series.reindex(expected).notna().all():
                        return series, {"source": "Trading Economics: actual all-items CPI, 2015=100, NSA",
                                        "url": TE, "retrieved": retrieved}, notes
                except (ValueError, OverflowError):
                    continue
    notes.append("Trading Economics: " + (error or "full dated actual monthly history unavailable; using ONS D7BT."))
    text, retrieved, error = download(ONS_CSV)
    if text:
        try:
            return parse_csv(text), {"source": "ONS D7BT: all-items CPI, 2015=100, NSA",
                                    "url": ONS, "retrieved": retrieved}, notes
        except (ValueError, OverflowError) as exc:
            error = str(exc)
    notes.append("ONS CSV: " + str(error))
    text, retrieved, error = download(ONS)
    if text and "CPI INDEX 00: ALL ITEMS 2015=100" in text.upper() and "D7BT" in text.upper():
        for table in Tables(text).tables:
            if table and [s.lower() for s in table[0]] == ["period", "value"]:
                try:
                    return monthly_series(table[1:]), {"source": "ONS D7BT: all-items CPI, 2015=100, NSA",
                                                       "url": ONS, "retrieved": retrieved}, notes
                except (ValueError, OverflowError) as exc:
                    error = str(exc)
    notes.append("ONS monthly table: " + (error or "verified monthly table unavailable."))
    return None, None, notes


def calculate(fred, supplement, current_month):
    """Splice, clean, and normalize the full sample BEFORE any chart filtering."""
    grid = pd.date_range("1999-01-01", current_month - pd.offsets.MonthBegin(1), freq="MS")
    if grid.empty:
        raise ValueError("There are no completed months since January 1999.")
    audit = pd.DataFrame(index=grid)
    for key, series in fred.items():
        audit[key + "_FRED"] = series.reindex(grid)
    audit["UK_supplement"] = supplement.reindex(grid) if supplement is not None else float("nan")
    audit["UK"] = audit["UK_FRED"]
    audit["UK_source"] = "missing"
    audit.loc[audit["UK"].notna(), "UK_source"] = "FRED GBRCPIALLMINMEI"
    splice = {"month": None, "factor": None, "error": None, "count": 0}
    if supplement is not None:
        overlap = audit.loc[(audit["UK_FRED"] > 0) & (audit["UK_supplement"] > 0)]
        if overlap.empty:
            splice["error"] = "No positive overlapping month: supplementary CPI was not used."
        else:
            m = overlap.index.max()
            factor = overlap.at[m, "UK_FRED"] / overlap.at[m, "UK_supplement"]
            # A definition check uses the fixed source IDs and upload confirmation.
            # This additional numerical check detects inconsistent histories.
            ratios = overlap.tail(12)["UK_FRED"] / overlap.tail(12)["UK_supplement"]
            if ((ratios / factor - 1).abs() > 0.01).any():
                splice["error"] = "Overlapping CPI levels differ by more than 1% after linking; extension rejected."
            else:
                use = (audit.index > m) & audit["UK_FRED"].isna() & (audit["UK_supplement"] > 0)
                audit.loc[use, "UK"] = factor * audit.loc[use, "UK_supplement"]
                audit.loc[use, "UK_source"] = "supplement linked to FRED"
                splice.update(month=m, factor=factor, count=int(use.sum()))
    audit["US"] = audit["US_FRED"]
    audit["US_estimated"] = False
    oct_2025 = pd.Timestamp("2025-10-01")
    neighbors = audit["US_FRED"].reindex(pd.to_datetime(["2025-09-01", "2025-11-01"]))
    if oct_2025 in audit.index and pd.isna(audit.at[oct_2025, "US_FRED"]) and (neighbors > 0).all():
        audit.at[oct_2025, "US"] = neighbors.mean()
        audit.at[oct_2025, "US_estimated"] = True
    audit["included"] = (audit[["S_FRED", "US", "UK"]] > 0).all(axis=1)
    valid = audit.loc[audit["included"]].copy()
    if valid.empty:
        raise ValueError("No positive common observations in completed months since January 1999.")
    raw = valid["S_FRED"] * valid["UK"] / valid["US"]
    if not raw.map(math.isfinite).all():
        raise ValueError("RER overflow: check the input levels.")
    rer_mean, s_mean = float(raw.mean()), float(valid["S_FRED"].mean())
    audit["RER_raw"] = raw
    audit["RER_mean"] = rer_mean
    audit["S_mean"] = s_mean
    audit["RER_index"] = 100 * audit["RER_raw"] / rer_mean
    audit["nominal_index"] = (100 * valid["S_FRED"] / s_mean).reindex(grid)
    for name in ("RER_index", "nominal_index"):
        if not math.isclose(audit.loc[audit["included"], name].mean(), 100, abs_tol=1e-9):
            raise ValueError("Normalization check failed.")
    return audit, splice


def chart(frame):
    """Synthetic crossing points are used only in line traces, never in the data."""
    paths = {"red": ([], []), "blue": ([], [])}
    points = list(frame["RER_index"].items())
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if pd.isna(y0) or pd.isna(y1):
            continue  # Do not draw through missing months.
        segments = [(x0, y0, x1, y1)]
        if (y0 - 100) * (y1 - 100) < 0:
            crossing = x0 + (x1 - x0) * ((100 - y0) / (y1 - y0))
            segments = [(x0, y0, crossing, 100), (crossing, 100, x1, y1)]
        for a, b, c, d in segments:
            color = "red" if (b + d) / 2 > 100 else "blue"
            paths[color][0].extend([a, c, None])
            paths[color][1].extend([b, d, None])
    fig = go.Figure()
    for color, (xs, ys) in paths.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                name="RER above average" if color == "red" else "RER below average",
                                line=dict(color=color, width=3), connectgaps=False))
    observed = frame.dropna(subset=["RER_index"])
    fig.add_trace(go.Scatter(x=observed.index, y=observed["RER_index"], mode="markers",
                            showlegend=False, name="RER", marker=dict(size=4, color=[
                                "red" if y > 100 else "blue" for y in observed["RER_index"]]),
                            hovertemplate="%{x|%b %Y}<br>RER: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["nominal_index"], mode="lines+markers",
                            name="Nominal exchange rate", line=dict(color="gray", dash="dash", width=2),
                            marker=dict(size=3), connectgaps=False,
                            hovertemplate="%{x|%b %Y}<br>Nominal: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=100, line_color="#777777", line_width=1,
                  annotation_text="Sample average", annotation_position="top left")
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#222222"), height=500, hovermode="closest",
                      xaxis_title="Month", yaxis_title="Index (full-sample average = 100)",
                      legend=dict(orientation="h", y=1.12), margin=dict(t=70))
    return fig


def main():
    st.set_page_config(page_title="Real exchange rate of the British pound", layout="wide")
    st.title("Real exchange rate of the British pound")
    st.latex(r"\mathrm{RER}_t=S_t\frac{P_t^{UK}}{P_t^{US}},\qquad I_t^{RER}=100\frac{\mathrm{RER}_t}{\overline{\mathrm{RER}}},\qquad I_t^S=100\frac{S_t}{\overline S}")
    st.caption("S is U.S. dollars per British pound, measured as a monthly average.")
    current = pd.Timestamp(datetime.now(ZoneInfo("America/Chicago")).date()).to_period("M").to_timestamp()
    if st.sidebar.button("Refresh data"):
        download.clear()
    with st.sidebar.expander("CSV uploads if downloads fail"):
        st.write("Use original FRED CSVs, one per series. Uploads are used only if that download fails.")
        uploads = {key: st.file_uploader("FRED " + sid, type="csv", key=sid) for key, sid in IDS.items()}
        source = st.selectbox("Supplementary UK CPI source", ["ONS D7BT", "Trading Economics"])
        upload = st.file_uploader("Supplementary UK CPI CSV", type="csv", key="supplement")
        confirmed = st.checkbox("I verified these are actual monthly UK all-items CPI levels, 2015=100, NSA.")
        st.caption("Supplementary CSV: date,cpi; or the original ONS D7BT download. Include an overlap with FRED.")

    fred, metadata, notices = {}, {}, []
    with st.spinner("Loading monthly source data..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda sid: download(FRED, sid), IDS.values()))
        for (key, sid), (text, retrieved, error) in zip(IDS.items(), results):
            try:
                if error:
                    raise ValueError(error)
                series = parse_csv(text, sid)
                meta = {"source": "FRED " + sid, "url": f"https://fred.stlouisfed.org/series/{sid}", "retrieved": retrieved}
                st.session_state["verified_" + sid] = (series, meta)
            except (ValueError, OverflowError, TypeError) as exc:
                notices.append(f"{sid} download unavailable: {exc}")
                saved = st.session_state.get("verified_" + sid)
                if uploads[key] is not None:
                    try:
                        series = parse_csv(uploads[key].getvalue().decode("utf-8-sig"), sid)
                        meta = {"source": "FRED CSV upload: " + sid, "url": f"https://fred.stlouisfed.org/series/{sid}", "retrieved": utc_now()}
                        st.session_state["verified_" + sid] = (series, meta)
                    except (ValueError, UnicodeError, OverflowError) as upload_error:
                        notices.append(f"{sid} upload rejected: {upload_error}")
                        if saved is None:
                            continue
                        series, meta = saved
                elif saved is not None:
                    series, meta = saved
                    notices.append(f"Using the last verified {sid} data from this session.")
                else:
                    continue
            fred[key], metadata[key] = series, meta
        supplement, supplement_meta, supplement_notes = load_supplement(current)
        notices.extend(supplement_notes)
        if supplement is None and upload is not None:
            try:
                if not confirmed:
                    raise ValueError("Confirm the supplementary CPI definition before using this upload.")
                supplement = parse_csv(upload.getvalue().decode("utf-8-sig"))
                supplement_meta = {"source": source + " CSV upload: user-confirmed actual all-items CPI, 2015=100, NSA",
                                   "url": ONS if source == "ONS D7BT" else TE, "retrieved": utc_now()}
            except (ValueError, UnicodeError, OverflowError) as exc:
                supplement = None
                notices.append(f"Supplementary upload rejected: {exc}")
    with st.expander("Source availability and validation messages"):
        for notice in notices:
            st.write(notice)
    if len(fred) != 3:
        st.error("All three FRED series are needed. Upload the missing original CSVs in the sidebar, then retry.")
        st.stop()
    try:
        audit, splice = calculate(fred, supplement, current)
    except (ValueError, OverflowError) as exc:
        st.error(str(exc))
        st.stop()
    if splice["error"]:
        st.warning(splice["error"])
    valid = audit.loc[audit["included"]]
    first, last = valid.index.min(), valid.index.max()
    st.write(f"**Benchmark:** {first:%b %Y}–{last:%b %Y}, {len(valid)} valid common months. "
             f"**Latest common month:** {last:%b %Y}. Both full-sample indexes average 100.")
    if supplement is None or splice["error"]:
        st.info(f"No usable UK CPI extension. The verified FRED sample ends in {last:%b %Y}; no observations were invented.")
    if splice["month"] is not None:
        st.write(f"**UK CPI link:** {splice['month']:%b %Y}; factor = {splice['factor']:.10g}; "
                 f"{splice['count']} later monthly UK CPI values added. Source: {supplement_meta['source']}.")
        st.latex(r"P_t^{UK}=P_m^{UK,\mathrm{FRED}}\left(P_t^{UK,\mathrm{new}}/P_m^{UK,\mathrm{new}}\right),\quad t>m")
    st.caption("UK definitions: FRED GBRCPIALLMINMEI is OECD total/all-items CPI, 2015=100, NSA; "
               "ONS D7BT is CPI INDEX 00: ALL ITEMS 2015=100. Trading Economics identifies its matching index as 2015=100, NSA, sourced from ONS.")
    if audit["US_estimated"].any():
        st.warning("October 2025 U.S. CPI is estimated as the average of observed September and November 2025 levels.")
    source_rows = [{"input": key, **meta} for key, meta in metadata.items()]
    if supplement_meta:
        source_rows.append({"input": "UK supplement", **supplement_meta})
    st.dataframe(pd.DataFrame(source_rows), hide_index=True)

    # Date selection affects only the chart, never the benchmark or exports.
    start = st.sidebar.date_input("Chart starts", first.date(), min_value=first.date(), max_value=last.date())
    end = st.sidebar.date_input("Chart ends", last.date(), min_value=first.date(), max_value=last.date())
    if start > end:
        st.error("Chart start must be on or before chart end.")
    else:
        display = audit.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if display["included"].any():
            st.plotly_chart(chart(display), width="stretch", theme=None)
        else:
            st.warning("No valid common observations in these chart dates.")
    st.write("A rising RER indicates real appreciation: the UK CPI basket becomes more expensive "
             "relative to the U.S. basket in dollar terms. A rising nominal index means more dollars per pound. "
             "The CPI baskets differ across countries, and CPI indexes measure price changes rather than "
             "comparable basket price levels. The sample average is a historical reference, not an equilibrium "
             "exchange rate or a measure of fair value. Changing the chart dates leaves both indexes unchanged; "
             "new data or revisions can change the full-sample benchmark.")
    audit["US_source"] = "missing"
    audit.loc[audit["US_FRED"].notna(), "US_source"] = "FRED CPIAUCNS"
    audit.loc[audit["US_estimated"], "US_source"] = "estimated: mean of Sep and Nov 2025 FRED CPIAUCNS"
    audit["FX_source"] = audit["S_FRED"].map(lambda v: "FRED EXUSUK monthly average" if pd.notna(v) else "missing")
    for key, meta in metadata.items():
        for field, value in meta.items():
            audit[key + "_" + field + "_record"] = value
    for field in ("source", "url", "retrieved"):
        audit["supplement_" + field] = supplement_meta[field] if supplement_meta else "unavailable"
    audit["link_month"] = splice["month"]
    audit["link_factor"] = splice["factor"]
    audit["benchmark_start"], audit["benchmark_end"] = first, last
    audit["benchmark_n"] = len(valid)
    audit["splice_status"] = splice["error"] or ("linked" if splice["month"] is not None else "no supplement")
    output = audit.rename_axis("month").reset_index()
    st.caption("Full audit table: only rows with included=True enter the calculation. Other gaps remain blank. "
               "Original FRED levels, supplementary levels, source records, estimates, means and indexes are exported.")
    st.dataframe(output, hide_index=True)
    st.download_button("Download full data and calculations (CSV)", output.to_csv(index=False).encode("utf-8"),
                       "real_exchange_rate_gbp.csv", "text/csv")


if __name__ == "__main__":
    main()
