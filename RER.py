# Real exchange rate of the British pound

## Setup (Python 3.12)

Save all three files in one folder. In a terminal there, run:

    python -m venv .venv

Activate with `source .venv/bin/activate` (macOS/Linux) or
`.venv\Scripts\activate` (Windows), then run:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

No credentials or API keys are needed. Downloads use 30-second timeouts and
one-hour caching. **Refresh data** clears the download cache.

## Calculation

FRED provides EXUSUK (monthly-average dollars per pound), CPIAUCNS (U.S.
all-items CPI), and GBRCPIALLMINMEI (UK all-items CPI). Both CPIs are not
seasonally adjusted. The UK CPI index is 2015=100.

Trading Economics is checked for complete dated actual monthly history;
otherwise the app uses official ONS D7BT CSV/table data. It never extracts
forecasts, graph values, inflation rates, CPIH or RPI.

At the latest positive overlap m, later UK CPI is linked using:

    UK_CPI(t) = FRED_UK_CPI(m) * new_UK_CPI(t) / new_UK_CPI(m)
    RER(t) = dollars_per_pound(t) * UK_CPI(t) / US_CPI(t)
    RER_index(t) = 100 * RER(t) / mean(monthly RER)
    nominal_index(t) = 100 * dollars_per_pound(t) / mean(dollars_per_pound)

FRED values are preserved. An overlap is required; the last 12 overlap ratios
must agree within 1% of the link factor. Rejected extensions retain FRED data.
Only positive common observations from January 1999 in completed months enter
both means (America/Chicago calendar). Both indexes average 100. Chart dates
are applied afterward. New data/revisions may change the full-sample means.

Only missing October 2025 U.S. CPI can be estimated, using observed positive
September and November levels; the estimate is flagged. No other gaps are
filled. Red/blue RER segments meet at interpolated crossings of 100; those
 drawing points never enter the data. Missing months break lines.

A higher RER indicates real appreciation. National CPI baskets differ and
their index levels are arbitrary: the sample mean is a historical reference,
not fair value or an equilibrium exchange rate.

## Failed downloads and exports

Use the sidebar uploads when downloads fail. FRED CSVs need `DATE` or
`observation_date` and the exact series-ID column. Supplementary CSVs need
`date,cpi`, or the original ONS D7BT format. Dates: `YYYY-MM-DD`, `YYYY-MM`,
or `YYYY MON`. Annual/quarterly rows are discarded; blanks and `.` are missing.
Invalid dates, nonnumeric values and duplicate months are rejected.

Label supplementary uploads ONS D7BT or Trading Economics, confirm actual
all-items CPI levels (2015=100, NSA), and include an overlapping FRED month.
The app records that user-confirmed provenance; it cannot authenticate uploads.
Last verified FRED data are retained within the session, until server restart.

The full table/CSV exports original inputs, supplementary/linked CPI, raw RER,
means, indexes, flags, splice details, source links and retrieval/acceptance
times. Only rows marked `included=True` form the calculated sample.

Sources:

- https://fred.stlouisfed.org/series/EXUSUK
- https://fred.stlouisfed.org/series/CPIAUCNS
- https://fred.stlouisfed.org/series/GBRCPIALLMINMEI
- https://tradingeconomics.com/united-kingdom/consumer-price-index-cpi
- https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23
