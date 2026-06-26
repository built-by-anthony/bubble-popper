import os
import time
from datetime import date

import psycopg
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
EDGAR_USER_AGENT = os.environ["EDGAR_USER_AGENT"]

COMPANIES = [
    {"ticker": "msft",  "cik": "CIK0000789019"},
    {"ticker": "googl", "cik": "CIK0001652044"},
    {"ticker": "amzn",  "cik": "CIK0001018724"},
    {"ticker": "meta",  "cik": "CIK0001326801"},
    {"ticker": "nvda",  "cik": "CIK0001045810"},
]

# Annual 10-K only — audited, no YTD subtraction needed
FLOW_CONCEPTS = {
    "revenue":    ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "net_income": ["NetIncomeLoss"],
}

# Point-in-time — safe to pull from 10-K and 10-Q
BALANCE_CONCEPTS = {
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "total_assets":   ["Assets"],
}

# yfinance row labels that map to our concepts
YFINANCE_FLOW_MAP = {
    "revenue":    "Total Revenue",
    "rd_expense": "Research And Development",
    "net_income": "Net Income",
}

YFINANCE_BALANCE_MAP = {
    "long_term_debt": "Long Term Debt",
    "total_assets":   "Total Assets",
}

UPSERT_SQL = """
    INSERT INTO fact_observation (metric_id, obs_date, raw_value, source, series_id, valid_as_of)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (metric_id, obs_date)
    DO UPDATE SET raw_value   = EXCLUDED.raw_value,
                  series_id   = EXCLUDED.series_id,
                  valid_as_of = EXCLUDED.valid_as_of
    WHERE fact_observation.valid_as_of < EXCLUDED.valid_as_of
"""


def fetch_company_facts(cik: str, session: requests.Session) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/{cik}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_annual(us_gaap: dict, tags: list[str]) -> dict[str, float]:
    """Annual 10-K values only — merges tags, deduplicates by end date."""
    by_end: dict[str, dict] = {}
    for tag in tags:
        for o in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
            if o.get("form") != "10-K":
                continue
            end = o["end"]
            if end not in by_end or o["filed"] > by_end[end]["filed"]:
                by_end[end] = o
    return {end: o["val"] for end, o in by_end.items()}


def extract_balance(us_gaap: dict, tags: list[str]) -> dict[str, float]:
    """Quarterly and annual balance sheet values — merges tags, deduplicates by end date."""
    by_end: dict[str, dict] = {}
    for tag in tags:
        for o in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
            if o.get("form") not in ("10-K", "10-Q"):
                continue
            end = o["end"]
            if end not in by_end or o["filed"] > by_end[end]["filed"]:
                by_end[end] = o
    return {end: o["val"] for end, o in by_end.items()}


def fetch_yfinance_fallback(ticker: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Annual yfinance data as fallback when EDGAR tags are missing."""
    t = yf.Ticker(ticker)
    flow, balance = {}, {}

    income = t.income_stmt
    bs = t.balance_sheet

    for name, label in YFINANCE_FLOW_MAP.items():
        if label in income.index:
            flow[name] = {
                col.date().isoformat(): val
                for col, val in income.loc[label].items()
                if val == val and val is not None  # drop NaN
            }

    for name, label in YFINANCE_BALANCE_MAP.items():
        if label in bs.index:
            balance[name] = {
                col.date().isoformat(): val
                for col, val in bs.loc[label].items()
                if val == val and val is not None
            }

    return flow, balance


def merge_with_fallback(
    edgar: dict[str, float],
    fallback: dict[str, float],
) -> dict[str, float]:
    """Use EDGAR values where available; fill gaps with yfinance."""
    merged = dict(fallback)
    merged.update(edgar)
    return merged


def compute_derived(
    flow: dict[str, dict[str, float]],
    balance: dict[str, dict[str, float]],
    ticker: str,
    source: str,
) -> list[tuple]:
    rows = []
    valid_as_of = date.today()

    revenue = flow.get("revenue", {})
    rd = flow.get("rd_expense", {})
    net_income = flow.get("net_income", {})
    long_term_debt = balance.get("long_term_debt", {})
    total_assets = balance.get("total_assets", {})

    for end_date, rev in revenue.items():
        if rev <= 0:
            continue
        if end_date in rd:
            val = rd[end_date] / rev * 100
            if 0 <= val <= 200:
                rows.append((f"{ticker}_rd_to_revenue", end_date, val, source, ticker.upper(), valid_as_of))
        if end_date in net_income:
            val = net_income[end_date] / rev * 100
            if -100 <= val <= 100:
                rows.append((f"{ticker}_net_margin", end_date, val, source, ticker.upper(), valid_as_of))

    for end_date, assets in total_assets.items():
        if assets <= 0:
            continue
        if end_date in long_term_debt:
            rows.append((f"{ticker}_debt_to_assets", end_date, long_term_debt[end_date] / assets * 100, source, ticker.upper(), valid_as_of))

    return rows


def ingest_edgar():
    session = requests.Session()
    session.headers.update({"User-Agent": EDGAR_USER_AGENT})
    valid_as_of = date.today()

    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            for company in COMPANIES:
                ticker = company["ticker"]
                cik = company["cik"]

                print(f"Fetching EDGAR {cik} ({ticker.upper()}) ...", flush=True)
                us_gaap = fetch_company_facts(cik, session).get("facts", {}).get("us-gaap", {})

                edgar_flow = {name: extract_annual(us_gaap, tags) for name, tags in FLOW_CONCEPTS.items()}
                edgar_balance = {name: extract_balance(us_gaap, tags) for name, tags in BALANCE_CONCEPTS.items()}

                print(f"  Fetching yfinance fallback for {ticker.upper()} ...", flush=True)
                yf_flow, yf_balance = fetch_yfinance_fallback(ticker.upper())

                flow = {name: merge_with_fallback(edgar_flow.get(name, {}), yf_flow.get(name, {})) for name in FLOW_CONCEPTS}
                balance = {name: merge_with_fallback(edgar_balance.get(name, {}), yf_balance.get(name, {})) for name in BALANCE_CONCEPTS}

                # Tag source as EDGAR when the date exists in EDGAR, otherwise YFINANCE
                edgar_dates = {d for v in edgar_flow.values() for d in v} | {d for v in edgar_balance.values() for d in v}

                edgar_rows = compute_derived(
                    {n: {d: v for d, v in vals.items() if d in edgar_dates} for n, vals in flow.items()},
                    {n: {d: v for d, v in vals.items() if d in edgar_dates} for n, vals in balance.items()},
                    ticker, "EDGAR",
                )
                yf_rows = compute_derived(
                    {n: {d: v for d, v in vals.items() if d not in edgar_dates} for n, vals in flow.items()},
                    {n: {d: v for d, v in vals.items() if d not in edgar_dates} for n, vals in balance.items()},
                    ticker, "YFINANCE",
                )

                rows = edgar_rows + yf_rows
                cur.executemany(UPSERT_SQL, rows)
                print(f"  Wrote {cur.rowcount} rows ({len(edgar_rows)} EDGAR, {len(yf_rows)} yfinance)", flush=True)

                time.sleep(0.5)

        conn.commit()

    print("EDGAR ingestion complete.")


if __name__ == "__main__":
    ingest_edgar()
