import os
import time
from datetime import date

import psycopg
import requests
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
    "capex":      ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}

# Point-in-time — safe to pull from 10-K and 10-Q
BALANCE_CONCEPTS = {
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "total_assets":   ["Assets"],
}

UPSERT_SQL = """
    INSERT INTO fact_observation (metric_id, obs_date, raw_value, source, series_id, valid_as_of)
    VALUES (%s, %s, %s, 'EDGAR', %s, %s)
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


def compute_derived(
    flow: dict[str, dict[str, float]],
    balance: dict[str, dict[str, float]],
    ticker: str,
) -> list[tuple]:
    rows = []
    valid_as_of = date.today()

    revenue = flow.get("revenue", {})
    rd = flow.get("rd_expense", {})
    net_income = flow.get("net_income", {})
    long_term_debt = balance.get("long_term_debt", {})
    total_assets = balance.get("total_assets", {})

    capex = flow.get("capex", {})

    for end_date, rev in revenue.items():
        if rev <= 0:
            continue
        if end_date in rd:
            val = rd[end_date] / rev * 100
            if 0 <= val <= 200:
                rows.append((f"{ticker}_rd_to_revenue", end_date, val, ticker.upper(), valid_as_of))
            if rd[end_date] > 0:
                rows.append((f"{ticker}_rd_expense", end_date, rd[end_date], ticker.upper(), valid_as_of))
        if end_date in net_income:
            val = net_income[end_date] / rev * 100
            if -100 <= val <= 100:
                rows.append((f"{ticker}_net_margin", end_date, val, ticker.upper(), valid_as_of))

    for end_date, capex_val in capex.items():
        if capex_val > 0:
            rows.append((f"{ticker}_capex", end_date, capex_val, ticker.upper(), valid_as_of))

    for end_date, assets in total_assets.items():
        if assets <= 0:
            continue
        if end_date in long_term_debt:
            rows.append((f"{ticker}_debt_to_assets", end_date, long_term_debt[end_date] / assets * 100, ticker.upper(), valid_as_of))

    return rows


def ingest_edgar():
    session = requests.Session()
    session.headers.update({"User-Agent": EDGAR_USER_AGENT})

    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            for company in COMPANIES:
                ticker = company["ticker"]
                cik = company["cik"]

                print(f"Fetching {cik} ({ticker.upper()}) ...", flush=True)
                us_gaap = fetch_company_facts(cik, session).get("facts", {}).get("us-gaap", {})

                flow = {name: extract_annual(us_gaap, tags) for name, tags in FLOW_CONCEPTS.items()}
                balance = {name: extract_balance(us_gaap, tags) for name, tags in BALANCE_CONCEPTS.items()}

                rows = compute_derived(flow, balance, ticker)
                cur.executemany(UPSERT_SQL, rows)
                print(f"  Wrote {cur.rowcount} rows", flush=True)

                time.sleep(0.2)

        conn.commit()

    print("EDGAR ingestion complete.")


if __name__ == "__main__":
    ingest_edgar()
