import os
import time
from datetime import date, datetime

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

# First matching tag wins — companies don't all use the same XBRL concept
CONCEPTS = {
    "revenue":             ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "rd_expense":          ["ResearchAndDevelopmentExpense"],
    "net_income":          ["NetIncomeLoss"],
    "long_term_debt":      ["LongTermDebt", "LongTermDebtNoncurrent"],
    "total_assets":        ["Assets"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex":               ["PaymentsToAcquirePropertyPlantAndEquipment"],
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


def period_days(obs: dict) -> int | None:
    if "start" not in obs:
        return None
    start = datetime.strptime(obs["start"], "%Y-%m-%d")
    end = datetime.strptime(obs["end"], "%Y-%m-%d")
    return (end - start).days


def extract_concept(facts: dict, tags: list[str]) -> dict[str, float]:
    """
    Returns {end_date: value} for the first matching XBRL tag.
    For flow items (income stmt, cash flow), keeps only quarterly (~90d) and
    annual (~365d) periods to avoid mixing YTD figures with discrete periods.
    Deduplicates by end date, keeping the latest filed observation.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        if tag not in us_gaap:
            continue
        raw = us_gaap[tag].get("units", {}).get("USD", [])
        obs = [o for o in raw if o.get("form") in ("10-K", "10-Q")]
        if not obs:
            continue

        # For flow items with start dates, filter to discrete periods only
        has_start = any("start" in o for o in obs)
        if has_start:
            obs = [
                o for o in obs
                if "start" not in o or period_days(o) in range(75, 100) or period_days(o) in range(340, 380)
            ]

        # Deduplicate by end date — keep latest filed
        by_end: dict[str, dict] = {}
        for o in obs:
            end = o["end"]
            if end not in by_end or o["filed"] > by_end[end]["filed"]:
                by_end[end] = o

        return {end: entry["val"] for end, entry in by_end.items()}

    return {}


def compute_derived(concepts: dict[str, dict[str, float]], ticker: str) -> list[tuple]:
    rows = []
    valid_as_of = date.today()

    revenue = concepts.get("revenue", {})
    rd = concepts.get("rd_expense", {})
    net_income = concepts.get("net_income", {})
    long_term_debt = concepts.get("long_term_debt", {})
    total_assets = concepts.get("total_assets", {})
    op_cf = concepts.get("operating_cash_flow", {})
    capex = concepts.get("capex", {})

    for end_date, rev in revenue.items():
        if rev == 0:
            continue
        if end_date in rd:
            rows.append((f"{ticker}_rd_to_revenue", end_date, rd[end_date] / rev * 100, ticker.upper(), valid_as_of))
        if end_date in net_income:
            rows.append((f"{ticker}_net_margin", end_date, net_income[end_date] / rev * 100, ticker.upper(), valid_as_of))

    for end_date, assets in total_assets.items():
        if assets == 0:
            continue
        if end_date in long_term_debt:
            rows.append((f"{ticker}_debt_to_assets", end_date, long_term_debt[end_date] / assets * 100, ticker.upper(), valid_as_of))

    for end_date, cf in op_cf.items():
        if end_date in capex:
            fcf = cf - capex[end_date]
            rows.append((f"{ticker}_free_cash_flow", end_date, fcf, ticker.upper(), valid_as_of))

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
                facts = fetch_company_facts(cik, session)

                concepts = {
                    name: extract_concept(facts, tags)
                    for name, tags in CONCEPTS.items()
                }

                rows = compute_derived(concepts, ticker)
                cur.executemany(UPSERT_SQL, rows)
                print(f"  Wrote {cur.rowcount} rows", flush=True)

                time.sleep(0.2)

        conn.commit()

    print("EDGAR ingestion complete.")


if __name__ == "__main__":
    ingest_edgar()
