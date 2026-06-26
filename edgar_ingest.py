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

# Tags are merged across all fallbacks — companies switch XBRL tags across years
FLOW_CONCEPTS = {
    "revenue":             ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "rd_expense":          ["ResearchAndDevelopmentExpense"],
    "net_income":          ["NetIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex":               ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}

# Balance sheet items are point-in-time snapshots — safe to use from 10-K and 10-Q directly
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


def extract_balance(us_gaap: dict, tags: list[str]) -> dict[str, float]:
    """Point-in-time balance sheet values — merge tags, deduplicate by end date."""
    by_end: dict[str, dict] = {}
    for tag in tags:
        for o in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
            if o.get("form") not in ("10-K", "10-Q"):
                continue
            end = o["end"]
            if end not in by_end or o["filed"] > by_end[end]["filed"]:
                by_end[end] = o
    return {end: o["val"] for end, o in by_end.items()}


def extract_flow(us_gaap: dict, tags: list[str]) -> dict[str, float]:
    """
    Discrete quarterly and annual flow values via YTD subtraction.

    10-Qs report cumulative YTD figures (Q2 = Jan-Jun, Q3 = Jan-Sep).
    We subtract consecutive periods to get single-quarter values:
      Q1 discrete = Q1 YTD
      Q2 discrete = Q2 YTD - Q1 YTD
      Q3 discrete = Q3 YTD - Q2 YTD
      Q4 discrete = FY    - Q3 YTD   (stored at fiscal year end date)

    When quarterly data is missing for a year, falls back to the annual figure.
    """
    by_period: dict[tuple, dict] = {}
    for tag in tags:
        for o in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
            if o.get("form") not in ("10-K", "10-Q"):
                continue
            fy, fp = o.get("fy"), o.get("fp")
            if not fy or not fp:
                continue
            key = (fy, fp)
            if key not in by_period or o["filed"] > by_period[key]["filed"]:
                by_period[key] = o

    by_fy: dict[int, dict[str, dict]] = {}
    for (fy, fp), obs in by_period.items():
        by_fy.setdefault(fy, {})[fp] = obs

    result: dict[str, float] = {}
    for periods in by_fy.values():
        q1 = periods.get("Q1")
        q2 = periods.get("Q2")
        q3 = periods.get("Q3")
        fy_obs = periods.get("FY")

        if q1:
            result[q1["end"]] = q1["val"]
        if q2 and q1:
            result[q2["end"]] = q2["val"] - q1["val"]
        if q3 and q2:
            result[q3["end"]] = q3["val"] - q2["val"]
        if fy_obs and q3:
            result[fy_obs["end"]] = fy_obs["val"] - q3["val"]
        elif fy_obs and not (q1 or q2 or q3):
            result[fy_obs["end"]] = fy_obs["val"]

    return result


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
    op_cf = flow.get("operating_cash_flow", {})
    capex = flow.get("capex", {})
    long_term_debt = balance.get("long_term_debt", {})
    total_assets = balance.get("total_assets", {})

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
            rows.append((f"{ticker}_free_cash_flow", end_date, cf - capex[end_date], ticker.upper(), valid_as_of))

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

                flow = {name: extract_flow(us_gaap, tags) for name, tags in FLOW_CONCEPTS.items()}
                balance = {name: extract_balance(us_gaap, tags) for name, tags in BALANCE_CONCEPTS.items()}

                rows = compute_derived(flow, balance, ticker)
                cur.executemany(UPSERT_SQL, rows)
                print(f"  Wrote {cur.rowcount} rows", flush=True)

                time.sleep(0.2)

        conn.commit()

    print("EDGAR ingestion complete.")


if __name__ == "__main__":
    ingest_edgar()
