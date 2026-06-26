import os
import time
from datetime import date

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ["FRED_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"

METRICS = [
    {"metric_id": "yield_curve_10y2y",              "series_id": "T10Y2Y"},
    {"metric_id": "yield_curve_10y3m",              "series_id": "T10Y3M"},
    {"metric_id": "treasury_10y",                   "series_id": "DGS10"},
    {"metric_id": "fed_funds",                      "series_id": "DFF"},
    {"metric_id": "financial_conditions",           "series_id": "NFCI"},
    {"metric_id": "financial_conditions_leverage",  "series_id": "NFCILEVERAGE"},
    {"metric_id": "financial_conditions_nonfin_leverage", "series_id": "NFCINONFINLEVERAGE"},
    {"metric_id": "m2_money_supply",                "series_id": "M2SL"},
]

UPSERT_SQL = """
    INSERT INTO fact_observation (metric_id, obs_date, raw_value, source, series_id, valid_as_of)
    VALUES (%s, %s, %s, 'FRED', %s, %s)
    ON CONFLICT (metric_id, obs_date)
    DO UPDATE SET raw_value   = EXCLUDED.raw_value,
                  series_id   = EXCLUDED.series_id,
                  valid_as_of = EXCLUDED.valid_as_of
    WHERE fact_observation.valid_as_of < EXCLUDED.valid_as_of
"""


def fetch_observations(series_id: str) -> list[dict]:
    resp = requests.get(
        FRED_ENDPOINT,
        params={"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["observations"]


def compute_m2_yoy(observations: list[dict], valid_as_of: datetime) -> list[tuple]:
    values = {
        date.fromisoformat(obs["date"]): float(obs["value"])
        for obs in observations
        if obs["value"] != "."
    }
    rows = []
    for obs_date, value in values.items():
        prior = obs_date.replace(year=obs_date.year - 1)
        if prior in values:
            yoy = (value - values[prior]) / values[prior] * 100
            rows.append(("m2_yoy_growth", obs_date, yoy, "M2SL", valid_as_of))
    return rows


def ingest_fred():
    valid_as_of = date.today()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for metric in METRICS:
                metric_id = metric["metric_id"]
                series_id = metric["series_id"]

                print(f"Fetching {series_id} -> {metric_id} ...", flush=True)
                observations = fetch_observations(series_id)

                rows = [
                    (metric_id, obs["date"], float(obs["value"]), series_id, valid_as_of)
                    for obs in observations
                    if obs["value"] != "."
                ]

                cur.executemany(UPSERT_SQL, rows)
                print(f"  Wrote {cur.rowcount} rows (fetched {len(rows)} from FRED)", flush=True)

                if metric_id == "m2_money_supply":
                    yoy_rows = compute_m2_yoy(observations, valid_as_of)
                    cur.executemany(UPSERT_SQL, yoy_rows)
                    print(f"  Wrote {cur.rowcount} rows for m2_yoy_growth", flush=True)

                time.sleep(0.5)

        conn.commit()

    print("FRED ingestion complete.")


if __name__ == "__main__":
    ingest_fred()
