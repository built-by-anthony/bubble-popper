import os
from datetime import date
from io import StringIO

import pandas as pd
import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# multpl.com maintains the Shiller CAPE (CAPE / Shiller PE) series, full history
# back to 1871 AND kept current — unlike Shiller's own ie_data.xls which froze
# at 2023-09. Same underlying series.
MULTPL_URL = "https://www.multpl.com/shiller-pe/table/by-month"

UPSERT_SQL = """
    INSERT INTO fact_observation (metric_id, obs_date, raw_value, source, series_id, valid_as_of)
    VALUES (%s, %s, %s, 'SHILLER', 'CAPE', %s)
    ON CONFLICT (metric_id, obs_date)
    DO UPDATE SET raw_value   = EXCLUDED.raw_value,
                  series_id   = EXCLUDED.series_id,
                  valid_as_of = EXCLUDED.valid_as_of
    WHERE fact_observation.valid_as_of < EXCLUDED.valid_as_of
"""


def fetch_cape() -> list[tuple]:
    print("Fetching Shiller CAPE from multpl.com...", flush=True)
    resp = requests.get(MULTPL_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    df = pd.read_html(StringIO(resp.text))[0]
    df["parsed"] = pd.to_datetime(df["Date"], format="mixed")
    # Normalize every reading to the first of its month
    df["obs_date"] = df["parsed"].values.astype("datetime64[M]")
    # Current month carries a partial-month reading dated mid-month; keep the
    # freshest reading per month by sorting on the true date and dropping dupes.
    df = df.sort_values("parsed").drop_duplicates("obs_date", keep="last")

    valid_as_of = date.today()
    rows = [
        ("shiller_cape", pd.Timestamp(od).date(), float(val), valid_as_of)
        for od, val in zip(df["obs_date"], df["Value"])
        if pd.notna(val)
    ]
    return rows


def ingest_shiller():
    rows = fetch_cape()
    print(f"Parsed {len(rows)} CAPE observations ({rows[0][1]} to {rows[-1][1]})")
    print(f"Latest CAPE: {rows[-1][2]:.1f}")

    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
            print(f"Upserted {cur.rowcount} rows")
        conn.commit()

    print("Shiller ingestion complete.")


if __name__ == "__main__":
    ingest_shiller()
