# bubble-popper

Macroeconomic and fundamental data ingestion pipeline for detecting and timing the AI bubble. Pulls from FRED and SEC EDGAR into a single Postgres observation table.

## Data sources

**FRED** — macro/leverage context
- Yield curve spreads (10y2y, 10y3m)
- Treasury 10y rate
- Fed funds rate
- NFCI financial conditions (headline + leverage + non-financial leverage)
- M2 money supply + YoY growth

**SEC EDGAR** — company fundamentals (MSFT, GOOGL, AMZN, META, NVDA)
- R&D as % of revenue
- Net margin
- Long-term debt as % of assets
- Free cash flow

## Setup

```bash
uv sync
cp .env.example .env
# fill in .env
uv run main.py
```

## Environment variables

| Variable | Description |
|---|---|
| `FRED_API_KEY` | Free key from [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org) |
| `DATABASE_URL` | Supabase connection pooler URL (port 6543) |
| `EDGAR_USER_AGENT` | Required by SEC — e.g. `Your Name email@example.com` |

## Running

```bash
uv run main.py          # both sources
uv run fred_ingest.py   # FRED only
uv run edgar_ingest.py  # EDGAR only
```

## Storage

All observations land in a single `fact_observation` table in Postgres (Supabase). Writes are idempotent — re-running on the same day is a no-op.

## Orchestration

GitHub Actions cron runs daily at 8am UTC. Trigger manually from the Actions tab.
