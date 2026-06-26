# CLAUDE.md — FRED Ingestion

Guidance for working on the FRED data-pull portion of the AI-bubble data warehouse. Read this before writing or modifying ingestion code.

## What this part of the project does

Pulls macroeconomic time series from the FRED API (St. Louis Fed) and lands them in a single stacked observation table. This is one of several ingestion sources (others: SEC EDGAR, yfinance/Stooq, Google Trends) feeding a shared dimensional model. The goal of the warehouse is to detect/time the AI bubble via a composite signal index; FRED supplies the macro/leverage context cluster.

## Core design principle: one stacked table, not one table per series

Every FRED series has the same shape (date, value). Do **not** create a table per series. All series land in `fact_observation`, with series identity in a column:

```
fact_observation
  metric_id     -- internal name, e.g. 'fed_funds'   (NOT the FRED code)
  obs_date
  raw_value     -- float
  source        -- 'FRED'
  valid_as_of   -- ingestion timestamp
```

Adding a new series must be a config change (one new mapping entry + one `dim_metric` row), never a schema change. If a task tempts you toward `CREATE TABLE` per series, stop — that breaks the model.

## The metric_id ↔ FRED series_id mapping

Keep our clean internal `metric_id` separate from FRED's vendor code. The FRED code lives in `dim_metric.source_series_id`. Never use the FRED code as the `metric_id` — if we later pull the same concept from another source, the `metric_id` must stay stable so nothing downstream breaks.

Current FRED series (all verified live):

| metric_id | source_series_id | frequency | direction |
|---|---|---|---|
| yield_curve_10y2y | T10Y2Y | daily | -1 |
| yield_curve_10y3m | T10Y3M | daily | -1 |
| treasury_10y | DGS10 | daily | mixed |
| fed_funds | DFF | daily | -1 |
| financial_conditions | NFCI | weekly | -1 |
| financial_conditions_leverage | NFCILEVERAGE | weekly | -1 |
| financial_conditions_nonfin_leverage | NFCINONFINLEVERAGE | weekly | -1 |
| m2_money_supply | M2SL | monthly | +1 |
| m2_yoy_growth | (derived from M2SL) | monthly | +1 |

`direction` = does a HIGH value mean MORE bubbly? The NFCI family is inverted (low/negative = loose conditions = frothy), so those are -1 and get sign-flipped in the normalization view. Get these signs right; they are the whole point of the signal panel.

## API specifics (FRED v2, as of 2026)

- Endpoint: `https://api.stlouisfed.org/fred/series/observations`
- Params: `series_id`, `api_key`, `file_type=json`
- A key is **required** (free, register at fredaccount.stlouisfed.org). FRED enforced this with v2 in Nov 2025 — keyless scraping no longer works.
- Rate limit: ~120 requests/min. We're far under it with ~8 series, but pace calls anyway as the list grows.
- Response: an `observations` array of `{date, value}` objects.

### Response gotchas — handle these or ingestion breaks

1. **Missing values come as the string `"."`, not null.** Drop or null these rows; never coerce `"."` to float.
2. **`value` is always a string**, even when numeric. Cast to float explicitly.
3. **Mixed frequencies are expected** (daily DFF, weekly NFCI, monthly M2SL). This is fine — different row densities per `metric_id`. Do not try to align frequencies at storage time; resample in the downstream view.

## Idempotency — the non-negotiable rule

The cron re-runs and FRED revises data. The table MUST have a unique constraint on `(metric_id, obs_date)`, and writes MUST upsert:

```sql
INSERT ... ON CONFLICT (metric_id, obs_date)
DO UPDATE SET raw_value = EXCLUDED.raw_value,
              valid_as_of = EXCLUDED.valid_as_of
```

Without this, every cron run duplicates rows. Put the constraint in place before the first real run.

## Secrets

The FRED key goes in an env var, never hardcoded. This repo is intended to go public.

## Verification after a run

- Row count per `metric_id` — daily series should dwarf monthly.
- Min/max `obs_date` per series.
- Spot-check one value against the FRED website.

## Stack context

- Storage: Postgres (Supabase). Small data — do not reach for warehouse-scale tooling.
- Orchestration: GitHub Actions cron. No Airflow.
- Keep ingestion as one function per source; the FRED puller is a single loop over the mapping list.

## Out of scope for this part

Schema DDL beyond the observation table, the normalization/composite-index view, and the other ingestion sources are handled elsewhere. Don't build them here unless asked.