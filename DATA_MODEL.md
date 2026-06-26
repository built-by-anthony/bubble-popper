# Data Model & Theory

## The thesis

Asset bubbles follow a pattern: cheap money flows into a narrative, valuations detach from fundamentals, leverage builds up, and the whole thing unwinds when the narrative cracks or liquidity tightens. The AI bubble, if it exists, should be visible in the data as a combination of:

- **Loose macro conditions** — low rates, easy financial conditions, rapid money supply growth — that provide the fuel
- **Overextended company fundamentals** — R&D spend outpacing revenue growth, margins compressing, free cash flow turning negative — that signal the burn rate
- **Leverage buildup** — debt growing as a share of assets, both at the macro level and company level

No single signal is reliable. The composite index averages across all of them.

---

## Data sources

### FRED (macro/leverage context)

Pulled daily via the FRED API v2. All series land in `fact_observation` with `source = 'FRED'`.

| metric_id | FRED series | What it measures | Direction |
|---|---|---|---|
| `yield_curve_10y2y` | T10Y2Y | 10y minus 2y Treasury spread | -1 (inversion = recession risk) |
| `yield_curve_10y3m` | T10Y3M | 10y minus 3m Treasury spread | -1 |
| `treasury_10y` | DGS10 | 10-year Treasury yield | mixed |
| `fed_funds` | DFF | Federal funds rate | -1 (low rate = loose money) |
| `financial_conditions` | NFCI | Chicago Fed financial conditions | -1 (negative = loose = frothy) |
| `financial_conditions_leverage` | NFCILEVERAGE | Leverage subindex of NFCI | -1 |
| `financial_conditions_nonfin_leverage` | NFCINONFINLEVERAGE | Non-financial leverage subindex | -1 |
| `m2_money_supply` | M2SL | Total money supply (monthly) | +1 |
| `m2_yoy_growth` | derived from M2SL | YoY % change in M2 | +1 |

`direction` indicates whether a HIGH value means MORE bubble-like. NFCI metrics are inverted — low/negative means loose financial conditions, which is frothy.

`m2_yoy_growth` is computed in Python during ingestion, not pulled directly from FRED.

---

### SEC EDGAR (company fundamentals)

Pulled quarterly via the EDGAR XBRL API. One call per company fetches all reported facts. All series land in `fact_observation` with `source = 'EDGAR'`.

**Companies tracked:** MSFT, GOOGL, AMZN, META, NVDA

**Metrics per company:**

| metric_id pattern | What it measures | Bubble signal |
|---|---|---|
| `{ticker}_rd_to_revenue` | R&D expense / revenue (%) | High = burning cash on unproven tech |
| `{ticker}_net_margin` | Net income / revenue (%) | Low/negative = fundamentals not supporting valuation |
| `{ticker}_debt_to_assets` | Long-term debt / total assets (%) | High = leveraged up during hype cycle |
| `{ticker}_free_cash_flow` | Operating cash flow - capex ($) | Negative = cash burn mode |

#### Handling YTD figures in 10-Qs

SEC 10-Q filings report **cumulative year-to-date figures**, not single-quarter figures. A Q3 filing for revenue means January through September, not just July through September.

To get discrete quarterly values we subtract consecutive periods:

```
Q1 discrete = Q1 YTD
Q2 discrete = Q2 YTD − Q1 YTD
Q3 discrete = Q3 YTD − Q2 YTD
Q4 discrete = Full Year (10-K) − Q3 YTD
```

Balance sheet items (debt, assets) are point-in-time snapshots and do not need this treatment — we use them directly from both 10-K and 10-Q filings.

#### XBRL tag fallbacks

Companies do not all use the same XBRL concept names, and individual companies sometimes switch tags across filing years. We merge observations across all fallback tags, keeping the latest filed value per period:

- Revenue: `RevenueFromContractWithCustomerExcludingAssessedTax` → `Revenues` → `SalesRevenueNet`
- Capex: `PaymentsToAcquirePropertyPlantAndEquipment` → `PaymentsToAcquireProductiveAssets`

---

## Storage

All observations land in a single table:

```sql
fact_observation (
    metric_id   text,        -- e.g. 'fed_funds', 'nvda_net_margin'
    obs_date    date,        -- period end date
    raw_value   float,       -- as reported (%, $, rate)
    source      text,        -- 'FRED' or 'EDGAR'
    series_id   text,        -- FRED series code or ticker
    valid_as_of date,        -- ingestion date (not updated twice on same day)
    PRIMARY KEY (metric_id, obs_date)
)
```

All writes are upserts. Re-running on the same day is a no-op (`valid_as_of` guards the update). Running the next day picks up any FRED revisions or new EDGAR filings.

---

## Composite signal

The dashboard computes a composite bubble signal by:

1. Z-scoring each FRED metric over its full history (mean = 0, std = 1)
2. Averaging the z-scores across all selected metrics
3. Plotting the result as a single time series

Above zero = conditions more bubble-like than the historical average. The signal is directional, not predictive — it shows where we are in the cycle relative to history, not when the bubble pops.
