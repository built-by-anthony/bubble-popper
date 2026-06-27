# Planned Improvements

## Data sources to add

### FRED
- **WTI Crude Oil (`DCOILWTICO`)** — daily since 1986. Indirect bubble signal: oil spikes trigger Fed rate hikes which compress multiples; oil collapses signal demand destruction ahead of recessions. Useful macro context for the predictive model.

### Shiller CAPE
- Available from Robert Shiller's website (http://www.econ.yale.edu/~shiller/data.htm), not FRED. Monthly S&P 500 cyclically adjusted P/E going back to 1871. Strong long-term valuation signal — CAPE > 30 has historically preceded decade-long low returns. Would need a dedicated scraper or manual download.

## Model enhancements

### More training examples
- Extend FRED features back to the 1960s where series are available — adds 1973 oil shock, 1987 Black Monday, early 90s recession as additional crash events. The core problem is only 3–4 crashes in the current training window; more history is the highest-leverage fix.

### Better features
- **Shiller CAPE** — single strongest long-term bubble indicator, currently missing. Available from Robert Shiller's website (see Data sources section). Would require a dedicated scraper or manual download.
- **Credit growth rate** — not just spreads but actual loan growth. FRED series `TOTLL` (total loans and leases).
- **Margin debt** — retail leverage indicator, available monthly from FINRA. High margin debt = late-stage bubble behavior.

### Better model architecture
- **Gradient boosting (XGBoost/LightGBM)** — handles non-linear relationships and feature interactions better than logistic regression. Requires more training data first or it will overfit the handful of crashes harder.
- **Survival analysis** — time-to-event modeling ("how long until a crash") is a more honest framing than binary classification ("will it crash in 12 months"). Removes the arbitrary 12-month window sensitivity.
- **Label ensembling** — run the model at multiple drawdown thresholds (20%, 25%, 30%) and average the probabilities to reduce sensitivity to the arbitrary cutoff.

## Dashboard enhancements

### Streamlit API deprecations
- `use_container_width=True` → `width="stretch"` (deprecated after 2025-12-31). Trivial replace across all charts.
- `st.components.v1.html` → new components API (deprecated after 2026-06-01). Affects the AI spend clocks.
