"""
AI Bubble Regime Score.

A transparent, theory-driven composite index — not a forecasting model.
It measures how bubble-like current macro/market conditions are relative
to history, by z-scoring a small set of canonical signals, sign-adjusting
each so that "positive = more bubble-like", and averaging the components
that exist for any given month.

The output is a regime score in standard-deviation units, plus a
percentile rank and the nearest historical analog dates. This is the
honest version of what the previous "crash probability" model claimed
to be: a measurement of where we are in the cycle, not a forecast.
"""
import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


@dataclass
class Component:
    metric_ids: tuple[str, ...]   # one or more DB metric_ids (summed if multiple)
    label: str
    pillar: str
    sign: int                     # +1 = high is bubbly, -1 = low is bubbly
    derive: str                   # "raw" | "yoy" — transformation before z-scoring


# Component selection: each has a theory-backed story, deep enough history
# to be meaningful, and is sign-adjusted so the composite reads consistently.
COMPONENTS: list[Component] = [
    # Valuation — the strongest historical bubble signal
    Component(("shiller_cape",), "CAPE", "Valuation", +1, "raw"),

    # Liquidity & monetary stance
    Component(("m2_yoy_growth",), "M2 YoY Growth", "Liquidity", +1, "raw"),
    Component(("fed_funds",), "Fed Funds (inv)", "Liquidity", -1, "raw"),

    # Credit conditions
    Component(("credit_spreads_baa",), "Credit Spreads (inv)", "Credit", -1, "raw"),
    Component(("credit_growth",), "Credit Growth YoY", "Credit", +1, "yoy"),

    # Term structure
    Component(("yield_curve_10y2y",), "Yield Curve (inv)", "Term Structure", -1, "raw"),

    # Sentiment / volatility
    Component(("vix",), "VIX (inv)", "Sentiment", -1, "raw"),

    # AI-specific overinvestment. YoY growth (not absolute level) — the level
    # always grows for non-stationary series like capex, which would spuriously
    # show as "extreme" every year. The bubble signal is acceleration beyond trend.
    Component(
        ("msft_capex", "googl_capex", "amzn_capex", "meta_capex", "nvda_capex"),
        "Hyperscaler Capex YoY", "AI Spend", +1, "yoy",
    ),
]


@dataclass
class RegimeResult:
    current_score: float                  # composite z-score for the most recent month
    current_date: date
    percentile: float                     # percentile of current_score in modern era (1990+)
    history: pd.DataFrame                 # date, score
    components_now: pd.DataFrame          # label, pillar, sign, raw_value, zscore, contribution
    pillar_scores: pd.DataFrame           # pillar, score
    analogs: pd.DataFrame                 # date, score, distance — nearest historical matches
    coverage: pd.DataFrame                # date index, count of components available


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_series(metric_ids: tuple[str, ...]) -> pd.Series:
    """Load one or more metrics, return monthly series. When multiple metrics
    are passed (e.g. hyperscaler capex across 5 tickers with different fiscal
    year ends) each is forward-filled to monthly *first*, then summed — so the
    composite reflects all available companies even when their reporting dates
    don't align."""
    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(metric_ids))
            cur.execute(
                f"SELECT metric_id, obs_date, raw_value FROM fact_observation "
                f"WHERE metric_id IN ({placeholders}) ORDER BY obs_date",
                list(metric_ids),
            )
            rows = cur.fetchall()

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["metric_id", "obs_date", "raw_value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    if len(metric_ids) > 1:
        # Per-metric monthly ffill, then sum row-wise — handles staggered fiscal years
        per_metric = (
            df.pivot_table(index="obs_date", columns="metric_id", values="raw_value")
              .resample("ME").last().ffill()
        )
        return per_metric.sum(axis=1, min_count=1)

    s = df.set_index("obs_date")["raw_value"]
    return s.resample("ME").last().ffill()


def _transform(series: pd.Series, derive: str) -> pd.Series:
    if derive == "yoy":
        return series.pct_change(12) * 100
    return series


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score over full available history. Returns NaN for periods with no data."""
    s = series.dropna()
    if s.std() == 0 or len(s) < 24:
        return pd.Series(np.nan, index=series.index)
    return (series - s.mean()) / s.std()


# ── Regime computation ───────────────────────────────────────────────────────

def build_regime() -> RegimeResult:
    component_zs: dict[str, pd.Series] = {}
    component_raw_now: dict[str, float] = {}
    component_z_now: dict[str, float] = {}
    component_meta = []

    for c in COMPONENTS:
        raw = _load_series(c.metric_ids)
        if raw.empty:
            continue
        transformed = _transform(raw, c.derive)
        z = _zscore(transformed) * c.sign

        component_zs[c.label] = z

        # Latest value (for the breakdown panel)
        latest_idx = transformed.last_valid_index()
        if latest_idx is not None:
            component_raw_now[c.label] = float(transformed.loc[latest_idx])
            component_z_now[c.label] = float(z.loc[latest_idx]) if not pd.isna(z.loc[latest_idx]) else 0.0
        else:
            component_raw_now[c.label] = float("nan")
            component_z_now[c.label] = 0.0

        component_meta.append({"label": c.label, "pillar": c.pillar, "sign": c.sign})

    # Build a single dataframe of all sign-adjusted z-scores aligned by month
    aligned = pd.DataFrame(component_zs)
    # Composite = mean of available components per month
    composite = aligned.mean(axis=1).dropna()
    coverage = aligned.notna().sum(axis=1)

    current_score = float(composite.iloc[-1])
    current_date = composite.index[-1].date()

    # Percentile uses the modern era (1990+) where most components exist —
    # avoids the early-history regime where only CAPE is available.
    modern = composite[composite.index >= "1990-01-01"]
    percentile = float((modern <= current_score).mean() * 100)

    # Per-component snapshot for the breakdown
    meta_df = pd.DataFrame(component_meta)
    meta_df["raw_value"] = meta_df["label"].map(component_raw_now)
    meta_df["zscore"] = meta_df["label"].map(component_z_now)
    meta_df["contribution"] = meta_df["zscore"]  # sign already applied
    components_now = meta_df.sort_values("contribution", ascending=False)

    pillar_scores = (
        components_now.groupby("pillar")["contribution"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"contribution": "score"})
    )

    # Historical analogs: nearest prior monthly scores by absolute distance.
    # Restricted to the modern era (1990+) where most components exist — comparing
    # against pre-1990 readings is misleading because those scores are mostly
    # just CAPE in isolation. Also exclude the last 18 months to avoid trivial
    # near-duplicates of the current reading.
    cutoff = composite.index.max() - pd.DateOffset(months=18)
    modern_candidates = composite[
        (composite.index >= "1990-01-01") & (composite.index <= cutoff)
    ]
    distances = (modern_candidates - current_score).abs().sort_values()
    analogs = pd.DataFrame({
        "date": distances.head(5).index.date,
        "score": modern_candidates.loc[distances.head(5).index].values,
        "distance": distances.head(5).values,
    })

    history = composite.reset_index()
    history.columns = ["date", "score"]
    coverage_df = coverage.reset_index()
    coverage_df.columns = ["date", "count"]

    return RegimeResult(
        current_score=current_score,
        current_date=current_date,
        percentile=percentile,
        history=history,
        components_now=components_now,
        pillar_scores=pillar_scores,
        analogs=analogs,
        coverage=coverage_df,
    )


if __name__ == "__main__":
    r = build_regime()
    print(f"\n=== AI Bubble Regime Score ===")
    print(f"Date:        {r.current_date}")
    print(f"Score:       {r.current_score:+.2f}σ")
    print(f"Percentile:  {r.percentile:.1f}% (vs 1990-present)")
    print(f"\nComponents now (sorted by contribution):")
    print(r.components_now.to_string(index=False))
    print(f"\nPillar scores:")
    print(r.pillar_scores.to_string(index=False))
    print(f"\nNearest historical analogs:")
    print(r.analogs.to_string(index=False))
