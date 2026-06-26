import os
from datetime import date

import numpy as np
import pandas as pd
import psycopg
import yfinance as yf
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

LOOKFORWARD_DAYS = 365
DRAWDOWN_THRESHOLD = 0.30

FRED_FEATURES = [
    "yield_curve_10y2y",
    "fed_funds",
    "financial_conditions",
    "m2_yoy_growth",
    "credit_spreads_baa",
    "vix",
]

# Combined hyperscaler capex — annual, forward-filled daily
CAPEX_TICKERS = ["msft", "googl", "amzn", "meta", "nvda"]


def fetch_ndx() -> pd.DataFrame:
    """Download Nasdaq 100 daily close, compute forward max drawdown label."""
    ticker = yf.Ticker("^NDX")
    hist = ticker.history(period="max")
    hist.index = hist.index.tz_localize(None).normalize()
    prices = hist["Close"].rename("price")

    # For each date, compute max drawdown over next 365 days
    labels = []
    price_arr = prices.values
    dates = prices.index

    for i, _ in enumerate(dates):
        future = price_arr[i: i + LOOKFORWARD_DAYS + 1]
        if len(future) < LOOKFORWARD_DAYS // 2:
            labels.append(np.nan)
            continue
        peak = future[0]
        trough = future.min()
        drawdown = (peak - trough) / peak
        labels.append(1 if drawdown >= DRAWDOWN_THRESHOLD else 0)

    df = pd.DataFrame({"price": price_arr, "label": labels}, index=dates)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def fetch_features() -> pd.DataFrame:
    """Load FRED signals + capex from DB, resample to daily, forward-fill."""
    capex_ids = [f"{t}_capex" for t in CAPEX_TICKERS]
    all_ids = FRED_FEATURES + capex_ids

    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(all_ids))
            cur.execute(
                f"""
                SELECT metric_id, obs_date, raw_value
                FROM fact_observation
                WHERE metric_id IN ({placeholders})
                ORDER BY obs_date
                """,
                all_ids,
            )
            rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["metric_id", "obs_date", "raw_value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    # Sum all capex into a single combined column
    capex_df = df[df["metric_id"].isin(capex_ids)].copy()
    if not capex_df.empty:
        capex_agg = capex_df.groupby("obs_date")["raw_value"].sum().reset_index()
        capex_agg["metric_id"] = "hyperscaler_capex"
        df = pd.concat([df[~df["metric_id"].isin(capex_ids)], capex_agg], ignore_index=True)

    pivoted = df.pivot_table(index="obs_date", columns="metric_id", values="raw_value")

    daily_idx = pd.date_range(pivoted.index.min(), pivoted.index.max(), freq="D")
    pivoted = pivoted.reindex(daily_idx).ffill()
    pivoted.index.name = "date"
    return pivoted


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add rate-of-change and rolling signals for FRED features only."""
    df = features[FRED_FEATURES].copy()

    for col in FRED_FEATURES:
        if col not in df.columns:
            continue
        df[f"{col}_chg_90d"] = df[col].pct_change(90)
        df[f"{col}_chg_365d"] = df[col].pct_change(365)
        df[f"{col}_zscore"] = (df[col] - df[col].rolling(252).mean()) / df[col].rolling(252).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()


def build_dataset() -> pd.DataFrame:
    print("Fetching NDX price history...", flush=True)
    ndx = fetch_ndx()

    print("Fetching FRED features from DB...", flush=True)
    features = fetch_features()

    print("Engineering features...", flush=True)
    features = engineer_features(features)

    dataset = ndx.join(features, how="inner")
    dataset = dataset.dropna()

    print(f"Dataset: {len(dataset)} rows, {dataset['label'].mean():.1%} positive (crash within 12mo)")
    return dataset


FEATURE_COLS = [
    "yield_curve_10y2y", "yield_curve_10y2y_chg_90d", "yield_curve_10y2y_zscore",
    "fed_funds", "fed_funds_chg_90d", "fed_funds_chg_365d",
    "financial_conditions", "financial_conditions_chg_90d", "financial_conditions_zscore",
    "m2_yoy_growth", "m2_yoy_growth_chg_90d",
    "credit_spreads_baa", "credit_spreads_baa_chg_90d", "credit_spreads_baa_zscore",
    "vix", "vix_chg_90d", "vix_zscore",
]


def train(dataset: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """
    Train-test split respecting time order (no lookahead).
    First 80% of rows = train, last 20% = test.
    Returns fitted pipeline and test results.
    """
    dataset = dataset.sort_index()
    features = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[features]
    y = dataset["label"]

    split = int(len(dataset) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1)),
    ])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nTest AUC: {auc:.3f}")
    print(classification_report(y_test, pipe.predict(X_test)))

    results = X_test.copy()
    results["label"] = y_test
    results["crash_prob"] = proba

    return pipe, results


def current_signal(pipe: Pipeline, dataset: pd.DataFrame) -> float:
    """Return crash probability for the most recent date in the dataset."""
    features = [c for c in FEATURE_COLS if c in dataset.columns]
    latest = dataset[features].iloc[[-1]]
    prob = pipe.predict_proba(latest)[0, 1]
    print(f"\nCurrent crash probability (30% NDX drawdown in 12mo): {prob:.1%}")
    print(f"As of: {dataset.index[-1].date()}")
    return prob


if __name__ == "__main__":
    ds = build_dataset()
    pipe, results = train(ds)
    current_signal(pipe, ds)
