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
ENSEMBLE_THRESHOLDS = [0.15, 0.20, 0.25]

FRED_FEATURES = [
    "yield_curve_10y2y",
    "fed_funds",
    "financial_conditions",
    "m2_yoy_growth",
    "credit_spreads_baa",
    "vix",
    "credit_growth",
    "recession_indicator",
    "oil_wti",
]

CAPEX_TICKERS = ["msft", "googl", "amzn", "meta", "nvda"]

BINARY_FEATURES = {"recession_indicator"}
CONTINUOUS_FEATURES = [f for f in FRED_FEATURES if f not in BINARY_FEATURES]

FEATURE_COLS = [
    "yield_curve_10y2y", "yield_curve_10y2y_chg_90d", "yield_curve_10y2y_zscore",
    "fed_funds", "fed_funds_chg_90d", "fed_funds_chg_365d",
    "financial_conditions", "financial_conditions_chg_90d", "financial_conditions_zscore",
    "m2_yoy_growth", "m2_yoy_growth_chg_90d",
    "credit_spreads_baa", "credit_spreads_baa_chg_90d", "credit_spreads_baa_zscore",
    "vix", "vix_chg_90d", "vix_zscore",
    "credit_growth", "credit_growth_chg_90d", "credit_growth_chg_365d", "credit_growth_zscore",
    "recession_indicator",
    "oil_wti", "oil_wti_chg_90d", "oil_wti_chg_365d", "oil_wti_zscore",
]


def fetch_ndx(threshold: float) -> pd.DataFrame:
    """Download Nasdaq 100 daily close, compute forward max drawdown label."""
    ticker = yf.Ticker("^NDX")
    hist = ticker.history(period="max")
    hist.index = hist.index.tz_localize(None).normalize()
    prices = hist["Close"].rename("price")

    labels = []
    price_arr = prices.values

    for i in range(len(price_arr)):
        future = price_arr[i: i + LOOKFORWARD_DAYS + 1]
        if len(future) < LOOKFORWARD_DAYS // 2:
            labels.append(np.nan)
            continue
        peak = future[0]
        trough = future.min()
        drawdown = (peak - trough) / peak
        labels.append(1 if drawdown >= threshold else 0)

    df = pd.DataFrame({"price": price_arr, "label": labels}, index=prices.index)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def fetch_features() -> pd.DataFrame:
    """Load FRED signals from DB, resample to daily, forward-fill."""
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
    """Add rate-of-change and rolling signals for continuous FRED features."""
    df = features[FRED_FEATURES].copy()

    for col in CONTINUOUS_FEATURES:
        if col not in df.columns:
            continue
        df[f"{col}_chg_90d"] = df[col].pct_change(90)
        df[f"{col}_chg_365d"] = df[col].pct_change(365)
        df[f"{col}_zscore"] = (df[col] - df[col].rolling(252).mean()) / df[col].rolling(252).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()


def _train_single(dataset: pd.DataFrame, verbose: bool = False) -> tuple[Pipeline, float]:
    """Train one logistic regression model, return pipeline and test AUC."""
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

    if verbose:
        print(f"  AUC: {auc:.3f}")
        print(classification_report(y_test, pipe.predict(X_test)))

    return pipe, auc


def build_dataset() -> pd.DataFrame:
    """Compatibility shim — returns dataset at default threshold (0.20)."""
    print("Fetching NDX price history...", flush=True)
    ndx = fetch_ndx(0.20)
    print("Fetching FRED features from DB...", flush=True)
    features = engineer_features(fetch_features())
    dataset = ndx.join(features, how="inner").dropna()
    print(f"Dataset: {len(dataset)} rows, {dataset['label'].mean():.1%} positive")
    return dataset


def train(dataset: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """Compatibility shim used by dashboard's load_model()."""
    pipe, auc = _train_single(dataset, verbose=True)
    features = [c for c in FEATURE_COLS if c in dataset.columns]
    proba = pipe.predict_proba(dataset[features])[:, 1]
    results = dataset[features].copy()
    results["label"] = dataset["label"]
    results["crash_prob"] = proba
    return pipe, results


def build_ensemble() -> tuple[float, pd.Series, pd.DataFrame]:
    """
    Train one model per threshold, average probabilities across all three.
    Returns (current_prob, avg_coefs, prob_history).
    """
    print("Fetching FRED features from DB...", flush=True)
    features_eng = engineer_features(fetch_features())
    feat_cols = [c for c in FEATURE_COLS if c in features_eng.columns]

    all_proba_series = []
    all_coefs = []
    aucs = []

    for threshold in ENSEMBLE_THRESHOLDS:
        print(f"\nTraining model at {threshold:.0%} threshold...", flush=True)
        ndx = fetch_ndx(threshold)
        dataset = ndx.join(features_eng, how="inner").dropna()
        print(f"  {len(dataset)} rows, {dataset['label'].mean():.1%} positive")

        pipe, auc = _train_single(dataset, verbose=True)
        aucs.append(auc)

        proba = pipe.predict_proba(dataset[feat_cols])[:, 1]
        all_proba_series.append(pd.Series(proba, index=dataset.index))
        all_coefs.append(pipe.named_steps["clf"].coef_[0])

    avg_proba = pd.concat(all_proba_series, axis=1).mean(axis=1)
    avg_coefs = pd.Series(
        np.mean(all_coefs, axis=0), index=feat_cols
    ).sort_values(key=abs, ascending=False)

    current_prob = float(avg_proba.iloc[-1])
    prob_history = avg_proba.reset_index()
    prob_history.columns = ["date", "crash_prob"]

    print(f"\nEnsemble AUC (avg): {np.mean(aucs):.3f}")
    print(f"Current crash probability (ensemble): {current_prob:.1%}")
    print(f"As of: {avg_proba.index[-1].date()}")

    return current_prob, avg_coefs, prob_history


def current_signal(pipe: Pipeline, dataset: pd.DataFrame) -> float:
    features = [c for c in FEATURE_COLS if c in dataset.columns]
    prob = float(pipe.predict_proba(dataset[features].iloc[[-1]])[0, 1])
    print(f"\nCurrent crash probability: {prob:.1%}")
    return prob


if __name__ == "__main__":
    prob, coefs, history = build_ensemble()
