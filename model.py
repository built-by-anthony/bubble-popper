import os
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import psycopg
import yfinance as yf
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

LOOKFORWARD_DAYS = 365
ENSEMBLE_THRESHOLDS = [0.15, 0.20, 0.25]

# S&P 500 — deep history back to 1928, captures ~2x the crash events of NDX.
# AI megacaps are ~30% of the S&P so this remains directly relevant to the AI bubble.
LABEL_INDEX = "^GSPC"

# Deep-history features only — all available back to at least 1976. The binding
# constraint is yield_curve_10y2y (1976). Adds 1980, 1982, 1987, 1990 crashes.
FRED_FEATURES = [
    "yield_curve_10y2y",
    "fed_funds",
    "m2_yoy_growth",
    "credit_growth",
    "recession_indicator",
    "shiller_cape",
]

CAPEX_TICKERS = ["msft", "googl", "amzn", "meta", "nvda"]
BINARY_FEATURES = {"recession_indicator"}
CONTINUOUS_FEATURES = [f for f in FRED_FEATURES if f not in BINARY_FEATURES]

FEATURE_COLS = [
    "yield_curve_10y2y", "yield_curve_10y2y_chg_90d",
    "fed_funds", "fed_funds_chg_365d",
    "m2_yoy_growth",
    "credit_growth", "credit_growth_chg_365d",
    "recession_indicator",
    "shiller_cape",
]


@dataclass
class ModelResult:
    current_prob: float                        # ensemble-mean calibrated probability for today
    current_ci_lo: float                       # min across ensemble (crude lower band)
    current_ci_hi: float                       # max across ensemble (crude upper band)
    current_date: date                         # date of the prediction
    coefs: pd.Series                           # mean coefficients across ensemble
    prob_history: pd.DataFrame                 # date, crash_prob (ensemble mean, calibrated)
    walk_forward: pd.DataFrame                 # per-fold: test_start, test_end, auc, brier, n_pos
    calibration: pd.DataFrame                  # predicted_bin, observed_frequency, count
    per_threshold: list[dict] = field(default_factory=list)  # debug detail per threshold model


# ── Data loading ──────────────────────────────────────────────────────────────

def fetch_ndx(threshold: float) -> pd.DataFrame:
    """Download S&P 500 daily close, compute forward max drawdown label."""
    ticker = yf.Ticker(LABEL_INDEX)
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
    """Load FRED + Shiller features from DB, resample to daily, forward-fill."""
    capex_ids = [f"{t}_capex" for t in CAPEX_TICKERS]
    all_ids = FRED_FEATURES + capex_ids + ["shiller_cape"]

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
    df = features[FRED_FEATURES].copy()
    for col in CONTINUOUS_FEATURES:
        if col not in df.columns:
            continue
        df[f"{col}_chg_90d"] = df[col].pct_change(90)
        df[f"{col}_chg_365d"] = df[col].pct_change(365)
        df[f"{col}_zscore"] = (df[col] - df[col].rolling(252).mean()) / df[col].rolling(252).std()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()


# ── Modeling ──────────────────────────────────────────────────────────────────

def _base_pipeline() -> Pipeline:
    """Logistic regression with standardization. No class_weight balancing —
    calibration handles class imbalance honestly instead of inflating positives."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.05)),
    ])


def _fit_calibrated(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 3) -> CalibratedClassifierCV:
    """Fit a logistic regression and calibrate via isotonic regression using
    time-series CV (no shuffling — calibration data is always future-of-fit)."""
    n_pos = int(y_train.sum())
    splits = max(2, min(n_splits, n_pos - 1)) if n_pos > 2 else 2
    cal = CalibratedClassifierCV(
        estimator=_base_pipeline(),
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=splits),
    )
    cal.fit(X_train, y_train)
    return cal


def _walk_forward(X: pd.DataFrame, y: pd.Series, n_folds: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """Walk-forward CV: train on past, evaluate next chunk. Returns:
      - per-fold metrics (auc, brier, n_pos, dates)
      - concatenated out-of-sample calibrated predictions across all folds (for calibration curve)
    """
    n = len(X)
    folds = []
    oos_preds = []
    oos_actuals = []

    # Start eval after a meaningful initial training window
    start_eval = int(n * 0.30)
    step = (n - start_eval) // n_folds

    for k in range(n_folds):
        tr_end = start_eval + k * step
        te_end = tr_end + step if k < n_folds - 1 else n
        if te_end - tr_end < 30:
            continue

        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]

        if yte.nunique() < 2 or ytr.sum() < 4:
            folds.append({
                "fold": k + 1,
                "test_start": X.index[tr_end].date(),
                "test_end": X.index[te_end - 1].date(),
                "n_test": len(yte),
                "n_pos": int(yte.sum()),
                "auc": np.nan,
                "brier": np.nan,
            })
            continue

        try:
            cal = _fit_calibrated(Xtr, ytr)
            proba = cal.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, proba)
            brier = brier_score_loss(yte, proba)
            oos_preds.extend(proba)
            oos_actuals.extend(yte.values)
        except Exception:
            auc = brier = np.nan

        folds.append({
            "fold": k + 1,
            "test_start": X.index[tr_end].date(),
            "test_end": X.index[te_end - 1].date(),
            "n_test": len(yte),
            "n_pos": int(yte.sum()),
            "auc": auc,
            "brier": brier,
        })

    return pd.DataFrame(folds), pd.Series(oos_preds, name="proba"), pd.Series(oos_actuals, name="actual")


def _calibration_table(preds: pd.Series, actuals: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions, report observed frequency per bin."""
    if len(preds) == 0:
        return pd.DataFrame(columns=["predicted_mean", "observed_frequency", "count"])
    try:
        prob_true, prob_pred = calibration_curve(actuals, preds, n_bins=n_bins, strategy="quantile")
    except Exception:
        return pd.DataFrame(columns=["predicted_mean", "observed_frequency", "count"])
    bin_edges = np.quantile(preds, np.linspace(0, 1, n_bins + 1))
    counts = np.histogram(preds, bins=bin_edges)[0]
    return pd.DataFrame({
        "predicted_mean": prob_pred,
        "observed_frequency": prob_true,
        "count": counts[: len(prob_pred)],
    })


def build_ensemble() -> ModelResult:
    """Train calibrated ensemble across drawdown thresholds with honest CV."""
    print("Loading features...", flush=True)
    features_eng = engineer_features(fetch_features())
    feat_cols = [c for c in FEATURE_COLS if c in features_eng.columns]

    today_features = features_eng[feat_cols].iloc[[-1]]
    today_date = features_eng.index[-1].date()

    per_threshold_preds = []
    per_threshold_history = []
    per_threshold_coefs = []
    per_threshold_walk = []
    all_oos_preds = []
    all_oos_actuals = []
    per_threshold_debug = []

    for threshold in ENSEMBLE_THRESHOLDS:
        print(f"\nThreshold {threshold:.0%}...", flush=True)
        ndx = fetch_ndx(threshold)
        dataset = ndx.join(features_eng, how="inner").dropna().sort_index()
        X = dataset[feat_cols]
        y = dataset["label"]

        # Walk-forward for this threshold
        walk_df, oos_preds, oos_actuals = _walk_forward(X, y, n_folds=6)
        walk_df["threshold"] = threshold
        per_threshold_walk.append(walk_df)
        all_oos_preds.extend(oos_preds.values)
        all_oos_actuals.extend(oos_actuals.values)

        # Production model: trained on all data, calibrated via time-series CV
        cal = _fit_calibrated(X, y)
        prob_today = float(cal.predict_proba(today_features[feat_cols])[0, 1])
        per_threshold_preds.append(prob_today)

        # Historical predictions for the chart (in-sample but calibrated)
        history_proba = cal.predict_proba(X)[:, 1]
        per_threshold_history.append(pd.Series(history_proba, index=X.index))

        # Average the underlying logistic coefficients across calibration folds
        raw_coefs = []
        for cc in cal.calibrated_classifiers_:
            est = cc.estimator if hasattr(cc, "estimator") else cc.base_estimator
            raw_coefs.append(est.named_steps["clf"].coef_[0])
        per_threshold_coefs.append(np.mean(raw_coefs, axis=0))

        valid_aucs = walk_df["auc"].dropna()
        valid_briers = walk_df["brier"].dropna()
        per_threshold_debug.append({
            "threshold": threshold,
            "n_rows": len(dataset),
            "pos_rate": float(y.mean()),
            "current_prob": prob_today,
            "wf_auc_mean": float(valid_aucs.mean()) if len(valid_aucs) else np.nan,
            "wf_auc_std": float(valid_aucs.std()) if len(valid_aucs) else np.nan,
            "wf_brier_mean": float(valid_briers.mean()) if len(valid_briers) else np.nan,
        })
        print(f"  rows={len(dataset)} pos={y.mean():.1%} today={prob_today:.1%} "
              f"WF AUC {valid_aucs.mean():.3f}±{valid_aucs.std():.3f}")

    # Ensemble headline: mean of calibrated predictions, spread as crude CI
    current_prob = float(np.mean(per_threshold_preds))
    current_ci_lo = float(np.min(per_threshold_preds))
    current_ci_hi = float(np.max(per_threshold_preds))

    # Historical chart: mean of calibrated predictions across thresholds
    history_df = pd.concat(per_threshold_history, axis=1).mean(axis=1).reset_index()
    history_df.columns = ["date", "crash_prob"]

    avg_coefs = pd.Series(
        np.mean(per_threshold_coefs, axis=0), index=feat_cols
    ).sort_values(key=abs, ascending=False)

    walk_forward = pd.concat(per_threshold_walk, ignore_index=True)
    calibration = _calibration_table(
        pd.Series(all_oos_preds), pd.Series(all_oos_actuals), n_bins=8
    )

    print(f"\nEnsemble today: {current_prob:.1%}  ({current_ci_lo:.1%} – {current_ci_hi:.1%})")
    print(f"Out-of-sample AUC (avg across folds & thresholds): "
          f"{walk_forward['auc'].dropna().mean():.3f} ± {walk_forward['auc'].dropna().std():.3f}")
    print(f"Brier score (avg): {walk_forward['brier'].dropna().mean():.3f}")

    return ModelResult(
        current_prob=current_prob,
        current_ci_lo=current_ci_lo,
        current_ci_hi=current_ci_hi,
        current_date=today_date,
        coefs=avg_coefs,
        prob_history=history_df,
        walk_forward=walk_forward,
        calibration=calibration,
        per_threshold=per_threshold_debug,
    )


if __name__ == "__main__":
    result = build_ensemble()
    print("\nCalibration table (out-of-sample):")
    print(result.calibration.to_string(index=False))
    print("\nWalk-forward folds:")
    print(result.walk_forward.to_string(index=False))
