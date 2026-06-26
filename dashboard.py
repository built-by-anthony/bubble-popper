import os
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from model import FEATURE_COLS, build_dataset, train, current_signal

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

FRED_METRICS = {
    "yield_curve_10y2y":                  "Yield Curve 10y-2y",
    "yield_curve_10y3m":                  "Yield Curve 10y-3m",
    "treasury_10y":                        "Treasury 10y",
    "fed_funds":                           "Fed Funds Rate",
    "financial_conditions":               "Financial Conditions (NFCI)",
    "financial_conditions_leverage":      "NFCI Leverage",
    "financial_conditions_nonfin_leverage": "NFCI Non-Fin Leverage",
    "m2_money_supply":                    "M2 Money Supply",
    "m2_yoy_growth":                      "M2 YoY Growth %",
    "credit_spreads_baa":                  "Credit Spreads (BAA)",
    "vix":                                "VIX",
}

COMPANY_METRICS = {
    "rd_to_revenue":  "R&D as % of Revenue",
    "net_margin":     "Net Margin %",
    "debt_to_assets": "Debt as % of Assets",
}

COMPANIES = ["msft", "googl", "amzn", "meta", "nvda"]


@st.cache_data(ttl=3600)
def load_model():
    dataset = build_dataset()
    pipe, results = train(dataset)
    prob = current_signal(pipe, dataset)

    features = [c for c in FEATURE_COLS if c in dataset.columns]
    coefs = pd.Series(
        pipe.named_steps["clf"].coef_[0],
        index=features,
    ).sort_values(key=abs, ascending=False)

    # Full probability history from the whole dataset
    all_proba = pipe.predict_proba(dataset[features])[:, 1]
    prob_history = pd.DataFrame({"date": dataset.index, "crash_prob": all_proba})

    return prob, coefs, prob_history


CHATGPT_LAUNCH = datetime(2022, 11, 30)


@st.cache_data(ttl=3600)
def load_capex_rate() -> tuple[float, float, dict]:
    """
    Returns:
      - annual_rate: combined most-recent annual capex across all companies
      - total_since_launch: pro-rated capex spend since ChatGPT launch (Nov 30 2022)
      - by_company: most recent annual capex per company
    """
    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT metric_id, obs_date, raw_value
                FROM fact_observation
                WHERE (metric_id LIKE '%_capex' OR metric_id LIKE '%_rd_expense')
                  AND obs_date >= '2022-01-01'
                ORDER BY metric_id, obs_date ASC
            """)
            rows = cur.fetchall()

    now = datetime.utcnow()

    # Group by company, summing capex + R&D per fiscal year
    by_ticker: dict[str, list] = {}
    rd_by_ticker: dict[str, dict] = {}

    for metric_id, obs_date, value in rows:
        if metric_id.endswith("_capex"):
            ticker = metric_id.replace("_capex", "").upper()
            by_ticker.setdefault(ticker, []).append((obs_date, value))
        elif metric_id.endswith("_rd_expense"):
            ticker = metric_id.replace("_rd_expense", "").upper()
            rd_by_ticker.setdefault(ticker, {})[obs_date] = value

    # Merge R&D into capex series
    for ticker, obs in by_ticker.items():
        rd = rd_by_ticker.get(ticker, {})
        by_ticker[ticker] = [
            (obs_date, value + rd.get(obs_date, 0))
            for obs_date, value in obs
        ]

    total_since_launch = 0.0
    by_company = {}

    for ticker, obs in by_ticker.items():
        obs.sort()
        for i, (obs_date, value) in enumerate(obs):
            fy_end = datetime.combine(obs_date, datetime.min.time())
            prev_date = obs[i - 1][0] if i > 0 else obs_date.replace(year=obs_date.year - 1)
            fy_start = datetime.combine(prev_date, datetime.min.time())

            overlap_start = max(fy_start, CHATGPT_LAUNCH)
            overlap_end = min(fy_end, now)
            if overlap_end <= overlap_start:
                continue

            fy_days = max((fy_end - fy_start).days, 1)
            overlap_days = (overlap_end - overlap_start).days
            total_since_launch += value * (overlap_days / fy_days)

        # Current partial period since last fiscal year end
        last_date, last_value = obs[-1]
        last_dt = datetime.combine(last_date, datetime.min.time())
        if now > last_dt:
            days_elapsed = (now - last_dt).days
            total_since_launch += last_value / 365 * days_elapsed

        by_company[ticker] = obs[-1][1]

    annual_rate = sum(by_company.values())
    return annual_rate, total_since_launch, by_company


@st.cache_data(ttl=3600)
def load_observations(metric_ids: list[str], start: date, end: date) -> pd.DataFrame:
    placeholders = ", ".join(["%s"] * len(metric_ids))
    sql = f"""
        SELECT metric_id, obs_date, raw_value
        FROM fact_observation
        WHERE metric_id IN ({placeholders})
          AND obs_date BETWEEN %s AND %s
        ORDER BY obs_date, source
    """
    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, metric_ids + [start, end])
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["metric_id", "obs_date", "raw_value"])


st.set_page_config(page_title="Bubble Popper", layout="wide")
st.title("Bubble Popper — AI Bubble Signal Dashboard")
st.markdown("""
This dashboard tracks macroeconomic and company fundamental signals to detect and time the AI bubble.
Data updates daily via automated ingestion from **FRED** (Federal Reserve Economic Data) and **SEC EDGAR**.

**How to use:**
- Use the **date range slider** in the sidebar to zoom in on any period
- **Macro Signals** — toggle individual FRED series and normalize to z-scores to compare on the same scale
- **Company Fundamentals** — compare R&D spend, margins, and leverage across the 5 major AI companies
- **Composite Bubble Signal** — a single index averaging all selected macro signals; above zero means conditions are more bubble-like than the historical average
""")

# --- AI Spend Clock ---
st.header("AI Infrastructure Spend Clock")
st.caption("Combined capex + R&D spend from MSFT, GOOGL, AMZN, META, and NVDA since the ChatGPT launch on Nov 30, 2022 — the moment the AI arms race began. Pro-rated from annual 10-K filings. Note: AMZN does not report R&D separately so only capex is included for AMZN. Ticking up in real time at the current annual run rate.")

annual_rate, total_since_launch, by_company = load_capex_rate()
seconds_per_year = 365.25 * 24 * 3600
rate_per_second = annual_rate / seconds_per_year

components.html(f"""
<style>
  .clock-wrap {{
    background: #0e1117;
    border-radius: 12px;
    padding: 32px 40px;
    text-align: center;
    font-family: 'Courier New', monospace;
  }}
  .clock-label {{
    color: #888;
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .clock-sublabel {{
    color: #555;
    font-size: 12px;
    margin-bottom: 16px;
  }}
  .clock-value {{
    color: #ff4b4b;
    font-size: 52px;
    font-weight: bold;
    letter-spacing: 2px;
  }}
  .clock-sub {{
    color: #555;
    font-size: 13px;
    margin-top: 12px;
  }}
  .company-row {{
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 20px;
    flex-wrap: wrap;
  }}
  .company-item {{ text-align: center; }}
  .company-name {{ color: #888; font-size: 11px; letter-spacing: 1px; }}
  .company-rate {{ color: #ccc; font-size: 15px; font-weight: bold; }}
</style>
<div class="clock-wrap">
  <div class="clock-label">AI Investment (Capex + R&D)</div>
  <div class="clock-sublabel">Since ChatGPT Launch — November 30, 2022</div>
  <div class="clock-value" id="clock">$0</div>
  <div class="clock-sub">${annual_rate/1e9:.1f}B combined annual rate &nbsp;·&nbsp; ${rate_per_second:,.0f} per second</div>
  <div class="company-row">
    {"".join(f'<div class="company-item"><div class="company-name">{k}</div><div class="company-rate">${v/1e9:.1f}B/yr</div></div>' for k, v in sorted(by_company.items()))}
  </div>
</div>
<script>
  const startValue = {total_since_launch:.2f};
  const ratePerMs = {rate_per_second / 1000:.6f};
  const startTime = Date.now();

  function fmt(n) {{
    return '$' + Math.floor(n).toLocaleString('en-US');
  }}

  setInterval(() => {{
    const elapsed = Date.now() - startTime;
    const val = startValue + elapsed * ratePerMs;
    document.getElementById('clock').innerText = fmt(val);
  }}, 100);
</script>
""", height=280)

st.divider()

# --- Model Section ---
st.header("Crash Probability Model ⚠️ Experimental")
st.warning(
    "**This model is experimental and should not be used for investment decisions.** "
    "It is a logistic regression trained on only 3–4 historical crash events (dot-com, GFC, 2022). "
    "That is not enough data to build a reliable predictor. The output reflects whether current macro "
    "conditions resemble historical pre-crash periods — not a true probability of a crash occurring. "
    "AUC 0.804 on out-of-sample data, but with very few real events to validate against."
)

with st.spinner("Training model..."):
    crash_prob, coefs, prob_history = load_model()

col_a, col_b = st.columns([1, 3])
with col_a:
    st.metric(
        label="Current Crash Probability",
        value=f"{crash_prob:.1%}",
        help="Probability of a 30%+ NDX drawdown within 12 months, as of today.",
    )
    st.caption(f"As of {date.today()}")

with col_b:
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=prob_history["date"], y=prob_history["crash_prob"],
        mode="lines", name="Crash Probability",
        line=dict(color="crimson", width=1.5),
        fill="tozeroy", fillcolor="rgba(220,20,60,0.08)",
    ))
    fig_prob.add_hline(y=0.3, line_dash="dash", line_color="orange",
                       annotation_text="30% threshold", annotation_position="top left")
    fig_prob.update_layout(
        xaxis_title="Date",
        yaxis_title="Crash Probability",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

st.subheader("Feature Importance")
st.caption("Model coefficients — positive means the feature increases crash probability, negative means it reduces it.")
fig_coef = px.bar(
    coefs.head(12).reset_index(),
    x="index", y=0,
    labels={"index": "Feature", 0: "Coefficient"},
    color=coefs.head(12).values,
    color_continuous_scale=["green", "white", "crimson"],
    color_continuous_midpoint=0,
)
fig_coef.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(t=20))
st.plotly_chart(fig_coef, use_container_width=True)

st.divider()

# --- Sidebar controls ---
st.sidebar.header("Controls")

date_range = st.sidebar.slider(
    "Date range",
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    value=(date(2015, 1, 1), date.today()),
)
start_date, end_date = date_range

# --- FRED Section ---
st.header("Macro Signals (FRED)")
st.caption("Daily and weekly macro data from the Federal Reserve. Select any combination of series. Enable z-score normalization to overlay metrics with different units on the same chart.")

selected_fred = st.multiselect(
    "Select FRED metrics",
    options=list(FRED_METRICS.keys()),
    default=["yield_curve_10y2y", "fed_funds", "m2_yoy_growth", "financial_conditions"],
    format_func=lambda x: FRED_METRICS[x],
)

if selected_fred:
    df_fred = load_observations(selected_fred, start_date, end_date)
    if not df_fred.empty:
        normalize = st.checkbox("Normalize to z-score (compare on same scale)", value=True)

        if normalize:
            def zscore(s):
                return (s - s.mean()) / s.std()
            df_fred["raw_value"] = df_fred.groupby("metric_id")["raw_value"].transform(zscore)
            y_label = "Z-Score"
        else:
            y_label = "Value"

        df_fred["metric_label"] = df_fred["metric_id"].map(FRED_METRICS)
        fig = px.line(
            df_fred, x="obs_date", y="raw_value", color="metric_label",
            labels={"obs_date": "Date", "raw_value": y_label, "metric_label": "Metric"},
        )
        fig.update_layout(legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected range.")

# --- EDGAR Section ---
st.header("Company Fundamentals")
st.caption("Annual figures from SEC 10-K filings for MSFT, GOOGL, AMZN, META, and NVDA. Balance sheet metrics (debt/assets) are quarterly. Note: AMZN does not report R&D as a separate line item.")

col1, col2 = st.columns(2)
with col1:
    selected_companies = st.multiselect(
        "Companies",
        options=COMPANIES,
        default=COMPANIES,
        format_func=str.upper,
    )
with col2:
    selected_edgar_metric = st.selectbox(
        "Metric",
        options=list(COMPANY_METRICS.keys()),
        format_func=lambda x: COMPANY_METRICS[x],
    )

if selected_companies and selected_edgar_metric:
    metric_ids = [f"{c}_{selected_edgar_metric}" for c in selected_companies]
    df_edgar = load_observations(metric_ids, start_date, end_date)

    if not df_edgar.empty:
        df_edgar["company"] = df_edgar["metric_id"].str.split("_").str[0].str.upper()
        fig2 = px.line(
            df_edgar, x="obs_date", y="raw_value", color="company",
            labels={
                "obs_date": "Date",
                "raw_value": COMPANY_METRICS[selected_edgar_metric],
                "company": "Company",
            },
        )
        fig2.update_layout(legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for selected range.")

# --- Composite Signal ---
st.header("Composite Bubble Signal")
st.caption("Z-score average of all selected FRED metrics averaged into a single index. Above zero = conditions are more bubble-like than the historical average. The signal is directional — it shows where we are in the cycle relative to history, not when the bubble pops.")

if selected_fred:
    df_signal = load_observations(selected_fred, start_date, end_date)
    if not df_signal.empty:
        def zscore(s):
            return (s - s.mean()) / s.std()
        df_signal["z"] = df_signal.groupby("metric_id")["raw_value"].transform(zscore)
        composite = (
            df_signal.groupby("obs_date")["z"]
            .mean()
            .reset_index()
            .rename(columns={"z": "signal"})
        )
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=composite["obs_date"], y=composite["signal"],
            mode="lines", name="Bubble Signal",
            line=dict(color="crimson", width=2),
            fill="tozeroy", fillcolor="rgba(220,20,60,0.1)",
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Signal (Z-Score)",
            hovermode="x unified",
        )
        st.plotly_chart(fig3, use_container_width=True)
