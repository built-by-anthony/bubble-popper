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


@st.cache_data(ttl=3600)
def load_capex_rate() -> tuple[float, dict]:
    """Return combined annual capex rate and per-company breakdown from most recent filings."""
    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (metric_id) metric_id, obs_date, raw_value
                FROM fact_observation
                WHERE metric_id LIKE '%_capex'
                ORDER BY metric_id, obs_date DESC
            """)
            rows = cur.fetchall()
    by_company = {row[0].replace("_capex", "").upper(): row[2] for row in rows}
    annual_total = sum(by_company.values())
    return annual_total, by_company


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
st.caption("Combined annual capex from MSFT, GOOGL, AMZN, META, and NVDA — the five companies building AI infrastructure. Based on most recent 10-K filings. Ticking up from Jan 1 of the current year.")

annual_rate, by_company = load_capex_rate()
seconds_per_year = 365.25 * 24 * 3600
rate_per_second = annual_rate / seconds_per_year

# YTD seconds elapsed since Jan 1
now = datetime.utcnow()
jan1 = datetime(now.year, 1, 1)
ytd_seconds = (now - jan1).total_seconds()
ytd_spend = ytd_seconds * rate_per_second

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
  .company-item {{
    text-align: center;
  }}
  .company-name {{
    color: #888;
    font-size: 11px;
    letter-spacing: 1px;
  }}
  .company-rate {{
    color: #ccc;
    font-size: 15px;
    font-weight: bold;
  }}
</style>
<div class="clock-wrap">
  <div class="clock-label">AI Infrastructure Spend — Year to Date {now.year}</div>
  <div class="clock-value" id="clock">$0</div>
  <div class="clock-sub">${annual_rate/1e9:.1f}B combined annual rate &nbsp;·&nbsp; ${rate_per_second:,.0f} per second</div>
  <div class="company-row">
    {"".join(f'<div class="company-item"><div class="company-name">{k}</div><div class="company-rate">${v/1e9:.1f}B/yr</div></div>' for k, v in sorted(by_company.items()))}
  </div>
</div>
<script>
  const startValue = {ytd_spend:.2f};
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
""", height=260)

st.divider()

# --- Model Section ---
st.header("Crash Probability Model")
st.caption(
    "Logistic regression trained on FRED macro signals. Predicts the probability of a 30%+ "
    "Nasdaq 100 drawdown occurring within the next 12 months. Trained on data from 1990–present "
    "with a time-ordered 80/20 train/test split to prevent lookahead bias."
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
