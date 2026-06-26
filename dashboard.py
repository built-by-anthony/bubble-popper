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
    "yield_curve_10y2y":                    "Yield Curve 10y-2y",
    "yield_curve_10y3m":                    "Yield Curve 10y-3m",
    "treasury_10y":                          "Treasury 10y",
    "fed_funds":                             "Fed Funds Rate",
    "financial_conditions":                 "Financial Conditions (NFCI)",
    "financial_conditions_leverage":        "NFCI Leverage",
    "financial_conditions_nonfin_leverage": "NFCI Non-Fin Leverage",
    "m2_money_supply":                      "M2 Money Supply",
    "m2_yoy_growth":                        "M2 YoY Growth %",
    "credit_spreads_baa":                    "Credit Spreads (BAA)",
    "vix":                                  "VIX",
    "credit_growth":                        "Credit Growth (TOTLL)",
    "recession_indicator":                  "Recession Indicator (USREC)",
}

COMPANY_METRICS = {
    "rd_to_revenue":  "R&D as % of Revenue",
    "net_margin":     "Net Margin %",
    "debt_to_assets": "Debt as % of Assets",
}

COMPANIES = ["msft", "googl", "amzn", "meta", "nvda"]
CHATGPT_LAUNCH = datetime(2022, 11, 30)
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ── Data loaders ──────────────────────────────────────────────────────────────

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

    all_proba = pipe.predict_proba(dataset[features])[:, 1]
    prob_history = pd.DataFrame({"date": dataset.index, "crash_prob": all_proba})

    return prob, coefs, prob_history


@st.cache_data(ttl=3600)
def load_spend_data():
    """
    Returns capex-only, rd-only, and combined totals since ChatGPT launch,
    plus current annual rates and per-company breakdowns.
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

    capex_series: dict[str, list] = {}
    rd_series: dict[str, dict] = {}

    for metric_id, obs_date, value in rows:
        if metric_id.endswith("_capex"):
            t = metric_id.replace("_capex", "").upper()
            capex_series.setdefault(t, []).append((obs_date, value))
        elif metric_id.endswith("_rd_expense"):
            t = metric_id.replace("_rd_expense", "").upper()
            rd_series.setdefault(t, {})[obs_date] = value

    def since_launch(by_ticker: dict[str, list]) -> tuple[float, dict, float]:
        total = 0.0
        latest = {}
        for ticker, obs in by_ticker.items():
            obs = sorted(obs)
            co_total = 0.0
            for i, (obs_date, value) in enumerate(obs):
                fy_end = datetime.combine(obs_date, datetime.min.time())
                prev = obs[i - 1][0] if i > 0 else obs_date.replace(year=obs_date.year - 1)
                fy_start = datetime.combine(prev, datetime.min.time())
                overlap_start = max(fy_start, CHATGPT_LAUNCH)
                overlap_end = min(fy_end, now)
                if overlap_end <= overlap_start:
                    continue
                fy_days = max((fy_end - fy_start).days, 1)
                co_total += value * ((overlap_end - overlap_start).days / fy_days)
            last_dt = datetime.combine(obs[-1][0], datetime.min.time())
            if now > last_dt:
                co_total += obs[-1][1] / 365 * (now - last_dt).days
            latest[ticker] = obs[-1][1]
            total += co_total
        annual = sum(latest.values())
        return total, latest, annual

    # Capex only
    capex_total, capex_by_co, capex_annual = since_launch(capex_series)

    # R&D only — build same shape as capex_series
    rd_as_series = {t: list(sorted(d.items())) for t, d in rd_series.items()}
    rd_total, rd_by_co, rd_annual = since_launch(rd_as_series)

    # Combined — merge R&D into capex per fiscal year
    combined_series = {}
    for ticker, obs in capex_series.items():
        rd = rd_series.get(ticker, {})
        combined_series[ticker] = [(d, v + rd.get(d, 0)) for d, v in obs]
    combined_total, combined_by_co, combined_annual = since_launch(combined_series)

    return {
        "capex":    {"total": capex_total,    "annual": capex_annual,    "by_co": capex_by_co},
        "rd":       {"total": rd_total,       "annual": rd_annual,       "by_co": rd_by_co},
        "combined": {"total": combined_total, "annual": combined_annual, "by_co": combined_by_co},
    }


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


# ── Clock component ───────────────────────────────────────────────────────────

def render_clock(label: str, total: float, annual: float, by_co: dict, clock_id: str):
    rate_per_sec = annual / SECONDS_PER_YEAR
    company_html = "".join(
        f'<div class="company-item">'
        f'<div class="company-name">{k}</div>'
        f'<div class="company-rate">${v/1e9:.1f}B/yr</div>'
        f'</div>'
        for k, v in sorted(by_co.items())
    )
    components.html(f"""
<style>
  .clock-wrap-{clock_id} {{
    background: #0e1117;
    border-radius: 10px;
    padding: 24px 20px;
    text-align: center;
    font-family: 'Courier New', monospace;
  }}
  .clock-label-{clock_id} {{
    color: #888;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
  .clock-sublabel-{clock_id} {{
    color: #444;
    font-size: 10px;
    margin-bottom: 12px;
  }}
  .clock-value-{clock_id} {{
    color: #ff4b4b;
    font-size: 28px;
    font-weight: bold;
    letter-spacing: 1px;
  }}
  .clock-rate-{clock_id} {{
    color: #555;
    font-size: 11px;
    margin-top: 8px;
  }}
  .company-row-{clock_id} {{
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
    flex-wrap: wrap;
  }}
  .company-item {{ text-align: center; }}
  .company-name {{ color: #666; font-size: 9px; letter-spacing: 1px; }}
  .company-rate {{ color: #aaa; font-size: 11px; font-weight: bold; }}
</style>
<div class="clock-wrap-{clock_id}">
  <div class="clock-label-{clock_id}">{label}</div>
  <div class="clock-sublabel-{clock_id}">Since ChatGPT Launch — Nov 30, 2022</div>
  <div class="clock-value-{clock_id}" id="{clock_id}">$0</div>
  <div class="clock-rate-{clock_id}">${annual/1e9:.1f}B/yr &nbsp;·&nbsp; ${rate_per_sec:,.0f}/sec</div>
  <div class="company-row-{clock_id}">{company_html}</div>
</div>
<script>
  const startValue_{clock_id} = {total:.2f};
  const ratePerMs_{clock_id} = {rate_per_sec / 1000:.6f};
  const startTime_{clock_id} = Date.now();
  function fmt_{clock_id}(n) {{ return '$' + Math.floor(n).toLocaleString('en-US'); }}
  setInterval(() => {{
    const val = startValue_{clock_id} + (Date.now() - startTime_{clock_id}) * ratePerMs_{clock_id};
    document.getElementById('{clock_id}').innerText = fmt_{clock_id}(val);
  }}, 100);
</script>
""", height=220)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bubble Popper", layout="wide")
st.title("Bubble Popper — AI Bubble Signal Dashboard")
st.caption("Tracking macroeconomic and company fundamental signals to detect and time the AI bubble. Data updates daily via FRED and SEC EDGAR.")

tab_overview, tab_macro, tab_companies, tab_model = st.tabs([
    "Overview", "Macro Signals", "Company Fundamentals", "Model"
])

# ── Overview tab ──────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("AI Investment Since ChatGPT Launch")
    st.caption("Combined spend from MSFT, GOOGL, AMZN, META, and NVDA. Pro-rated from annual 10-K filings. AMZN does not report R&D separately — capex only for AMZN.")

    spend = load_spend_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_clock("Infrastructure (Capex)", spend["capex"]["total"], spend["capex"]["annual"], spend["capex"]["by_co"], "capex")
    with col2:
        render_clock("Research & Development", spend["rd"]["total"], spend["rd"]["annual"], spend["rd"]["by_co"], "rd")
    with col3:
        render_clock("Total (Capex + R&D)", spend["combined"]["total"], spend["combined"]["annual"], spend["combined"]["by_co"], "combined")

    st.divider()

    st.subheader("Crash Probability")
    st.warning(
        "**Experimental — not investment advice.** Logistic regression trained on 3–4 historical crash events. "
        "Reflects whether current macro conditions resemble historical pre-crash periods."
    )
    with st.spinner("Training model..."):
        crash_prob, coefs, prob_history = load_model()

    _, m_col, _ = st.columns([1, 2, 1])
    with m_col:
        st.markdown(f"""
<div style="text-align:center; padding: 32px 0 16px 0;">
  <div style="color:#888; font-size:13px; letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;">
    Probability of 20%+ Nasdaq Crash in Next 12 Months
  </div>
  <div style="color:#ff4b4b; font-size:80px; font-weight:bold; line-height:1; font-family:'Courier New',monospace;">
    {crash_prob:.1%}
  </div>
  <div style="color:#555; font-size:12px; margin-top:12px;">As of {date.today()} &nbsp;·&nbsp; Experimental — not investment advice</div>
</div>
""", unsafe_allow_html=True)


# ── Macro Signals tab ─────────────────────────────────────────────────────────
with tab_macro:
    st.subheader("Macro Signals (FRED)")
    st.caption("Daily, weekly, and monthly macro data from the Federal Reserve. Enable z-score normalization to overlay metrics with different units on the same chart.")

    date_range = st.slider(
        "Date range",
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        value=(date(2015, 1, 1), date.today()),
    )
    start_date, end_date = date_range

    selected_fred = st.multiselect(
        "Select FRED metrics",
        options=list(FRED_METRICS.keys()),
        default=["yield_curve_10y2y", "fed_funds", "m2_yoy_growth", "financial_conditions"],
        format_func=lambda x: FRED_METRICS[x],
    )

    if selected_fred:
        df_fred = load_observations(selected_fred, start_date, end_date)
        if not df_fred.empty:
            normalize = st.checkbox("Normalize to z-score", value=True)
            if normalize:
                def zscore(s):
                    return (s - s.mean()) / s.std()
                df_fred["raw_value"] = df_fred.groupby("metric_id")["raw_value"].transform(zscore)
                y_label = "Z-Score"
            else:
                y_label = "Value"

            df_fred["metric_label"] = df_fred["metric_id"].map(FRED_METRICS)
            fig = px.line(df_fred, x="obs_date", y="raw_value", color="metric_label",
                          labels={"obs_date": "Date", "raw_value": y_label, "metric_label": "Metric"})
            fig.update_layout(legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Composite Bubble Signal")
        st.caption("Z-score average of selected metrics. Above zero = conditions more bubble-like than historical average.")
        df_signal = load_observations(selected_fred, start_date, end_date)
        if not df_signal.empty:
            def zscore(s):
                return (s - s.mean()) / s.std()
            df_signal["z"] = df_signal.groupby("metric_id")["raw_value"].transform(zscore)
            composite = df_signal.groupby("obs_date")["z"].mean().reset_index().rename(columns={"z": "signal"})
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=composite["obs_date"], y=composite["signal"],
                mode="lines", name="Bubble Signal",
                line=dict(color="crimson", width=2),
                fill="tozeroy", fillcolor="rgba(220,20,60,0.1)",
            ))
            fig3.add_hline(y=0, line_dash="dash", line_color="gray")
            fig3.update_layout(xaxis_title="Date", yaxis_title="Signal (Z-Score)", hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)


# ── Company Fundamentals tab ──────────────────────────────────────────────────
with tab_companies:
    st.subheader("Company Fundamentals")
    st.caption("Annual figures from SEC 10-K filings for MSFT, GOOGL, AMZN, META, and NVDA. Balance sheet metrics are quarterly. AMZN does not report R&D separately.")

    date_range_co = st.slider(
        "Date range",
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        value=(date(2010, 1, 1), date.today()),
        key="co_date",
    )
    start_co, end_co = date_range_co

    col1, col2 = st.columns(2)
    with col1:
        selected_companies = st.multiselect(
            "Companies", options=COMPANIES, default=COMPANIES, format_func=str.upper,
        )
    with col2:
        selected_metric = st.selectbox(
            "Metric", options=list(COMPANY_METRICS.keys()), format_func=lambda x: COMPANY_METRICS[x],
        )

    if selected_companies and selected_metric:
        metric_ids = [f"{c}_{selected_metric}" for c in selected_companies]
        df_edgar = load_observations(metric_ids, start_co, end_co)
        if not df_edgar.empty:
            df_edgar["company"] = df_edgar["metric_id"].str.split("_").str[0].str.upper()
            fig2 = px.line(df_edgar, x="obs_date", y="raw_value", color="company",
                           labels={"obs_date": "Date", "raw_value": COMPANY_METRICS[selected_metric], "company": "Company"})
            fig2.update_layout(legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for selected range.")


# ── Model tab ─────────────────────────────────────────────────────────────────
with tab_model:
    st.subheader("Crash Probability Model")
    st.warning(
        "**Experimental — not investment advice.** Logistic regression trained on 3–4 historical crash events (dot-com, GFC, 2022). "
        "Output reflects whether current macro conditions resemble historical pre-crash periods. AUC 0.669 out-of-sample."
    )

    with st.spinner("Training model..."):
        crash_prob, coefs, prob_history = load_model()

    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.metric("NDX 20%+ Drawdown in 12mo", f"{crash_prob:.1%}")
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
            xaxis_title="Date", yaxis_title="Crash Probability",
            yaxis_tickformat=".0%", hovermode="x unified", margin=dict(t=20),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("Feature Importance")
    st.caption("Model coefficients — positive increases crash probability, negative reduces it.")
    fig_coef = px.bar(
        coefs.head(12).reset_index(), x="index", y=0,
        labels={"index": "Feature", 0: "Coefficient"},
        color=coefs.head(12).values,
        color_continuous_scale=["green", "white", "crimson"],
        color_continuous_midpoint=0,
    )
    fig_coef.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(t=20))
    st.plotly_chart(fig_coef, use_container_width=True)
