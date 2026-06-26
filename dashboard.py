import os
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from dotenv import load_dotenv

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
}

EDGAR_METRICS = {
    "rd_to_revenue":   "R&D as % of Revenue",
    "net_margin":      "Net Margin %",
    "debt_to_assets":  "Debt as % of Assets",
    "free_cash_flow":  "Free Cash Flow ($)",
}

COMPANIES = ["msft", "googl", "amzn", "meta", "nvda"]


@st.cache_data(ttl=3600)
def load_observations(metric_ids: list[str], start: date, end: date) -> pd.DataFrame:
    placeholders = ", ".join(["%s"] * len(metric_ids))
    sql = f"""
        SELECT metric_id, obs_date, raw_value
        FROM fact_observation
        WHERE metric_id IN ({placeholders})
          AND obs_date BETWEEN %s AND %s
        ORDER BY obs_date
    """
    with psycopg.connect(DATABASE_URL, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, metric_ids + [start, end])
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["metric_id", "obs_date", "raw_value"])


st.set_page_config(page_title="Bubble Popper", layout="wide")
st.title("Bubble Popper — AI Bubble Signal Dashboard")

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
st.header("Company Fundamentals (EDGAR)")

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
        options=list(EDGAR_METRICS.keys()),
        format_func=lambda x: EDGAR_METRICS[x],
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
                "raw_value": EDGAR_METRICS[selected_edgar_metric],
                "company": "Company",
            },
        )
        fig2.update_layout(legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for selected range.")

# --- Composite Signal ---
st.header("Composite Bubble Signal")
st.caption("Z-score average of all selected FRED metrics. Higher = more bubble-like conditions.")

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
