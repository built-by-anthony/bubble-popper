import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from model import build_ensemble
from regime import build_regime

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
    "shiller_cape":                         "Shiller CAPE",
}

COMPANY_METRICS = {
    "rd_to_revenue":  "R&D as % of Revenue",
    "net_margin":     "Net Margin %",
    "debt_to_assets": "Debt as % of Assets",
}

COMPANIES = ["msft", "googl", "amzn", "meta", "nvda"]
CHATGPT_LAUNCH = datetime(2022, 11, 30)
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Major equity drawdowns to shade on the regime history chart — provides
# visual context for whether the score actually peaked before each crash.
CRASH_PERIODS = [
    ("2000-03-01", "2002-10-01", "Dot-com"),
    ("2007-10-01", "2009-03-01", "GFC"),
    ("2020-02-01", "2020-04-01", "COVID"),
    ("2022-01-01", "2022-10-01", "Rate Shock"),
]

# Plain-language descriptions for the breakdown tab. Keys match Component.label.
COMPONENT_DESCRIPTIONS = {
    "CAPE":                  "Shiller P/E — how expensive stocks are vs. 10-year average earnings",
    "M2 YoY Growth":         "How fast money supply is expanding — fuels asset prices",
    "Fed Funds (inv)":       "Fed funds rate (inverted) — cheap money encourages risk-taking",
    "Credit Spreads (inv)":  "Corporate–Treasury spread (inverted) — tight spreads signal complacency",
    "Credit Growth YoY":     "How fast bank loans are growing — measures leverage buildup",
    "Yield Curve (inv)":     "10y–2y Treasury spread (inverted) — flattening signals late cycle",
    "VIX (inv)":             "Volatility index (inverted) — low VIX = market complacency",
    "Hyperscaler Capex YoY": "Combined annual capex growth for the 5 AI hyperscalers",
}

PILLAR_DESCRIPTIONS = {
    "Valuation":      "Are stocks expensive vs. fundamentals?",
    "Liquidity":      "Is money cheap and abundant?",
    "Credit":         "Are lenders cautious or aggressive?",
    "Term Structure": "Is the yield curve signaling late-cycle?",
    "Sentiment":      "How complacent or fearful is the market?",
    "AI Spend":       "Is AI investment accelerating beyond trend?",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_regime():
    return build_regime()


@st.cache_data(ttl=3600)
def load_model():
    return build_ensemble()


@st.cache_data(ttl=3600)
def load_spend_data():
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
        return total, latest, sum(latest.values())

    capex_total, capex_by_co, capex_annual = since_launch(capex_series)
    rd_as_series = {t: list(sorted(d.items())) for t, d in rd_series.items()}
    rd_total, rd_by_co, rd_annual = since_launch(rd_as_series)
    combined_series = {
        t: [(d, v + rd_series.get(t, {}).get(d, 0)) for d, v in obs]
        for t, obs in capex_series.items()
    }
    combined_total, combined_by_co, combined_annual = since_launch(combined_series)

    return {
        "capex":    {"total": capex_total,    "annual": capex_annual,    "by_co": capex_by_co},
        "rd":       {"total": rd_total,       "annual": rd_annual,       "by_co": rd_by_co},
        "combined": {"total": combined_total, "annual": combined_annual, "by_co": combined_by_co},
    }


@st.cache_data(ttl=86400)
def load_sp500_history() -> pd.DataFrame:
    """Daily S&P 500 close, used to compute 12-month forward returns from
    historical analog dates. Cached for 24h — full-history fetch is ~2s."""
    import yfinance as yf
    hist = yf.Ticker("^GSPC").history(period="max")
    hist.index = hist.index.tz_localize(None).normalize()
    return hist[["Close"]].rename(columns={"Close": "price"})


def _forward_return(prices: pd.DataFrame, start_date, months: int = 12) -> float | None:
    """Total S&P return from start_date to start_date + N months. Returns None
    if either endpoint falls outside the available price history."""
    if prices.empty:
        return None
    start = pd.Timestamp(start_date)
    end = start + pd.DateOffset(months=months)
    if end > prices.index.max() or start < prices.index.min():
        return None
    p_start = prices.loc[:start].iloc[-1, 0]
    p_end = prices.loc[:end].iloc[-1, 0]
    return (p_end - p_start) / p_start


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


# ── Render helpers ────────────────────────────────────────────────────────────

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
  .clock-wrap-{clock_id} {{ background:#0e1117; border-radius:10px; padding:24px 20px; text-align:center; font-family:'Courier New',monospace; }}
  .clock-label-{clock_id} {{ color:#888; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }}
  .clock-sublabel-{clock_id} {{ color:#444; font-size:10px; margin-bottom:12px; }}
  .clock-value-{clock_id} {{ color:#ff4b4b; font-size:28px; font-weight:bold; letter-spacing:1px; }}
  .clock-rate-{clock_id} {{ color:#555; font-size:11px; margin-top:8px; }}
  .company-row-{clock_id} {{ display:flex; justify-content:center; gap:12px; margin-top:12px; flex-wrap:wrap; }}
  .company-item {{ text-align:center; }}
  .company-name {{ color:#666; font-size:9px; letter-spacing:1px; }}
  .company-rate {{ color:#aaa; font-size:11px; font-weight:bold; }}
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


# Hand-curated historical era labels so analog dates read like sentences instead
# of bare months. Order matters — first matching range wins. Anything not in this
# list falls back to "Month Year".
HISTORICAL_ERAS = [
    ((1999,  1), (2000,  3),  "the dot-com bubble peak"),
    ((2000,  4), (2002, 10),  "the dot-com crash"),
    ((2003,  1), (2006, 12),  "the housing-bubble buildup"),
    ((2007,  1), (2007, 12),  "the pre-GFC peak"),
    ((2008,  1), (2009,  6),  "the Global Financial Crisis"),
    ((2011,  6), (2011, 12),  "the European debt crisis"),
    ((2015,  6), (2016,  6),  "the late-2015 oil/China scare"),
    ((2018,  1), (2018, 12),  "the late-2018 rate-hike scare"),
    ((2020,  3), (2020,  5),  "the COVID crash bottom"),
    ((2020,  6), (2021,  6),  "the post-COVID everything-bubble"),
    ((2021,  7), (2021, 12),  "the late-2021 meme/crypto peak"),
    ((2022,  1), (2022, 12),  "the 2022 rate-shock unwind"),
]


def _era_label(d) -> str:
    for (y0, m0), (y1, m1), label in HISTORICAL_ERAS:
        if (d.year, d.month) >= (y0, m0) and (d.year, d.month) <= (y1, m1):
            return label
    return d.strftime("%B %Y")


def _percentile_framing(r) -> tuple[str, str]:
    """Honest percentile display. When today is technically at the all-time max
    but only by a tiny margin (within 0.05σ of the prior peak), display as
    "99th+ percentile" and add a contextual note naming the tied prior peak.
    Otherwise show the plain integer percentile with no extra context.
    Prevents the "100th percentile" framing from overclaiming uniqueness when
    we're really tied within measurement noise."""
    pct = r.percentile
    if pct < 99.5:
        return (f"{pct:.0f}th percentile", "")

    # Locate the prior peak, matching the analog selection rule:
    # modern era only, excluding the last 18 months.
    history = r.history.copy()
    history["date"] = pd.to_datetime(history["date"])
    cutoff = history["date"].max() - pd.DateOffset(months=18)
    prior = history[(history["date"] >= "1990-01-01") & (history["date"] <= cutoff)]
    if prior.empty:
        return ("99th+ percentile", "")

    prior_peak = prior.loc[prior["score"].idxmax()]
    gap = r.current_score - float(prior_peak["score"])
    if gap > 0.05:
        return (f"{pct:.0f}th percentile", "")

    era = _era_label(prior_peak["date"])
    return (
        "99th+ percentile",
        f"essentially tied with {era} (peak {float(prior_peak['score']):+.2f}σ in {prior_peak['date'].strftime('%b %Y')})",
    )


def _verdict_palette(pct: float) -> dict:
    """Returns verdict word, primary color, and gauge step colors based on percentile."""
    if pct >= 90:
        return {"verdict": "EXTREME", "color": "#ff4b4b",
                "blurb": "Conditions in the top 10% of bubble-like periods on record"}
    if pct >= 75:
        return {"verdict": "ELEVATED", "color": "#ff8c00",
                "blurb": "Conditions more bubble-like than 3 in 4 historical months"}
    if pct >= 50:
        return {"verdict": "MODERATE", "color": "#d4a017",
                "blurb": "Modestly above the historical average"}
    if pct >= 25:
        return {"verdict": "CALM", "color": "#5dade2",
                "blurb": "Below the historical average"}
    return {"verdict": "SUBDUED", "color": "#2ecc71",
            "blurb": "Among the calmest periods on record"}


def render_regime_hero(r):
    """Verdict-first hero with side-by-side gauge for non-technical readers."""
    pct = r.percentile
    score = r.current_score
    p = _verdict_palette(pct)

    top_analog = r.analogs.iloc[0]
    top_label = _era_label(top_analog["date"])
    second_analog = r.analogs.iloc[1] if len(r.analogs) > 1 else None
    second_label = _era_label(second_analog["date"]) if second_analog is not None else None

    # Headline sentence — same era as the next analog → just one phrase, else two.
    if second_label and second_label != top_label:
        sentence = f"resembles <strong style='color:#eee;'>{top_label}</strong> and <strong style='color:#eee;'>{second_label}</strong>"
    else:
        sentence = f"most resembles <strong style='color:#eee;'>{top_label}</strong>"

    top_contributor = r.components_now.iloc[0]
    driver = f"{top_contributor['label']} ({top_contributor['zscore']:+.1f}σ)"
    pct_display, pct_note = _percentile_framing(r)
    note_html = (
        f'<div style="color:#666; font-size:11px; margin-top:6px; font-style:italic;">'
        f'Note: {pct_note}.</div>'
        if pct_note else ""
    )

    col_text, col_gauge = st.columns([3, 2])

    with col_text:
        st.markdown(f"""
<div style="padding:24px 8px 8px 0;">
  <div style="color:#888; font-size:11px; letter-spacing:3px; text-transform:uppercase; margin-bottom:6px;">
    AI Bubble Regime
  </div>
  <div style="color:{p['color']}; font-size:64px; font-weight:800; line-height:1; letter-spacing:2px;">
    {p['verdict']}
  </div>
  <div style="color:#bbb; font-size:18px; line-height:1.5; margin-top:14px;">
    Today's macro and valuation conditions {sentence}.
  </div>
  <div style="color:#888; font-size:13px; margin-top:12px;">
    {p['blurb']} — <strong style="color:#aaa;">{pct_display}</strong> since 1990.
  </div>
  <div style="color:#555; font-size:11px; margin-top:18px; font-family:'Courier New',monospace;">
    Composite score {score:+.2f}σ · driven by {driver} · as of {r.current_date}
  </div>
  {note_html}
</div>
""", unsafe_allow_html=True)

    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 38, "color": p["color"]}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 90, 100],
                    "ticktext": ["Subdued", "Calm", "Avg", "Elevated", "Extreme", ""],
                    "tickfont": {"size": 10, "color": "#888"},
                },
                "bar": {"color": p["color"], "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],   "color": "rgba(46,204,113,0.28)"},
                    {"range": [25, 50],  "color": "rgba(93,173,226,0.22)"},
                    {"range": [50, 75],  "color": "rgba(212,160,23,0.28)"},
                    {"range": [75, 90],  "color": "rgba(255,140,0,0.35)"},
                    {"range": [90, 100], "color": "rgba(255,75,75,0.45)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": pct,
                },
            },
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#aaa"},
            height=240,
            margin=dict(t=20, b=10, l=10, r=10),
        )
        st.plotly_chart(gauge, use_container_width=True)


def render_regime_explainer():
    with st.expander("How to read this", expanded=False):
        st.markdown("""
**What this is.** A single score combining 8 widely-watched macro and valuation signals
— things like the Shiller CAPE ratio, the yield curve, credit spreads, the VIX, and AI
spending. Each signal is z-scored (compared to its own history) and sign-adjusted so
that "positive = more bubble-like."

**How to read the score.**
- Above zero = conditions more bubble-like than the historical average.
- Above the dashed line on the history chart = top 25% of all readings.
- The percentile tells you where today sits among every monthly reading since 1990.

**What this is *not*.** A forecast. It does not predict when (or whether) a crash will happen.
It measures *where we are in the cycle* relative to the historical record. Past similar
readings sometimes ended in crashes within 1–2 years (1999, 2007, 2021) and sometimes
didn't (1996, 2017). The closest historical analogs are listed in the Regime Breakdown tab.

**Why we replaced the old "crash probability."** Walk-forward cross-validation showed
that with only ~11 distinct historical crashes in the data, no probabilistic forecaster
can be rigorously fit. The honest output is regime measurement, not probability.
The original modeling work is preserved in the Experimental Model tab.
""")


def render_regime_history_chart(r):
    df = r.history.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()

    # Shade crash periods
    for start, end, label in CRASH_PERIODS:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255,75,75,0.10)",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation=dict(font=dict(size=10, color="#888")),
        )

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["score"],
        mode="lines", name="Regime Score",
        line=dict(color="crimson", width=1.7),
        fill="tozeroy", fillcolor="rgba(220,20,60,0.08)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_hline(y=r.current_score, line_dash="dot", line_color="orange", line_width=1,
                  annotation_text="current", annotation_position="right")

    fig.update_layout(
        height=380,
        xaxis_title="",
        yaxis_title="Regime Score (σ)",
        hovermode="x unified",
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bubble Popper", layout="wide")
st.title("Bubble Popper — AI Bubble Signal Dashboard")
st.caption(
    "A transparent regime indicator for the AI bubble. Built from public FRED, SEC EDGAR, "
    "and Shiller CAPE data updated daily. Not a forecast — a measurement of where current "
    "macro and valuation conditions sit relative to history."
)

tab_overview, tab_regime, tab_macro, tab_companies, tab_model = st.tabs([
    "Overview", "Regime Breakdown", "Macro Signals", "Company Fundamentals", "Experimental Model"
])

# ── Overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    with st.spinner("Computing regime score..."):
        r = load_regime()

    render_regime_hero(r)
    render_regime_explainer()

    st.divider()

    st.subheader("AI Investment Since ChatGPT Launch")
    st.caption(
        "Combined spend from MSFT, GOOGL, AMZN, META, and NVDA. Pro-rated from annual 10-K "
        "filings. AMZN does not report R&D separately — capex only for AMZN."
    )

    spend = load_spend_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        render_clock("Infrastructure (Capex)", spend["capex"]["total"], spend["capex"]["annual"], spend["capex"]["by_co"], "capex")
    with col2:
        render_clock("Research & Development", spend["rd"]["total"], spend["rd"]["annual"], spend["rd"]["by_co"], "rd")
    with col3:
        render_clock("Total (Capex + R&D)", spend["combined"]["total"], spend["combined"]["annual"], spend["combined"]["by_co"], "combined")


# ── Regime Breakdown ──────────────────────────────────────────────────────────
with tab_regime:
    with st.spinner("Computing regime score..."):
        r = load_regime()
    sp500 = load_sp500_history()

    # ── Score history ────────────────────────────────────────────────────────
    st.subheader("Where we are vs. history")
    st.caption(
        "The composite score over time. Shaded periods are the four major equity "
        "drawdowns since 2000. Notice the score was elevated heading into each — "
        "but elevated readings don't always lead to a crash (1996, 2017). "
        "**This measures regime, not timing.**"
    )
    render_regime_history_chart(r)

    st.divider()

    # ── Pillar contributions ────────────────────────────────────────────────
    st.subheader("What's driving the score")
    st.caption(
        "The composite is built from six pillars. Each bar shows that pillar's "
        "average contribution today — positive means it's pushing the score up "
        "(more bubble-like), negative means it's holding the score down."
    )

    pcol_chart, pcol_legend = st.columns([3, 2])
    with pcol_chart:
        pillar_df = r.pillar_scores.copy()
        fig_p = px.bar(
            pillar_df, x="score", y="pillar", orientation="h",
            color="score",
            color_continuous_scale=["#2ecc71", "#dddddd", "#ff4b4b"],
            color_continuous_midpoint=0,
            labels={"score": "Contribution (σ)", "pillar": ""},
        )
        fig_p.update_layout(
            height=300, showlegend=False, coloraxis_showscale=False, margin=dict(t=10, b=10),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with pcol_legend:
        legend_rows = "".join(
            f'<div style="margin:6px 0;">'
            f'<div style="color:#ccc; font-weight:600; font-size:13px;">{p}</div>'
            f'<div style="color:#888; font-size:12px;">{desc}</div>'
            f'</div>'
            for p, desc in PILLAR_DESCRIPTIONS.items()
        )
        st.markdown(
            f'<div style="padding:8px 4px;">{legend_rows}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Component detail ─────────────────────────────────────────────────────
    st.subheader("Component detail")
    st.caption(
        "The eight signals feeding the score. \"Latest\" is the current raw "
        "reading, \"Contribution\" is the sign-adjusted z-score that goes into "
        "the composite. Bigger absolute number = stronger pull on the score."
    )
    detail = r.components_now.copy()
    detail["description"] = detail["label"].map(COMPONENT_DESCRIPTIONS).fillna("")
    detail["raw_value"] = detail["raw_value"].apply(
        lambda v: f"{v:,.2f}" if abs(v) < 1e6 else f"${v/1e9:,.1f}B"
    )
    detail["zscore"] = detail["zscore"].apply(lambda v: f"{v:+.2f}σ")
    detail = detail.rename(columns={
        "label": "Signal", "description": "What it measures", "pillar": "Pillar",
        "raw_value": "Latest", "zscore": "Contribution",
    })[["Signal", "What it measures", "Pillar", "Latest", "Contribution"]]
    st.dataframe(detail, use_container_width=True, hide_index=True)

    st.divider()

    # ── Historical analogs ──────────────────────────────────────────────────
    st.subheader("When the score last looked like this")
    st.caption(
        "Prior months whose composite score most closely matches today's. "
        "\"What happened next\" is the **actual** S&P 500 total price change in the "
        "12 months that followed — historical context, not a prediction. "
        "Limited to 1990+ for fair comparison; recent dates without a full "
        "12-month follow-up are excluded."
    )
    analog_df = r.analogs.copy()
    analog_df["era"] = analog_df["date"].apply(_era_label)
    analog_df["fwd_12mo"] = analog_df["date"].apply(
        lambda d: _forward_return(sp500, d, months=12)
    )
    analog_df["fwd_24mo"] = analog_df["date"].apply(
        lambda d: _forward_return(sp500, d, months=24)
    )
    # Keep rows where at least 12mo forward is computable
    analog_df = analog_df[analog_df["fwd_12mo"].notna()].copy()

    def fmt_return(x):
        if pd.isna(x):
            return "—"
        return f"{x*100:+.1f}%"

    display = analog_df.copy()
    display["Month"] = display["date"].apply(lambda d: pd.Timestamp(d).strftime("%b %Y"))
    display["Period"] = display["era"]
    display["Score then"] = display["score"].apply(lambda v: f"{v:+.2f}σ")
    display["S&P next 12mo"] = display["fwd_12mo"].apply(fmt_return)
    display["S&P next 24mo"] = display["fwd_24mo"].apply(fmt_return)
    display = display[["Month", "Period", "Score then", "S&P next 12mo", "S&P next 24mo"]]
    st.dataframe(display, use_container_width=True, hide_index=True)

    if len(analog_df) >= 3:
        avg_12 = analog_df["fwd_12mo"].mean() * 100
        valid_24 = analog_df["fwd_24mo"].dropna()
        if len(valid_24) > 0:
            avg_24 = valid_24.mean() * 100
            min_24 = valid_24.min() * 100
            max_24 = valid_24.max() * 100
            st.caption(
                f"**Across these analogs, the S&P 500 averaged {avg_12:+.1f}% over the next 12 months "
                f"and {avg_24:+.1f}% over the next 24 months** (24mo range: {min_24:+.1f}% to {max_24:+.1f}%). "
                "Notice the 12-month vs. 24-month gap — late-stage bubbles often keep inflating for "
                "another year before correcting. That's the pattern you're seeing in the table. "
                "*Past patterns are not predictions.*"
            )


# ── Macro Signals ─────────────────────────────────────────────────────────────
with tab_macro:
    st.subheader("Macro Signals (FRED)")
    st.caption(
        "Raw macro data feeding the regime score plus additional context series. "
        "Enable z-score normalization to overlay metrics with different units."
    )

    date_range = st.slider(
        "Date range", min_value=date(2000, 1, 1), max_value=date.today(),
        value=(date(2015, 1, 1), date.today()),
    )
    start_date, end_date = date_range

    selected_fred = st.multiselect(
        "Select FRED metrics",
        options=list(FRED_METRICS.keys()),
        default=["yield_curve_10y2y", "fed_funds", "m2_yoy_growth", "shiller_cape"],
        format_func=lambda x: FRED_METRICS[x],
    )

    if selected_fred:
        df_fred = load_observations(selected_fred, start_date, end_date)
        if not df_fred.empty:
            normalize = st.checkbox("Normalize to z-score", value=True)
            if normalize:
                def _z(s):
                    return (s - s.mean()) / s.std()
                df_fred["raw_value"] = df_fred.groupby("metric_id")["raw_value"].transform(_z)
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


# ── Company Fundamentals ──────────────────────────────────────────────────────
with tab_companies:
    st.subheader("Company Fundamentals")
    st.caption(
        "Annual figures from SEC 10-K filings for MSFT, GOOGL, AMZN, META, and NVDA. "
        "Balance sheet metrics are quarterly. AMZN does not report R&D separately."
    )

    date_range_co = st.slider(
        "Date range", min_value=date(2000, 1, 1), max_value=date.today(),
        value=(date(2010, 1, 1), date.today()), key="co_date",
    )
    start_co, end_co = date_range_co

    col1, col2 = st.columns(2)
    with col1:
        selected_companies = st.multiselect(
            "Companies", options=COMPANIES, default=COMPANIES, format_func=str.upper,
        )
    with col2:
        selected_metric = st.selectbox(
            "Metric", options=list(COMPANY_METRICS.keys()),
            format_func=lambda x: COMPANY_METRICS[x],
        )

    if selected_companies and selected_metric:
        metric_ids = [f"{c}_{selected_metric}" for c in selected_companies]
        df_edgar = load_observations(metric_ids, start_co, end_co)
        if not df_edgar.empty:
            df_edgar["company"] = df_edgar["metric_id"].str.split("_").str[0].str.upper()
            fig2 = px.line(
                df_edgar, x="obs_date", y="raw_value", color="company",
                labels={"obs_date": "Date", "raw_value": COMPANY_METRICS[selected_metric], "company": "Company"},
            )
            fig2.update_layout(legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for selected range.")


# ── Experimental Model ────────────────────────────────────────────────────────
with tab_model:
    st.subheader("Calibrated Crash Probability — Experimental")
    st.error(
        "**This panel is for transparency, not for use. Walk-forward cross-validation "
        "shows the model is barely better than random.** With only ~11 distinct historical "
        "crash episodes in the training data, no probabilistic crash forecaster can be "
        "rigorously fit. The headline regime score on the Overview tab is the right way "
        "to read current conditions. This is preserved so anyone curious can see exactly "
        "where the modeling work landed."
    )

    with st.spinner("Training calibrated ensemble..."):
        result = load_model()

    valid_aucs = result.walk_forward["auc"].dropna()
    valid_briers = result.walk_forward["brier"].dropna()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Today's calibrated probability",
                  f"{result.current_prob:.0%}",
                  help="Ensemble mean across 15/20/25% drawdown thresholds.")
        st.caption(f"Range across ensemble: {result.current_ci_lo:.0%} – {result.current_ci_hi:.0%}")
    with col_b:
        st.metric("Walk-forward AUC",
                  f"{valid_aucs.mean():.2f} ± {valid_aucs.std():.2f}",
                  help="0.50 = random. ~0.70 = useful. Higher std = unstable.")
        st.caption(f"Across {len(valid_aucs)} valid folds (out of {len(result.walk_forward)})")
    with col_c:
        st.metric("Brier score", f"{valid_briers.mean():.3f}",
                  help="Lower is better. Compare to predict-base-rate baseline.")
        st.caption("Mean squared error of calibrated probabilities")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Walk-Forward Folds")
        st.caption("Each fold trains on past, tests on the next chunk. NaN = test set contained zero crashes.")
        wf = result.walk_forward.copy()
        wf["auc"] = wf["auc"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        wf["brier"] = wf["brier"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        wf["threshold"] = wf["threshold"].apply(lambda x: f"{x:.0%}")
        st.dataframe(
            wf.rename(columns={
                "fold": "Fold", "test_start": "Start", "test_end": "End",
                "n_test": "N", "n_pos": "Pos", "auc": "AUC",
                "brier": "Brier", "threshold": "Threshold",
            }),
            use_container_width=True, hide_index=True,
        )

    with col_r:
        st.subheader("Calibration Curve")
        st.caption("If the model were well-calibrated, dots would lie on the diagonal.")
        if not result.calibration.empty:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"), name="Perfect calibration",
            ))
            fig_cal.add_trace(go.Scatter(
                x=result.calibration["predicted_mean"],
                y=result.calibration["observed_frequency"],
                mode="markers+lines", name="Model",
                marker=dict(size=10, color="crimson"),
            ))
            fig_cal.update_layout(
                xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                height=320, margin=dict(t=10),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.info("Insufficient out-of-sample predictions to plot calibration.")
