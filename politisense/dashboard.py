"""
dashboard.py  —  PolitiSense Streamlit Dashboard
Run:  streamlit run dashboard.py
"""
import sys, os, time
from pathlib import Path
from datetime import datetime, timedelta, date

sys.path.insert(0, str(Path(__file__).parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="PolitiSense", page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container { padding-top:1.5rem; padding-bottom:2rem; }
  .metric-card { border-radius:10px; padding:14px 18px; text-align:center;
    border:1px solid rgba(128,128,128,0.2); background:rgba(128,128,128,0.07); }
  .metric-label { font-size:12px; color:rgba(128,128,128,0.85); margin-bottom:4px; margin-top:0; }
  .metric-value { font-size:22px; font-weight:600; margin:0; }
  .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .badge-critical  { background:rgba(220,53,69,0.15);  color:#e05c6a; }
  .badge-high      { background:rgba(253,126,20,0.15); color:#fd7e14; }
  .badge-medium    { background:rgba(255,193,7,0.15);  color:#d4a017; }
  .badge-low       { background:rgba(40,167,69,0.15);  color:#34a85a; }
  .badge-verified  { background:rgba(40,167,69,0.15);  color:#34a85a; }
  .badge-likely    { background:rgba(255,193,7,0.15);  color:#d4a017; }
  .badge-disputed  { background:rgba(253,126,20,0.15); color:#fd7e14; }
  .badge-unverified{ background:rgba(128,128,128,0.15);color:rgba(128,128,128,0.85); }
  .source-box { border-left:3px solid #4dabf7; border-radius:0 8px 8px 0;
    padding:10px 14px; font-size:13px; background:rgba(77,171,247,0.08); }
  .source-box a { color:#4dabf7; }
  .narrative-box { border-left:3px solid #fcc419; border-radius:0 8px 8px 0;
    padding:12px 16px; font-size:14px; line-height:1.7; background:rgba(252,196,25,0.08); }
  .rag-date-banner { border-radius:8px; padding:10px 16px; font-size:13px;
    background:rgba(77,171,247,0.10); border:1px solid rgba(77,171,247,0.25); margin-bottom:6px; }
  .progress-bg { background:rgba(128,128,128,0.15); border-radius:4px; height:5px; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS       = {"CRITICAL":"#dc3545","HIGH":"#fd7e14","MEDIUM":"#ffc107","LOW":"#28a745"}
STATUS_EMOJI      = {"verified":"✅","likely":"🟡","disputed":"🟠","unverified":"❌"}
SUSPENDED_TICKERS = {"ERUS","RSX","RSXJ"}


def parse_rag_date(report):
    raw = report.get("top_source_date") or report.get("event_date") or ""
    try:    return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except: return None


def risk_gauge(score, label):
    color = RISK_COLORS.get(label, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix":"/90","font":{"size":28,"color":color}},
        gauge={
            "axis":{"range":[0,90],"tickwidth":1,"tickcolor":"#888"},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[
                {"range":[0,35], "color":"rgba(40,167,69,0.12)"},
                {"range":[35,55],"color":"rgba(23,162,184,0.12)"},
                {"range":[55,75],"color":"rgba(255,193,7,0.12)"},
                {"range":[75,90],"color":"rgba(220,53,69,0.12)"},
            ],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=20,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color":"#aaa"})
    return fig


def compute_metrics(prices, event_date):
    ev  = pd.Timestamp(event_date)
    pre = prices[prices.index < ev]
    pos = prices[prices.index >= ev]
    if len(pre) < 2 or len(pos) < 2: return None
    pre_ret = ((pre.iloc[-1]-pre.iloc[0])/pre.iloc[0])*100
    pos_ret = ((pos.iloc[-1]-pos.iloc[0])/pos.iloc[0])*100
    pre_vol = float(pre.pct_change().dropna().std()*np.sqrt(252)*100)
    pos_vol = float(pos.pct_change().dropna().std()*np.sqrt(252)*100)
    vol_chg = ((pos_vol-pre_vol)/pre_vol*100) if pre_vol>0 else 0.0
    return {"pre_ret":round(float(pre_ret),2),"pos_ret":round(float(pos_ret),2),
            "pre_vol":round(pre_vol,2),"pos_vol":round(pos_vol,2),"vol_chg":round(vol_chg,2)}


def build_chart(prices, ticker, name, event_date, show_proj):
    ev       = pd.Timestamp(event_date)
    pre_data = prices[prices.index < ev]
    pos_data = prices[prices.index >= ev]
    fig = go.Figure()
    if len(pre_data):
        fig.add_vrect(x0=pre_data.index[0], x1=ev, fillcolor="rgba(74,144,226,0.08)", layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=pre_data.index, y=pre_data.values, name="Pre-event",
            line=dict(color="#4a90e2",width=2), hovertemplate="%{x|%b %d, %Y}  $%{y:.2f}<extra>Pre-event</extra>"))
    if len(pos_data):
        fig.add_vrect(x0=ev, x1=pos_data.index[-1], fillcolor="rgba(220,53,69,0.08)", layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=pos_data.index, y=pos_data.values, name="Post-event",
            line=dict(color="#e24b4a",width=2), hovertemplate="%{x|%b %d, %Y}  $%{y:.2f}<extra>Post-event</extra>"))
    ev_ts = pd.Timestamp(event_date).timestamp() * 1000  # plotly needs ms epoch
    fig.add_vline(x=ev_ts, line=dict(color="#fd7e14",width=2,dash="dash"),
        annotation_text=f"  {event_date.strftime('%b %d, %Y')} (whitehouse.gov)",
        annotation_position="top left", annotation_font_size=11, annotation_font_color="#fd7e14")
    if show_proj and len(pos_data) >= 3:
        x_num = np.arange(len(pos_data))
        slope, intercept = np.polyfit(x_num, pos_data.values.astype(float), 1)
        proj_dates = [pos_data.index[-1]+timedelta(days=i) for i in range(0,31)]
        proj_y     = [float(intercept+slope*(len(pos_data)-1+i)) for i in range(0,31)]
        fig.add_trace(go.Scatter(x=proj_dates, y=proj_y, name="30-day projection",
            line=dict(color="#888",width=1.5,dash="dot"),
            hovertemplate="%{x|%b %d, %Y}  Proj: $%{y:.2f}<extra>Projection</extra>"))
        fig.add_vrect(x0=pos_data.index[-1], x1=proj_dates[-1],
            fillcolor="rgba(128,128,128,0.05)", layer="below", line_width=0,
            annotation_text="  Projection", annotation_position="top right",
            annotation_font_size=10, annotation_font_color="#888")
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  {name}", font=dict(size=14), x=0),
        height=400, margin=dict(l=10,r=10,t=55,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=11)),
        xaxis=dict(showgrid=True,gridcolor="rgba(128,128,128,0.12)",tickformat="%b %d",tickfont=dict(size=11),zeroline=False),
        yaxis=dict(showgrid=True,gridcolor="rgba(128,128,128,0.12)",tickprefix="$",tickfont=dict(size=11),zeroline=False),
        hovermode="x unified",
    )
    return fig


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏛️ PolitiSense")
st.markdown("Trump policy risk analysis — Verify · Score · Market impact")
st.divider()

col_news, col_btn = st.columns([5,1])
with col_news:
    news_text = st.text_input("News headline",
        value="Trump announces 25% tariff on India and unspecified penalties for buying Russian oil",
        placeholder="Paste any Trump policy headline...")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyse ▶", type="primary", use_container_width=True)

st.caption("Event date is automatically sourced from the matched whitehouse.gov document via RAG.")

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run and news_text.strip():
    st.session_state.pop("price_cache", None)
    with st.spinner("Running PolitiSense agent — fingerprint → RAG → risk score → ETF map (~30s)..."):
        try:
            from pipeline import run_agent
            report = run_agent(news_text=news_text.strip(), thread_id="dashboard")
            st.session_state["report"] = report
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

# ── Results ────────────────────────────────────────────────────────────────────
report = st.session_state.get("report")

if report:
    fp        = report.get("policy_fingerprint", {})
    risk_score= report.get("risk_score", 0)
    risk_label= report.get("risk_label", "LOW")
    v_status  = report.get("verification_status", "unverified")
    v_conf    = report.get("verification_confidence", 0)
    bd        = report.get("risk_breakdown", {})
    tickers   = report.get("tickers", {})
    narrative = report.get("analyst_narrative", "")
    src_title = report.get("top_source_title", "N/A")
    src_date  = report.get("top_source_date", "")
    src_url   = report.get("top_source_url", "#")
    rag_date  = parse_rag_date(report)

    # Section 1
    st.markdown("### Policy analysis")
    col_gauge, col_verify, col_fp = st.columns([1.2,1.5,1.3])
    with col_gauge:
        st.markdown("**Risk score**")
        st.plotly_chart(risk_gauge(risk_score,risk_label), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="text-align:center;margin-top:-10px"><span class="badge badge-{risk_label.lower()}">{risk_label}</span></div>', unsafe_allow_html=True)
    with col_verify:
        st.markdown("**Verification**")
        emoji = STATUS_EMOJI.get(v_status,"❓")
        st.markdown(f'{emoji} <span class="badge badge-{v_status}">{v_status.upper()}</span>&nbsp;&nbsp;<span style="font-size:13px;opacity:0.7">{v_conf}/100 confidence</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="source-box"><b style="font-size:12px">Matched source</b><br>{src_title}<br><span style="font-size:11px;opacity:0.6">{src_date}</span><br><a href="{src_url}" target="_blank" style="font-size:11px;">whitehouse.gov ↗</a></div>', unsafe_allow_html=True)
    with col_fp:
        st.markdown("**Policy fingerprint**")
        st.markdown(f"| | |\n|---|---|\n| Type | `{fp.get('policy_type','N/A')}` |\n| Authority | `{fp.get('legal_authority','N/A') or 'N/A'}` |\n| Scope | `{fp.get('scope','N/A') or 'N/A'}` |\n| Target | `{fp.get('target','N/A') or 'N/A'}` |")

    st.divider()

    # Section 2: Risk breakdown
    st.markdown("### Risk breakdown")
    cols = st.columns(4)
    for col,(label,key,max_val) in zip(cols,[
        ("Verification signal","verification_signal",30),
        ("Policy type","policy_type",20),
        ("Legal authority","legal_authority",15),
        ("Scope breadth","scope_breadth",25),
    ]):
        val   = bd.get(key,0)
        pct   = int(val/max_val*100)
        color = "#34a85a" if pct>=70 else "#d4a017" if pct>=40 else "#e05c6a"
        with col:
            st.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p><p class="metric-value" style="color:{color}">{val}<span style="font-size:13px;opacity:0.5">/{max_val}</span></p><div class="progress-bg"><div style="width:{pct}%;background:{color};height:5px;border-radius:4px;"></div></div></div>', unsafe_allow_html=True)

    st.divider()

    # Section 3: Narrative
    if narrative:
        st.markdown("### Analyst narrative")
        st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)
        st.divider()

    # Section 4: Market charts
    st.markdown("### Market impact")
    if not rag_date:
        st.warning("RAG did not return a source date — cannot draw chart.")
        st.stop()

    st.markdown(f'<div class="rag-date-banner">📅 &nbsp;Chart event date from RAG: <b>{rag_date.strftime("%B %d, %Y")}</b> &nbsp;·&nbsp; <span style="opacity:0.7">"{src_title}"</span></div>', unsafe_allow_html=True)
    st.caption("Blue = pre-event  ·  Red = post-event  ·  Orange = policy date  ·  Dotted = 30-day projection")

    ctrl1, ctrl2 = st.columns([1,5])
    with ctrl1:
        show_proj = st.toggle("Show projection", value=True)
    with ctrl2:
        if st.button("↻ Refresh charts", help="Re-fetch market data"):
            st.session_state.pop("price_cache", None)
            st.rerun()

    # Deduplicate + skip suspended
    seen: set = set()
    unique_tickers: dict = {}
    for name, ticker in tickers.items():
        if ticker not in seen and ticker not in SUSPENDED_TICKERS:
            seen.add(ticker); unique_tickers[name] = ticker

    if not unique_tickers:
        st.info("No valid ETF tickers mapped.")
    else:
        if "price_cache" not in st.session_state:
            st.session_state["price_cache"] = {}
        cache   = st.session_state["price_cache"]
        missing = [t for t in unique_tickers.values() if t not in cache]

        if missing:
            bar = st.progress(0, text=f"Fetching {len(missing)} tickers...")
            start_str = str(rag_date - timedelta(days=70))
            end_str   = str(min(rag_date + timedelta(days=40), date.today()))
            for i, ticker in enumerate(missing):
                bar.progress((i+1)/len(missing), text=f"Fetching {ticker} ({i+1}/{len(missing)})...")
                try:
                    import yfinance as yf
                    t    = yf.Ticker(ticker)
                    hist = t.history(start=start_str, end=end_str, auto_adjust=True)
                    if not hist.empty:
                        close = hist["Close"].dropna()
                        if close.index.tz is not None:
                            close.index = close.index.tz_localize(None)
                        cache[ticker] = close
                    else:
                        cache[ticker] = None
                except Exception:
                    cache[ticker] = None
            bar.empty()

        tabs = st.tabs([f"{t} — {n}" for n,t in unique_tickers.items()])
        for tab,(name,ticker) in zip(tabs, unique_tickers.items()):
            with tab:
                prices = cache.get(ticker)
                if prices is None or len(prices) < 5:
                    st.warning(f"**{ticker}** — no data. Click ↻ Refresh charts to retry.")
                    continue
                if prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)
                m = compute_metrics(prices, rag_date)
                if m:
                    mc1,mc2,mc3,mc4 = st.columns(4)
                    def rc(v): return "#34a85a" if v>=0 else "#e05c6a"
                    def vc(v): return "#e05c6a" if v>=0 else "#34a85a"
                    for mcol,lbl,val,sub,cfn in [
                        (mc1,"Pre-event return", m["pre_ret"],"60 days before EO",rc),
                        (mc2,"Post-event return",m["pos_ret"],"30 days after EO", rc),
                        (mc3,"Volatility change",m["vol_chg"],"Annualised vol Δ", vc),
                        (mc4,"Post-event vol",   m["pos_vol"],"Annualised",       lambda v:"#aaa"),
                    ]:
                        sign = "+" if val>=0 else ""
                        mcol.markdown(f'<div class="metric-card"><p class="metric-label">{lbl}</p><p class="metric-value" style="color:{cfn(val)}">{sign}{val:.1f}%</p><p class="metric-label">{sub}</p></div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(build_chart(prices,ticker,name,rag_date,show_proj), use_container_width=True, config={"displayModeBar":False})

else:
    st.markdown('<div style="text-align:center;padding:4rem 0;opacity:0.5;"><div style="font-size:48px;margin-bottom:1rem;">🏛️</div><div style="font-size:18px;font-weight:500;margin-bottom:8px;">Enter a headline and click Analyse</div><div style="font-size:14px;">PolitiSense verifies against whitehouse.gov, scores geopolitical risk,<br>and charts market impact around the exact policy date</div></div>', unsafe_allow_html=True)
