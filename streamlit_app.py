"""
Streamlit Dashboard for LSTM‑Attention Engine.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant LSTM‑Attention", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def return_text(val):
    try:
        val = float(val)
    except:
        return "N/A"
    return f"{val*100:+.2f}%"

def render_attention_heatmap(attention, title="Attention Weights"):
    """Render a heatmap of the last‑step attention over the lookback window."""
    if not attention or len(attention) == 0:
        return
    fig = go.Figure(data=go.Heatmap(
        z=[attention],
        colorscale='Blues',
        showscale=True,
        x=[f"t-{i}" for i in range(len(attention), 0, -1)],
        y=["Attention"],
    ))
    fig.update_layout(title=title, height=200, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

def render_mode_tab(mode_data, mode_name):
    if not mode_data:
        st.warning(f"No {mode_name} data.")
        return
    top = mode_data.get('top_picks', [])
    if not top:
        st.info(f"No predictions for {mode_name}.")
        return
    pick = top[0]
    ticker = pick['ticker']
    pred = pick['predicted_return']

    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔮 {mode_name} TOP PICK</div>
        <div class="hero-ticker">{ticker}</div>
        <div>Predicted Return: {return_text(pred)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Attention heatmap for the top pick
    st.markdown("### Attention Weights (last time step)")
    attn = mode_data.get('attention_last_step', [])
    render_attention_heatmap(attn)

    st.markdown("### Top 3 Picks")
    rows = [{"Ticker": p['ticker'], "Predicted Return": return_text(p['predicted_return'])} for p in top]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    all_scores = mode_data.get('all_scores', [])
    if all_scores:
        st.markdown("### All ETFs")
        df_all = pd.DataFrame(all_scores)
        df_all['Predicted Return'] = df_all['predicted_return'].apply(return_text)
        df_all = df_all[['ticker', 'Predicted Return']].sort_values('Predicted Return', ascending=False)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

def render_shrinking_tab(shrinking_data):
    if not shrinking_data:
        st.warning("No shrinking data.")
        return
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔄 SHRINKING CONSENSUS</div>
        <div class="hero-ticker">{shrinking_data['ticker']}</div>
        <div>{shrinking_data['conviction']:.0f}% conviction · {shrinking_data['num_windows']} windows</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 All Windows"):
        rows = []
        for w in shrinking_data.get('windows', []):
            rows.append({
                'Window': f"{w['window_start']}-{w['window_end']}",
                'Top Pick': w['ticker'],
                'Predicted Return': return_text(w.get('predicted_return', 0))
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🔮 P2Quant LSTM‑Attention</div>', unsafe_allow_html=True)
st.markdown('<div>Bidirectional LSTM + Multi‑Head Self‑Attention – Interpretable ETF Forecasting</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

universes_data = data.get('universes', {})
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, keys):
    uni = universes_data.get(key, {})
    if not uni:
        with tab:
            st.info(f"No data for {key}.")
        continue
    with tab:
        d, g, s = st.tabs(["📅 Daily (504d)", "🌍 Global (2008‑YTD)", "🔄 Shrinking Consensus"])
        with d:
            render_mode_tab(uni.get('daily'), "Daily")
        with g:
            render_mode_tab(uni.get('global'), "Global")
        with s:
            render_shrinking_tab(uni.get('shrinking'))
