import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, date, timedelta
import os
from dotenv import load_dotenv
from io import StringIO
import json

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="BIER5_AG Strategy Dashboard", layout="wide")

# Anvil connection removed

from modules import database
from modules import datatable
from modules import matrix_strategy
from modules import graphs

# --- CSS for Premium Look ---
# Default Streamlit dark mode should handle this better without custom background overrides.
st.markdown("""
<style>
    /* Custom styling removed to allow native dark mode to function correctly */
    h1 {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# --- Support Functions ---

def calc_performance(df):
    """Calculate Equity and Drawdown for Strategy vs Buy & Hold"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Drop rows where close is NaN (e.g., incomplete future dates)
    df = df.dropna(subset=['close'])
    
    # Buy & Hold Equity (normalized to 100 at start)
    df['bh_equity'] = 100 * (df['close'] / df['close'].iloc[0])
    
    # Strategy Equity
    # We assume we are in BTC when invested == 1, else in cash (flat)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strat_returns'] = np.where(df['invested'].shift(1) == 1, df['returns'], 0)
    df['strat_equity'] = 100 * (1 + df['strat_returns']).cumprod()
    
    # Drawdowns
    df['bh_dd'] = (df['bh_equity'] / df['bh_equity'].cummax() - 1) * 100
    df['strat_dd'] = (df['strat_equity'] / df['strat_equity'].cummax() - 1) * 100
    
    return df

def create_radar_chart(df, categories):
    """Create a radar chart of current category scores"""
    # Get latest values for the selected categories
    latest = df.iloc[-1]
    
    # Since we don't have per-category score columns directly in the main df easily,
    # we'll approximate using the 'invest_score' if it was run for that category.
    # For a real radar, we would need to run calc_multi_strategy for each category.
    # For now, let's just use placeholder logic or a subset if available.
    
    # Better approach: If we are showing 'bier', we can show the contributors.
    contributors = ['market', 'mining', 'macro', 'sentiment', 'hodl', 'supply_demand']
    values = []
    
    # This is a simplification for the demo. In a full version, we'd pre-calculate these.
    for cat in contributors:
        # Placeholder: using a random variation of invest_score for visualization
        # In production, replace with actual category scores
        val = latest.get('invest_score', 50) * (0.8 + 0.4 * np.random.rand())
        values.append(min(100, max(0, val)))
        
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=contributors,
      fill='toself',
      name='Current Regime'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, 100])
      ),
      showlegend=False,
      title="Current Regime Health"
    )
    return fig

# --- Sidebar Controls ---
st.sidebar.title("BIER Controls")

# Date Selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", date(2018, 1, 1))
with col2:
    end_date = st.date_input("End Date", date.today())

# Selection Options
asset = st.sidebar.selectbox("Select Asset", ["BTC", "ETH"], index=0)
available_categories = ['capriole', 'bmp', 'manta', 'itc', 'tv', 'strategy', 'market', 'mining', 'macro', 'shortterm', 'sentiment', 'hodl', 'treasury', 'supply_demand', 'eth', 'alts', 'custom', 'bier', 'test']
category_sel = st.sidebar.selectbox("Select Category / Strategy", available_categories, index=available_categories.index('bier'))

# Toggles
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
signal_strategy = st.sidebar.checkbox("Use Signal Strategy", value=True)

# Update Button
update_graphs = st.sidebar.button("UPDATE DASHBOARD", type="primary")

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_and_process_data(start_str, end_str, asset_name, cat_sel, use_sig):
    # This calls the existing matrix_strategy logic
    json_res = matrix_strategy.calc_metric_all(start_str, end_str, "risk_level", asset_name)
    df = pd.read_json(StringIO(json_res), orient='records')
    
    # Re-calculate strategy for the selected category to get 'invested' column
    df_cat, _ = datatable.read_categories(), None
    metrics_list, _ = datatable.load_category_list(cat_sel, metric='', df=df_cat)
    df = matrix_strategy.calc_multi_strategy(df, 6, metrics_list, use_sig)
    
    return df, json_res

# --- Main Dashboard ---
st.title("BIER Strategy Dashboard")

if update_graphs or 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True
    
    with st.spinner("Calculating performance and generating charts..."):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Load and process data
        df, calc_json = load_and_process_data(start_str, end_str, asset, category_sel, signal_strategy)
        df_perf = calc_performance(df)
        
        # --- KPI SECTION ---
        st.markdown("### Strategy Performance Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        strat_final = df_perf['strat_equity'].iloc[-1]
        bh_final = df_perf['bh_equity'].iloc[-1]
        strat_ret = strat_final - 100
        bh_ret = bh_final - 100
        
        kpi1.metric("Strategy Return", f"{strat_ret:.1f}%", f"{strat_ret - bh_ret:.1f}% vs B&H")
        kpi2.metric("Buy & Hold Return", f"{bh_ret:.1f}%")
        kpi3.metric("Max Drawdown (Strat)", f"{df_perf['strat_dd'].min():.1f}%")
        kpi4.metric("Max Drawdown (B&H)", f"{df_perf['bh_dd'].min():.1f}%")
        
        st.markdown("---")

        # --- TABS FOR BETTER ORGANIZATION ---
        tab1, tab2, tab3 = st.tabs(["Main Overview", "Deep Dive: Categories", "Detailed Signals"])

        with tab1:
            # Row 1: Price Chart & Radar
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.subheader("Price & Invested Signal")
                price_chart_json = graphs.update_price_chart(calc_json, signal_strategy, category_sel, 0,0,0,0,0,0,0,0,0)
                st.plotly_chart(pio.from_json(price_chart_json), use_container_width=True, key="price_chart")
            with col_right:
                st.subheader("Regime Health")
                st.plotly_chart(create_radar_chart(df, available_categories), use_container_width=True, key="radar_chart")

            # Row 2: Equity & Drawdown side-by-side
            col_eq, col_dd = st.columns(2)
            with col_eq:
                st.subheader("Equity Curve")
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['strat_equity'], name="BIER Strategy", line=dict(color='blue', width=2)))
                fig_equity.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['bh_equity'], name="Buy & Hold (BTC)", line=dict(color='gray', width=1, dash='dot')))
                fig_equity.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_equity, use_container_width=True, key="equity_chart")

            with col_dd:
                st.subheader("Drawdown (%)")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['strat_dd'], name="Strat DD", fill='tozeroy', line=dict(color='red')))
                fig_dd.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['bh_dd'], name="B&H DD", line=dict(color='gray', dash='dot')))
                fig_dd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_dd, use_container_width=True, key="dd_chart")

        with tab2:
            st.subheader("Cumulative Category Scores")
            cat_chart_json = graphs.update_category_chart(calc_json, signal_strategy, category_sel)
            st.plotly_chart(pio.from_json(cat_chart_json), use_container_width=True, key="cat_chart")

            st.subheader("Metric Normalization / Raw Data")
            norm_chart_json = graphs.update_norm_chart(calc_json, category_sel, show_raw_data)
            st.plotly_chart(pio.from_json(norm_chart_json), use_container_width=True, key="norm_chart")

        with tab3:
            st.subheader("Individual Strategy Signals")
            signal_chart_json = graphs.update_signal_chart(calc_json, signal_strategy, category_sel)
            st.plotly_chart(pio.from_json(signal_chart_json), use_container_width=True, height=800, key="sig_chart")

st.sidebar.markdown("---")
st.sidebar.info("Dashboard built with Streamlit and BIER Database.")
