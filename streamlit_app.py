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
available_categories = ['all_categories', 'capriole', 'bmp', 'manta', 'itc', 'tv', 'strategy', 'market', 'mining', 'macro', 'shortterm', 'sentiment', 'hodl', 'treasury', 'supply_demand', 'eth', 'alts', 'custom', 'bier', 'test']
category_sel = st.sidebar.selectbox("Select Category / Strategy", available_categories, index=available_categories.index('bier'))

# Toggles
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
signal_strategy = st.sidebar.checkbox("Use Signal Strategy", value=True)
use_alt_signal = st.sidebar.checkbox("Use Alternative Signal Logic", value=False, help="Enable non-repainting peak/trough detection with regime bias")
if use_alt_signal:
    alt_signal_deviation = st.sidebar.slider("Signal Sensitivity", min_value=1, max_value=20, value=5, help="Higher value = less sensitive (fewer trades). Lower value = more sensitive.")
else:
    alt_signal_deviation = 5 # default value

# Custom metric selector (shown only when custom category is selected)
custom_metrics = []
if category_sel == 'custom':
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Custom Metrics Selection")
    
    # Get all available metrics
    all_metrics = database.get_all_metrics()
    
    # Get currently selected custom metrics from database
    df_cat = database.read_categories()
    current_custom = df_cat[df_cat['custom'] == 1]['metric'].tolist()
    
    # Multi-select widget
    custom_metrics = st.sidebar.multiselect(
        "Select Metrics",
        options=all_metrics,
        default=current_custom,
        help="Select metrics to include in custom strategy"
    )
    
    # Store Custom button
    if st.sidebar.button("Store Custom", type="secondary"):
        database.update_custom_metrics(custom_metrics)
        st.sidebar.success("Custom metrics saved!")
        st.rerun()

# Update Button
update_graphs = st.sidebar.button("UPDATE DASHBOARD", type="primary")

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_and_process_data(start_str, end_str, asset_name, cat_sel, use_sig, use_alt_signal=False, alt_signal_deviation=5, custom_metrics_list=None):
    # This calls the existing matrix_strategy logic
    json_res = matrix_strategy.calc_metric_all(start_str, end_str, "risk_level", asset_name)
    df = pd.read_json(StringIO(json_res), orient='records')
    
    # Re-calculate strategy for the selected category to get 'invested' column
    df_cat, _ = database.read_categories(), None
    
    # Handle all_categories: calculate weighted total score
    if cat_sel == 'all_categories':
        # Read category weights
        df_weights = database.read_category_weight_table()
        weights = df_weights.iloc[0]
        
        # Calculate score for each category and combine with weights
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for category in df_weights.columns:
            weight = weights[category]
            if weight > 0:
                metrics_list, _ = database.load_category_list(category, metric='', df=df_cat)
                if metrics_list:
                    df_temp = matrix_strategy.calc_multi_strategy(df.copy(), 6, metrics_list, use_sig, use_alt_signal, alt_signal_deviation)
                    weighted_sum += df_temp['invest_score'] * weight
                    total_weight += weight
        
        # Calculate weighted average
        df['invest_score'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate invested signal from the weighted score
        if use_alt_signal:
            df = matrix_strategy.calc_alternative_signal(df, 'invest_score', deviation=alt_signal_deviation)
        else:
            df['invest_score_ma'] = matrix_strategy.matrix_strategy().double_hull_ma(df['invest_score'], 5, 5)
            df = matrix_strategy.calc_peaks_valleys(df, 'invest_score_ma', peak_min=50, vert_dist=0, peak_dist=2, peak_width=0, peak_prominence=10, filt_double_extremes=False)
            df['extremes'] = df['peaks'].shift(6).fillna(0) - df['valleys'].shift(6).fillna(0)
            df['extremes'] = df['extremes'].replace(0, np.nan)
            df['extremes'] = df['extremes'].ffill(axis='rows')
            df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)
        
        # Add range column (for signal strategy compatibility)
        df['range'] = np.where((df['invest_score'] == 50), 1, np.nan)
        
        # Add trigger columns
        df['trigger_short'] = df['peaks']
        df['trigger_long'] = df['valleys']
        
    # If custom category and metrics provided, temporarily update the dataframe
    # If custom category and metrics provided, temporarily update the dataframe
    elif cat_sel == 'custom' and custom_metrics_list:
        # Temporarily modify df_cat for this calculation
        df_cat['custom'] = df_cat['metric'].isin(custom_metrics_list).astype(int)
        metrics_list, _ = database.load_category_list(cat_sel, metric='', df=df_cat)
        df = matrix_strategy.calc_multi_strategy(df, 6, metrics_list, use_sig, use_alt_signal, alt_signal_deviation)
    else:
        metrics_list, _ = database.load_category_list(cat_sel, metric='', df=df_cat)
        df = matrix_strategy.calc_multi_strategy(df, 6, metrics_list, use_sig, use_alt_signal, alt_signal_deviation)
    
    # Force consistent signal logic (Double Hull MA + Peak Detection) to match Price Chart
    # This ensures KPIs, Equity, and Drawdown charts match the visual signals
    matrix = matrix_strategy.matrix_strategy()
    df['score_ma'] = matrix.double_hull_ma(df['invest_score'], 5, 5)
    df = matrix_strategy.calc_peaks_valleys(df, 'score_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)
    
    # Recalculate invested status
    peak_shift = 6
    df = df.copy() # Defragment
    df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
    df['extremes'] = df['extremes'].replace(0, np.nan)
    df['extremes'] = df['extremes'].ffill(axis = 0)
    df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)

    # Update json_res to include the calculated invest_score for all_categories and custom
    if cat_sel == 'all_categories' or cat_sel == 'custom':
        json_res = df.to_json(orient='records')
    
    return df, json_res

# --- Main Dashboard ---
st.title("BIER Strategy Dashboard")

if update_graphs or 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True
    
    with st.spinner("Calculating performance and generating charts..."):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Load and process data
        df, calc_json = load_and_process_data(
            start_str, end_str, asset, category_sel, signal_strategy, 
            use_alt_signal, alt_signal_deviation,
            custom_metrics if category_sel == 'custom' else None
        )
        df_perf, metrics = matrix_strategy.calc_performance(df)
        
        # --- KPI SECTION ---
        st.markdown("### Strategy Performance Summary")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        strat_final = df_perf['strat_equity'].iloc[-1]
        bh_final = df_perf['bh_equity'].iloc[-1]
        strat_ret = strat_final - 100
        bh_ret = bh_final - 100
        
        kpi1.metric("Strategy Return", f"{strat_ret:.1f}%", f"B&H: {bh_ret:.1f}%")
        kpi2.metric("Max Drawdown", f"{metrics['strat_max_dd']:.1f}%", f"B&H: {metrics['bh_max_dd']:.1f}%")
        kpi3.metric("Sharpe Ratio", f"{metrics['strat_sharpe']:.2f}", f"B&H: {metrics['bh_sharpe']:.2f}")
        kpi4.metric("Sortino Ratio", f"{metrics['strat_sortino']:.2f}", f"B&H: {metrics['bh_sortino']:.2f}")
        kpi5.metric("Calmar Ratio", f"{metrics['strat_calmar']:.2f}", f"B&H: {metrics['bh_calmar']:.2f}")
        
        st.markdown("---")

        # --- TABS FOR BETTER ORGANIZATION ---
        tab1, tab2 = st.tabs(["Main Overview", "Deep Dive: Categories"])

        with tab1:
            # Row 1: Price Chart & Radar
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.subheader("Price & Invested Signal")
                price_chart_json = graphs.update_price_chart(calc_json, signal_strategy, category_sel, 0,0,0,0,0,0,0,0,0, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation)
                st.plotly_chart(pio.from_json(price_chart_json), width='stretch', key="price_chart")
            with col_right:
                st.subheader("Regime Health")
                radar_json = graphs.create_radar_chart(calc_json, available_categories, signal_strategy)
                st.plotly_chart(pio.from_json(radar_json), width='stretch', key="radar_chart")

            # Row 2: Equity & Drawdown side-by-side
            col_eq, col_dd = st.columns(2)
            with col_eq:
                st.subheader("Equity Curve (k\\$) — Starting Capital: \\$100,000")
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['strat_equity'], name="BIER Strategy", line=dict(color='blue', width=2)))
                fig_equity.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['bh_equity'], name="Buy & Hold (BTC)", line=dict(color='gray', width=1, dash='dot')))
                fig_equity.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_equity, width='stretch', key="equity_chart")

            with col_dd:
                st.subheader("Drawdown (%)")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['strat_dd'], name="Strat DD", fill='tozeroy', line=dict(color='red')))
                fig_dd.add_trace(go.Scatter(x=df_perf['date'], y=df_perf['bh_dd'], name="B&H DD", line=dict(color='gray', dash='dot')))
                fig_dd.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_dd, width='stretch', key="dd_chart")

        with tab2:
            st.subheader("Cumulative Category Scores")
            cat_chart_json = graphs.update_category_chart(calc_json, signal_strategy, category_sel, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation)
            st.plotly_chart(pio.from_json(cat_chart_json), width='stretch', key="cat_chart")

            st.subheader("Individual Metric Signals")
            norm_chart_json = graphs.update_norm_chart(calc_json, category_sel, show_raw_data, signal_strategy, custom_metrics if category_sel == 'custom' else None, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation)
            st.plotly_chart(pio.from_json(norm_chart_json), width='stretch', key="norm_chart")

st.sidebar.markdown("---")
st.sidebar.info("Dashboard built with Streamlit and BIER Database.")
