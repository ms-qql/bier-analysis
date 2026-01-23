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
from modules import backtest

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

allow_short = st.sidebar.checkbox("Allow Shortselling", value=False, help="If checked, strategy will short sell (value -1) instead of going to cash (value 0)")

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

st.sidebar.markdown("---")
# Backtest Section
st.sidebar.header('Batch Backtest')
if st.sidebar.button('Run Backtest Batch'):
  with st.spinner('Running Batch Backtest...'):
      # Convert dates to string format required by backtest logic
      bt_start_str = start_date.strftime('%Y-%m-%d')
      bt_end_str = end_date.strftime('%Y-%m-%d')
      result_msg = backtest.run_batch_backtest(bt_start_str, bt_end_str, asset, signal_strategy, use_alt_signal, alt_signal_deviation, allow_short)
      st.sidebar.success(result_msg)

if st.sidebar.button("Show Backtest Results"):
    st.session_state['show_backtest_results'] = True
    st.rerun()

if st.sidebar.button("Clear Backtest History", type="secondary"):
    database.delete_bier_backtest_table_content()
    st.sidebar.success("Backtest history cleared!")
    st.rerun()

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_and_process_data(start_str, end_str, asset_name, cat_sel, use_sig, use_alt_signal=False, alt_signal_deviation=5, custom_metrics_list=None, allow_short=False):
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
            df = matrix_strategy.calc_signal(df, 'invest_score', deviation=alt_signal_deviation)            
        
        # Add range column (for signal strategy compatibility)
        df['range'] = np.where((df['invest_score'] == 50), 1, np.nan)
        
        # Add trigger columns
        df['trigger_short'] = df['peaks']
        df['trigger_long'] = df['valleys']
        
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
    #matrix = matrix_strategy.matrix_strategy()
    #df['score_ma'] = matrix.double_hull_ma(df['invest_score'], 5, 5)
    #df = matrix_strategy.calc_peaks_valleys(df, 'score_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)
    
    # Recalculate invested status
    peak_shift = 6
    df = df.copy() # Defragment
    df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
    df['extremes'] = df['extremes'].replace(0, np.nan)
    df['extremes'] = df['extremes'].ffill(axis = 0)
    df['extremes'] = df['extremes'].replace(0, np.nan)
    df['extremes'] = df['extremes'].ffill(axis = 0)
    
    if allow_short:
        # If valley (<0) -> Long (1), If Peak (>0) -> Short (-1)
        df['invested'] = np.where(df['extremes'] < 0, 1, -1)
    else:
        # If valley (<0) -> Long (1), If Peak (>0) -> Cash (nan)
        df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)

    df.to_csv(f'data/calc_multi_strategy_streamlit.csv') 
    # Update json_res to include the calculated invest_score for all_categories and custom
    #if cat_sel == 'all_categories' or cat_sel == 'custom':
    json_res = df.to_json(orient='records')
    
    return df, json_res, metrics_list

# --- Main Dashboard ---
st.title("BIER Strategy Dashboard")

if update_graphs or 'data_loaded' not in st.session_state or st.session_state.get('show_backtest_results', False):
    st.session_state.data_loaded = True
    
    with st.spinner("Calculating performance and generating charts..."):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Load and process data
        df, calc_json, metrics_list = load_and_process_data(
            start_str, end_str, asset, category_sel, signal_strategy, 
            use_alt_signal, alt_signal_deviation,
            custom_metrics if category_sel == 'custom' else None,
            allow_short
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
        tab1, tab2, tab3 = st.tabs(["Main Overview", "Deep Dive: Categories", "Backtest Evaluation"])

        with tab1:
            # Row 1: Price Chart & Radar
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.subheader("Price & Invested Signal")
                price_chart_json = graphs.update_price_chart(calc_json, signal_strategy, category_sel, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation, allow_short=allow_short)
                st.plotly_chart(pio.from_json(price_chart_json), width='stretch', key="price_chart")
            with col_right:
                st.subheader("Regime Health")
                # Temporary disabled
                #radar_json = graphs.create_radar_chart(calc_json, available_categories, signal_strategy)
                #st.plotly_chart(pio.from_json(radar_json), width='stretch', key="radar_chart")

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

            st.subheader("Individual Metric Signals")

            norm_chart_json = graphs.update_norm_chart(calc_json, signal_strategy, category_sel, metrics_list, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation, allow_short=allow_short)
            #norm_chart_json = graphs.update_norm_chart(calc_json, category_sel, show_raw_data, signal_strategy, custom_metrics if category_sel == 'custom' else None, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation)
            st.plotly_chart(pio.from_json(norm_chart_json), width='stretch', key="norm_chart")


            # Temporary disabled
            #st.subheader("Cumulative Category Scores")
            #cat_chart_json = graphs.update_category_chart(calc_json, signal_strategy, category_sel, use_alt_signal=use_alt_signal, alt_signal_deviation=alt_signal_deviation)
            #st.plotly_chart(pio.from_json(cat_chart_json), width='stretch', key="cat_chart")

        with tab3:
            st.subheader("Backtest Evaluation")
            
            if st.session_state.get('show_backtest_results', False):
                st.markdown("Results from the Batch Backtest run.")
                
                # Read backtest table
                df_bt_res = database.read_backtest_table()
                
                if not df_bt_res.empty:
                    # Calculate Rankings for each metric
                    # Higher is better for return, sharpe, sortino, calmar
                    # Lower is better for max_dd (now stored as absolute value)
                    df_bt_res['rank_return'] = df_bt_res['return'].rank(ascending=False, method='min')
                    df_bt_res['rank_sharpe'] = df_bt_res['sharpe'].rank(ascending=False, method='min')
                    df_bt_res['rank_sortino'] = df_bt_res['sortino'].rank(ascending=False, method='min')
                    df_bt_res['rank_calmar'] = df_bt_res['calmar'].rank(ascending=False, method='min')
                    df_bt_res['rank_max_dd'] = df_bt_res['max_dd'].rank(ascending=True, method='min')
                    
                    # Calculate average rank
                    df_bt_res['avg_rank'] = df_bt_res[['rank_return', 'rank_sharpe', 'rank_sortino', 'rank_calmar', 'rank_max_dd']].mean(axis=1)
                    
                    # Summary Stats - Top of page with 5 columns
                    st.subheader("Best Performers")
                    kpi_b1, kpi_b2, kpi_b3, kpi_b4, kpi_b5 = st.columns(5)
                    
                    best_return = df_bt_res.loc[df_bt_res['return'].idxmax()]
                    best_sharpe = df_bt_res.loc[df_bt_res['sharpe'].idxmax()]
                    best_sortino = df_bt_res.loc[df_bt_res['sortino'].idxmax()]
                    best_calmar = df_bt_res.loc[df_bt_res['calmar'].idxmax()]
                    
                    # For max_dd, find smallest value (now stored as absolute)
                    df_dd_nonzero = df_bt_res[df_bt_res['max_dd'] != 0].copy()
                    if not df_dd_nonzero.empty:
                        best_dd = df_dd_nonzero.loc[df_dd_nonzero['max_dd'].idxmin()]
                    else:
                        best_dd = None
                    
                    kpi_b1.metric("Best Return", f"{best_return['return']:.2f}%", f"{best_return.get('name', 'Unknown')}")
                    if best_dd is not None:
                        kpi_b2.metric("Best Max DD", f"{best_dd['max_dd']:.2f}%", f"{best_dd.get('name', 'Unknown')}")
                    else:
                        kpi_b2.metric("Best Max DD", "N/A", "No data")
                    kpi_b3.metric("Best Sharpe", f"{best_sharpe['sharpe']:.2f}", f"{best_sharpe.get('name', 'Unknown')}")
                    kpi_b4.metric("Best Sortino", f"{best_sortino['sortino']:.2f}", f"{best_sortino.get('name', 'Unknown')}")
                    kpi_b5.metric("Best Calmar", f"{best_calmar['calmar']:.2f}", f"{best_calmar.get('name', 'Unknown')}")
                    
                    st.markdown("---")
                    
                    # Top 20 by Average Rank Chart
                    st.subheader("Top 20 Performers by Average Rank")
                    top_20_rank = df_bt_res.nsmallest(20, 'avg_rank')[['name', 'avg_rank', 'return', 'sharpe', 'sortino', 'calmar', 'max_dd', 
                                                                         'rank_return', 'rank_sharpe', 'rank_sortino', 'rank_calmar', 'rank_max_dd']].copy()
                    
                    # Reverse order so best is at top
                    top_20_rank = top_20_rank.iloc[::-1]
                    
                    fig_rank = go.Figure(go.Bar(
                        x=top_20_rank['avg_rank'],
                        y=top_20_rank['name'],
                        orientation='h',
                        marker=dict(color='teal'),
                        text=top_20_rank['avg_rank'].round(1),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>' +
                                      'Avg Rank: %{x:.1f}<br>' +
                                      'Return: ' + top_20_rank['return'].apply(lambda x: f'{x:.2f}%').values + '<br>' +
                                      'Sharpe: ' + top_20_rank['sharpe'].apply(lambda x: f'{x:.2f}').values + '<br>' +
                                      'Sortino: ' + top_20_rank['sortino'].apply(lambda x: f'{x:.2f}').values + '<br>' +
                                      'Calmar: ' + top_20_rank['calmar'].apply(lambda x: f'{x:.2f}').values + '<br>' +
                                      'Max DD: ' + top_20_rank['max_dd'].apply(lambda x: f'{x:.2f}%').values +
                                      '<extra></extra>'
                    ))
                    fig_rank.update_layout(
                        height=600,
                        xaxis_title="Average Rank (Lower is Better)",
                        yaxis_title="",
                        showlegend=False,
                        margin=dict(l=0, r=50, t=20, b=0)
                    )
                    st.plotly_chart(fig_rank, use_container_width=True)
                    
                    # Expandable details table with individual rankings
                    with st.expander("📊 View Individual Rankings per Category"):
                        # Prepare detailed ranking table (reverse back to show best first)
                        details_df = top_20_rank.iloc[::-1][['name', 'avg_rank', 'rank_return', 'rank_sharpe', 'rank_sortino', 'rank_calmar', 'rank_max_dd', 
                                                               'return', 'sharpe', 'sortino', 'calmar', 'max_dd']].copy()
                        
                        # Rename columns for better readability
                        details_df.columns = ['Strategy', 'Avg Rank', 'Return Rank', 'Sharpe Rank', 'Sortino Rank', 'Calmar Rank', 'MaxDD Rank',
                                             'Return (%)', 'Sharpe', 'Sortino', 'Calmar', 'Max DD (%)']
                        
                        # Format numeric columns
                        details_df['Avg Rank'] = details_df['Avg Rank'].round(1)
                        details_df['Return (%)'] = details_df['Return (%)'].round(2)
                        details_df['Sharpe'] = details_df['Sharpe'].round(2)
                        details_df['Sortino'] = details_df['Sortino'].round(2)
                        details_df['Calmar'] = details_df['Calmar'].round(2)
                        details_df['Max DD (%)'] = details_df['Max DD (%)'].round(2)
                        
                        st.dataframe(details_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    
                    # Filter Controls
                    st.subheader("Filter Results")
                    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5, filter_col6, filter_col7 = st.columns(7)
                    
                    with filter_col1:
                        filter_nb_metrics = st.number_input("# Metrics =", value=None, step=1, min_value=1, help="Show only results with this exact number of metrics")
                    with filter_col2:
                        filter_return = st.number_input("Return > (%)", value=None, step=1.0, format="%.1f", help="Show only results with return greater than this value")
                    with filter_col3:
                        filter_max_dd = st.number_input("Max DD < (%)", value=None, step=1.0, format="%.1f", help="Show only results with max drawdown less than this value")
                    with filter_col4:
                        filter_sharpe = st.number_input("Sharpe >", value=None, step=0.1, format="%.2f", help="Show only results with Sharpe ratio greater than this value")
                    with filter_col5:
                        filter_sortino = st.number_input("Sortino >", value=None, step=0.1, format="%.2f", help="Show only results with Sortino ratio greater than this value")
                    with filter_col6:
                        filter_calmar = st.number_input("Calmar >", value=None, step=0.1, format="%.2f", help="Show only results with Calmar ratio greater than this value")
                    with filter_col7:
                        filter_short = st.selectbox("Short Selling", ["All", "Enabled", "Disabled"], index=0, help="Filter by Short Selling flag")
                    
                    # Apply filters
                    df_filtered = df_bt_res.copy()
                    active_filters = []
                    
                    if filter_nb_metrics is not None:
                        df_filtered = df_filtered[df_filtered['nb_metrics'] == filter_nb_metrics]
                        active_filters.append(f"# Metrics = {filter_nb_metrics}")
                    
                    if filter_return is not None:
                        df_filtered = df_filtered[df_filtered['return'] > filter_return]
                        active_filters.append(f"Return > {filter_return}%")
                    
                    if filter_max_dd is not None:
                        df_filtered = df_filtered[df_filtered['max_dd'] < filter_max_dd]
                        active_filters.append(f"Max DD < {filter_max_dd}%")
                    
                    if filter_sharpe is not None:
                        df_filtered = df_filtered[df_filtered['sharpe'] > filter_sharpe]
                        active_filters.append(f"Sharpe > {filter_sharpe}")
                    
                    if filter_sortino is not None:
                        df_filtered = df_filtered[df_filtered['sortino'] > filter_sortino]
                        active_filters.append(f"Sortino > {filter_sortino}")
                    
                    if filter_calmar is not None:
                        df_filtered = df_filtered[df_filtered['calmar'] > filter_calmar]
                        active_filters.append(f"Calmar > {filter_calmar}")

                    if filter_short != "All":
                        if 'short_sell' in df_filtered.columns:
                            # Handle potential NaN or diverse types safely
                            df_filtered['short_sell'] = df_filtered['short_sell'].fillna(False)
                            if filter_short == "Enabled":
                                df_filtered = df_filtered[df_filtered['short_sell'] == True]
                                active_filters.append("Short Selling: Enabled")
                            else: # Disabled
                                df_filtered = df_filtered[df_filtered['short_sell'] == False] 
                                active_filters.append("Short Selling: Disabled")
                    
                    # Show active filters and result count
                    if active_filters:
                        st.info(f"**Active Filters:** {' AND '.join(active_filters)} | **Results:** {len(df_filtered)} of {len(df_bt_res)}")
                    else:
                        st.info(f"**No filters active** | **Total Results:** {len(df_bt_res)}")
                    
                    # Define standard columns to show (including ranking columns and nb_metrics)
                    std_cols = ['id', 'test_run', 'date', 'name', 'start_date', 'end_date', 'signal_strategy', 'nb_metrics', 'short_sell',
                                'return', 'max_dd', 'sharpe', 'sortino', 'calmar', 
                                'rank_return', 'rank_sharpe', 'rank_sortino', 'rank_calmar', 'rank_max_dd', 'avg_rank']
                    
                    # Filter to only show standard columns (hide metric columns)
                    display_cols = [c for c in std_cols if c in df_filtered.columns]
                    df_display = df_filtered[display_cols].sort_values('avg_rank')
                    
                    st.dataframe(df_display, use_container_width=True)
                    
                    # Top 10 Performance Charts
                    st.markdown("---")
                    st.subheader("Top 10 Performers by Metric")
                    
                    # Create 5 columns for the charts
                    chart_col1, chart_col2 = st.columns(2)
                    chart_col3, chart_col4 = st.columns(2)
                    chart_col5 = st.container()
                    
                    with chart_col1:
                        st.markdown("**Top 10 by Return**")
                        top_return = df_bt_res.nlargest(10, 'return')[['name', 'return']]
                        fig_return = go.Figure(go.Bar(
                            x=top_return['return'],
                            y=top_return['name'],
                            orientation='h',
                            marker=dict(color='green')
                        ))
                        fig_return.update_layout(
                            height=400,
                            xaxis_title="Return (%)",
                            yaxis_title="",
                            showlegend=False,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig_return, use_container_width=True)
                    
                    with chart_col2:
                        st.markdown("**Top 10 by Sharpe Ratio**")
                        top_sharpe = df_bt_res.nlargest(10, 'sharpe')[['name', 'sharpe']]
                        fig_sharpe = go.Figure(go.Bar(
                            x=top_sharpe['sharpe'],
                            y=top_sharpe['name'],
                            orientation='h',
                            marker=dict(color='blue')
                        ))
                        fig_sharpe.update_layout(
                            height=400,
                            xaxis_title="Sharpe Ratio",
                            yaxis_title="",
                            showlegend=False,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig_sharpe, use_container_width=True)
                    
                    with chart_col3:
                        st.markdown("**Top 10 by Sortino Ratio**")
                        top_sortino = df_bt_res.nlargest(10, 'sortino')[['name', 'sortino']]
                        fig_sortino = go.Figure(go.Bar(
                            x=top_sortino['sortino'],
                            y=top_sortino['name'],
                            orientation='h',
                            marker=dict(color='purple')
                        ))
                        fig_sortino.update_layout(
                            height=400,
                            xaxis_title="Sortino Ratio",
                            yaxis_title="",
                            showlegend=False,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig_sortino, use_container_width=True)
                    
                    with chart_col4:
                        st.markdown("**Top 10 by Calmar Ratio**")
                        top_calmar = df_bt_res.nlargest(10, 'calmar')[['name', 'calmar']]
                        fig_calmar = go.Figure(go.Bar(
                            x=top_calmar['calmar'],
                            y=top_calmar['name'],
                            orientation='h',
                            marker=dict(color='orange')
                        ))
                        fig_calmar.update_layout(
                            height=400,
                            xaxis_title="Calmar Ratio",
                            yaxis_title="",
                            showlegend=False,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig_calmar, use_container_width=True)
                    
                    with chart_col5:
                        st.markdown("**Top 10 by Max Drawdown (Smallest)**")
                        # Filter out 0 values and get smallest drawdowns (now stored as absolute)
                        df_dd_filtered = df_bt_res[df_bt_res['max_dd'] != 0].copy()
                        if not df_dd_filtered.empty:
                            top_dd = df_dd_filtered.nsmallest(10, 'max_dd')[['name', 'max_dd']]
                            fig_dd = go.Figure(go.Bar(
                                x=top_dd['max_dd'],
                                y=top_dd['name'],
                                orientation='h',
                                marker=dict(color='red')
                            ))
                            fig_dd.update_layout(
                                height=400,
                                xaxis_title="Max Drawdown (%)",
                                yaxis_title="",
                                showlegend=False,
                                margin=dict(l=0, r=0, t=20, b=0)
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)
                        else:
                            st.info("No valid drawdown data available.")
                else:
                    st.info("No backtest results found. Run a batch backtest from the sidebar.")
            else:
                st.info("Click 'Show Backtest Results' in the sidebar to view results.")
            


st.sidebar.markdown("---")
st.sidebar.info("Dashboard built with Streamlit and BIER Database.")
