import anvil.server

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
from io import StringIO
from scipy.signal import savgol_filter, find_peaks

from backtesting import Backtest, Strategy
from backtesting.test import SMA, GOOG
from backtesting.lib import TrailingStrategy

from . import matrix_strategy 
from . import database 




class Matrix(TrailingStrategy):
    stop_loss_pct = 90 / 100 # means not using it
    trailing_sl_no_atr = 50 #14 - means not using it
    invest_share = 95 / 100 
    allow_short = False   # Default, can be overridden by Backtest.run(allow_short=True)

    def init(self):
        super().init()
        super().set_trailing_sl(self.trailing_sl_no_atr) # trailing SL with x ATR distance
        score = self.data.Score
        self.score = self.I(SMA, score, 1) # use SMA of 1 to get score into plot (workaround)


    def next(self):
        super().next()
        price = self.data.Close[-1]
        long_signal = self.data.signal_bottoms[-1]
        short_signal = self.data.signal_peaks[-1]
        
        if long_signal > 0:
            if self.position.is_short:
                self.position.close()
            self.buy(size=self.invest_share, sl=price*(1-self.stop_loss_pct))
        elif short_signal > 0:
            if self.position.is_long:
                self.position.close()
            
            if self.allow_short:
                if not self.position.is_short:
                     self.sell(size=self.invest_share, sl=price*(1+self.stop_loss_pct))
            pass           

# For plots do NOT use data with datetime as index, as it creates an error
#bt_plot = Backtest(df_backtest, Matrix, cash=10000, commission=.002)
#plot = bt_plot.run()
#bt_plot.plot()


@anvil.server.callable
def save_backtest_score(calc_json, asset, use_signal, category, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight, metric):  
  df = pd.read_json(calc_json, orient='records')
  # --------------- signals ------------------------------  
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 
  
  df_categories = database.read_categories() # read once for all categories
  metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
  print("Metrics Signal: ", metrics_list)  
  
  df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list, use_signal)
  df = df.rename(columns={'invest_score': 'Score'})      
  df['signal_peaks'] = df['trigger_short'].shift(peak_shift) * df['Score']
  df['signal_bottoms'] = df['trigger_long'].shift(peak_shift) * df['Score'] 
  
  '''if (category == 'tetris_cat') | (category == 'corr_cat') | (category == 'single_cat'):
    df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list)
    df = df.rename(columns={'invest_score': 'Score'})      
    df['signal_peaks'] = df['trigger_short'].shift(peak_shift) * df['Score']
    df['signal_bottoms'] = df['trigger_long'].shift(peak_shift) * df['Score']    
  else:
    df = matrix_strategy.calc_strategy(df, peak_shift, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight) 
    df = df.rename(columns={'strategy_ma': 'Score'})  
    df['signal_peaks'] = df['peaks'].shift(peak_shift) * df['Score']
    df['signal_bottoms'] = df['valleys'].shift(peak_shift) * df['Score'] '''

  df['time'] = df['date'].dt.strftime('%Y-%m-%d')
  df_json = df[['time','open','high','low','close','Score','signal_peaks','signal_bottoms','invested','range']]  
  df_json = df_json.iloc[:-2] # drop last 2 rows as data not always complete
  #print('DF: ', df.head(), df.columns)  
  #print('JSON Backtest: ', df_json.tail(10).to_string(), df_json.info())  

  # Extract and print latest trades
  #trade_filt = (df_json['signal_peaks'] > 0) | (df_json['signal_bottoms'] > 0) # Slice all trade signals 
  #df_trade = df_json[trade_filt] # filter out all trades in last X periods
  #df_trade = df_trade.fillna(0)  
  #print('Trades: ', df_trade.to_string()) 

  '''# create file_name
  if category == 'tetris_cat':
    filename = 'backtest_' + asset.lower() + ' - tetris'
  elif category == 'corr_cat':
    filename = 'backtest_' + asset.lower() + ' - corr'    
    
  else:
    name_risk = '_risk_cat' if risk_weight > 0 else ''
    name_market = '_market_cat' if market_weight > 0 else ''
    name_mining = '_mining_cat' if mining_weight > 0 else ''
    name_macro = '_macro_cat' if macro_weight > 0 else ''  
    name_sentiment = '_sentiment_cat' if sentiment_weight > 0 else ''
    name_hodl = '_hodl_cat' if hodl_weight > 0 else ''
    name_shortterm = '_shortterm_cat' if shortterm_weight > 0 else '' 
    name_custom = '_customm_cat' if custom_weight > 0 else '' 
    name_single = ('_' + metric) if single_weight > 0 else ''   

    filename = 'backtest_' + asset.lower() + name_risk + name_market + name_mining + name_macro + name_sentiment + name_hodl + name_shortterm + name_custom + name_single '''
  
  filename = 'backtest_' + asset.lower() + '_' + category
  df_bt = perform_backtest_bier(df_json, filename)
  return df_bt #.to_json() 



def create_backtest_df(df_bt):
    
    df_bt = df_bt.fillna(value=0)    
    try:
        df_bt['Close'] = df_bt['index_price']/1000 # divided by 1000 for backtesting
        df_bt.drop(['index_price'], axis=1, inplace = True)    
    except:
        df_bt['Close'] = df_bt['close']/1000 # divided by 1000 for backtesting
        df_bt.drop(['close'], axis=1, inplace = True)                
    df_bt['Open'] = df_bt['Close'].shift(1)    
    df_bt['Open'] = df_bt['Open'].fillna(df_bt['Close'])    
    df_bt['High'] = df_bt[['Open', 'Close']].max(axis=1)
    df_bt['Low'] =  df_bt[['Open', 'Close']].min(axis=1)
    #df_bt.rename(columns = {'score_matrix_neo' : 'Score'}, inplace = True)
    df_bt['time'] = pd.to_datetime(df_bt['time'])
    df_bt = df_bt.set_index('time') 
    print('Df for Backtest:', df_bt.tail(), df_bt.info())

    return df_bt

def create_backtest_bier_df(df_bt):
    df_bt = df_bt.fillna(value=0)    
    df_bt['Close'] = df_bt['close']/1000 # divided by 1000 for backtesting             
    df_bt['Open'] = df_bt['open']/1000 # divided by 1000 for backtesting
    df_bt['High'] = df_bt['high']/1000 # divided by 1000 for backtesting
    df_bt['Low'] =  df_bt['low']/1000 # divided by 1000 for backtesting
    df_bt['time'] = pd.to_datetime(df_bt['time'])
    df_bt = df_bt.set_index('time') 
    #df_bt.drop([['open','high','low','close']], axis=1, inplace = True)       
    #print('Df for Backtest:', df_bt.tail(), df_bt.info())

    return df_bt


@anvil.server.callable
def perform_backtest_bier(df_bt, name):

    #df_bt = pd.read_json(df_json, orient='records')
    df_bt['open'] = df_bt['open'].replace(0, pd.NA).ffill()
    df_bt['high'] = df_bt['high'].replace(0, pd.NA).ffill()    
    df_bt['low'] = df_bt['low'].replace(0, pd.NA).ffill()
    df_bt['close'] = df_bt['close'].replace(0, pd.NA).ffill()
    #print('DF BT: ', df_bt.tail(), df_bt.info())

    # Extract and print latest trades
    #trade_filt = (df_bt['signal_peaks'] > 0) | (df_bt['signal_bottoms'] > 0) # Slice all trade signals 
    #df_trade = df_bt[trade_filt] # filter out all trades in last X periods
    #df_trade = df_trade.fillna(0)  
    #print('Trades: ', df_trade.to_string()) 
 
    # Create backtest dataframe
    df_bt = create_backtest_bier_df(df_bt)

    bt_stats = Backtest(df_bt, Matrix, cash=10000, commission=.0005) # = 0.05% - Bitget
    stats = bt_stats.run()

    df_bt['equity'] = stats['_equity_curve']['Equity']
    file_name = './bier5/backtests/' + name + '_score.csv'    
    df_bt.to_csv(file_name)
    
    #entries = stats['_trades']['EntryTime']
    #entries_equities = df_bt['equity'].loc[df_bt.index.isin(entries)]
    #exits = stats['_trades']['ExitTime']
    #exits_equities = df_bt['equity'].loc[df_bt.index.isin(exits)]    

    # Add 'real number' to compensate for Backtester adaptation
    stats['_trades']['Size_adapted'] = stats['_trades']['Size'] / 1000
    stats['_trades']['EntryPrice_adapted'] = stats['_trades']['EntryPrice'] * 1000    
    stats['_trades']['ExitPrice_adapted'] = stats['_trades']['ExitPrice'] * 1000    
    
    try:
        print(stats)
        file_name = './bier4/backtests/' + name + '_stats.csv'
        stats.to_csv(file_name)    
        print(stats['_trades'].to_string()) # Print list of all trades 
        file_name = './bier4/backtests/' + name + '_trades.csv'
        stats['_trades'].to_csv(file_name)
        file_name = './bier4/backtests/' + name + '_equity.csv'        
        stats['_equity_curve'].to_csv(file_name)     
    except:
       pass
    
    x_axis = df_bt.index
    y_axis_equity = df_bt['equity']
    y_axis_btc = df_bt['close'] / df_bt['close'].iloc[0] * df_bt['equity'].iloc[0]
    #y_axis_btc = df_bt['Close'] / df_bt['Close'].iloc[0] * df_bt['equity'].iloc[0]    

    #            hovertemplate="Entry Time: %{customdata[7]}<br>Entry Price: %{customdata[3]}<br>Size: %{customdata[0]}",   
    #            hovertemplate="Exit Time: %{customdata[8]}<br>Exit Price: %{customdata[4]}<br>P&L: %{customdata[5]}",  

    fig = go.Figure(go.Scatter(
            x = x_axis, 
            y = y_axis_equity,
            mode = 'lines',
            name = 'BIER Equity'))
    fig.add_trace(go.Scatter(
            x = x_axis, 
            y = y_axis_btc,
            mode = 'lines',
            name = 'BTC'))
    
    """fig.add_trace(go.Scatter(
            x = entries, 
            y = exits_equities,
            mode = 'markers',
            customdata=stats['_trades'],
            marker=dict(size=10, color='green',symbol='cross'), 
            hovertemplate="Entry Time: %{customdata[7]}<br>Entry Price: %{customdata[11]:.0f}<br>Size: %{customdata[10]:.3f}",   
            name = 'Entries'))      
    fig.add_trace(go.Scatter(
            x = exits, 
            y = exits_equities,
            mode = 'markers',
            customdata=stats['_trades'],            
            marker=dict(size=10, color='red',symbol='cross'),   
            hovertemplate="Exit Time: %{customdata[8]}<br>Exit Price: %{customdata[12]:.0f}<br>P&L: %{customdata[5]:.2f}",               
            name = 'Exits'))"""
  

    fig.update_layout(
        title = 'Equity Curve',
        legend={'orientation':'h'},
        paper_bgcolor= 'white', #Aussenrand,
        plot_bgcolor= 'white') #'#363636') # specify layout
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')
    return fig.to_json() 

@anvil.server.callable
def get_backtest_stats(start_date, end_date, asset, okx_data, strategy_selected):
    #print(f" Start: {start_date}, end: {end_date} for {asset} with OKX {okx_data}")
    strategy = 'score_' + strategy_selected # selected strategy for backtest
    df_bt = create_backtest_df(start_date, end_date, asset, okx_data, strategy)

    bt_stats = Backtest(df_bt, Matrix, cash=10000, commission=.002)
    stats = bt_stats.run()
    print(stats)
    stats['_trades'].to_csv('backtest_trades.csv')

    strategy_return = stats['Return [%]']
    bh_return = stats['Buy & Hold Return [%]']
    ann_return = stats['Return (Ann.) [%]']
    exposure = stats['Exposure Time [%]']  

    sharpe = stats['Sharpe Ratio']
    sortino = stats['Sortino Ratio']
    max_dd = stats['Max. Drawdown [%]']
    avg_dd = stats['Avg. Drawdown [%]']    

    trades = stats['# Trades']    
    win_rate = stats['Win Rate [%]']     
    profit_factor = stats['Profit Factor']    
    expectancy = stats['Expectancy [%]']     

    statistics = {'strategy_return': strategy_return, 'bh_return': bh_return, 'ann_return': ann_return, 'exposure': exposure,
                  'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd, 'avg_dd': avg_dd,
                  'trades': trades, 'win_rate': win_rate, 'profit_factor': profit_factor, 'expectancy': expectancy             
                  }

    return statistics 


def get_backtest_stats_dict(df_bt, allow_short=False):
    """
    Helper to run backtest and return a dictionary of key stats.
    """
    try:
        bt_stats = Backtest(df_bt, Matrix, cash=10000, commission=.0005)
        stats = bt_stats.run(allow_short=allow_short)
        
        # Manual Calculation for Consistency
        equity = stats['_equity_curve']['Equity']
        returns = equity.pct_change().fillna(0)
        
        # Calculate years
        if len(equity) > 0:
            days = (equity.index[-1] - equity.index[0]).days
            years = days / 365.25
        else:
            years = 0

        # Annualized Return
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility & Downside Dev
        volatility = returns.std() * np.sqrt(365)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0.0001
        
        # Ratios
        sharpe = ann_ret / volatility if volatility > 0 else 0
        sortino = ann_ret / downside_dev if downside_dev > 0 else 0
        
        # Using library MaxDD (usually correct) for Calmar
        max_dd = abs(stats['Max. Drawdown [%]'])
        calmar = (ann_ret * 100) / max_dd if max_dd > 0 else 0

        return {
            'return': stats['Return [%]'], # Keep library return (Total %) or use total_ret * 100
            'max_dd': stats['Max. Drawdown [%]'],
            'sharpe': sharpe, # Use manual
            'sortino': sortino, # Use manual
            'calmar': calmar, # Use manual (recalculated with Ann Return)
            # Add others if needed
        }
    except Exception as e:
        print(f"Backtest error: {e}")
        return {
            'return': 0.0,
            'max_dd': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'calmar': 0.0
        }

@anvil.server.callable
def run_batch_backtest(start_date, end_date, asset, signal_strategy, use_alt_signal, alt_signal_deviation, allow_short=False, max_combination_size=2):
    """
    Batch backtest for all metrics with 'test=1' in database.
    After individual tests, tests combinations of profitable metrics (up to max_combination_size).
    
    Args:
        max_combination_size: Maximum number of metrics to combine (default 5)
    """
    from itertools import combinations
    
    print("Starting Batch Backtest...")
    
    # 1. Initialize Table
    database.create_bier_backtest_table()
    
    # 2. Get Metrics to Test
    test_metrics = database.get_test_metrics()
    if not test_metrics:
        print("No test metrics found.")
        return "No metrics marked for testing."
        
    print(f"Metrics to test: {test_metrics}")

    # 3. Sync Columns
    database.sync_backtest_columns(test_metrics)
    
    # 4. Define Test Run ID
    test_run_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 5. Load base data once for efficiency
    json_res = matrix_strategy.calc_metric_all(start_date, end_date, "risk_level", asset)
    df_base = pd.read_json(StringIO(json_res), orient='records')

    results = []
    profitable_metrics = []

    # PHASE 1: Test Individual Metrics
    print("\n=== PHASE 1: Testing Individual Metrics ===")
    for metric in test_metrics:
        print(f"Testing metric: {metric}")
        try:
            metrics_list = [metric]
            
            df_calc = matrix_strategy.calc_multi_strategy(df_base.copy(), 1, metrics_list, signal_strategy, use_alt_signal, alt_signal_deviation)
            df_calc = df_calc.rename(columns={'invest_score': 'Score'})
            
            # Using standard logic from save_backtest_score
            peak_shift = 1
            if 'trigger_short' in df_calc.columns:
                df_calc['signal_peaks'] = df_calc['trigger_short'].shift(peak_shift) * df_calc['Score']
                df_calc['signal_bottoms'] = df_calc['trigger_long'].shift(peak_shift) * df_calc['Score']
            elif 'peaks' in df_calc.columns: 
                 df_calc['signal_peaks'] = df_calc['peaks'].shift(peak_shift) * df_calc['Score']
                 df_calc['signal_bottoms'] = df_calc['valleys'].shift(peak_shift) * df_calc['Score']
            
            # Prepare for Backtest
            df_calc['time'] = df_calc['date'].dt.strftime('%Y-%m-%d')
            df_json_cols = df_calc[['time','open','high','low','close','Score','signal_peaks','signal_bottoms']] 
            df_json_cols = df_json_cols.iloc[:-2]
            
            df_bt = create_backtest_bier_df(df_json_cols)
            stats = get_backtest_stats_dict(df_bt, allow_short)
            
            # Prepare Row Data
            row_data = {
                'test_run': test_run_id,
                'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'name': metric,
                'start_date': start_date,
                'end_date': end_date,
                'signal_strategy': str(signal_strategy),
                'nb_metrics': len(metrics_list),
                'return': stats['return'],
                'max_dd': abs(stats['max_dd']),
                'sharpe': stats['sharpe'],
                'sortino': stats['sortino'],
                'calmar': stats['calmar']
            }
            
            # Initialize all test metric columns to 0
            for m in test_metrics:
                row_data[m.lower()] = 0.0
            
            # Set this metric to 1.0
            row_data[metric.lower()] = 1.0
            database.save_backtest_row(row_data)
            results.append(f"{metric}: {stats['return']:.2f}%")
            
            # Track profitable metrics with good Sharpe ratio
            if stats['sharpe'] > 0.7:
                profitable_metrics.append(metric)
                print(f"  ✓ Good Sharpe: {stats['sharpe']:.2f} (Return: {stats['return']:.2f}%)")
            else:
                print(f"  ✗ Low Sharpe: {stats['sharpe']:.2f} (Return: {stats['return']:.2f}%)")
            
        except Exception as e:
            print(f"Error testing {metric}: {e}")
    
    # PHASE 2: Test Combinations of Profitable Metrics
    if len(profitable_metrics) >= 2:
        print(f"\n=== PHASE 2: Testing Combinations of {len(profitable_metrics)} Metrics (Sharpe > 0.5) ===")
        print(f"Max combination size: {max_combination_size}")
        
        total_combos = 0
        for combo_size in range(2, min(max_combination_size + 1, len(profitable_metrics) + 1)):
            combos = list(combinations(profitable_metrics, combo_size))
            total_combos += len(combos)
            print(f"  Size {combo_size}: {len(combos)} combinations")
        
        print(f"Total combinations to test: {total_combos}\n")
        
        combo_count = 0
        for combo_size in range(2, min(max_combination_size + 1, len(profitable_metrics) + 1)):
            for combo in combinations(profitable_metrics, combo_size):
                combo_count += 1
                combo_str = " + ".join(combo)
                print(f"[{combo_count}/{total_combos}] Testing: {combo_str}")
                
                try:
                    metrics_list = list(combo)
                    
                    df_calc = matrix_strategy.calc_multi_strategy(df_base.copy(), 1, metrics_list, signal_strategy, use_alt_signal, alt_signal_deviation)
                    df_calc = df_calc.rename(columns={'invest_score': 'Score'})
                    
                    peak_shift = 1
                    if 'trigger_short' in df_calc.columns:
                        df_calc['signal_peaks'] = df_calc['trigger_short'].shift(peak_shift) * df_calc['Score']
                        df_calc['signal_bottoms'] = df_calc['trigger_long'].shift(peak_shift) * df_calc['Score']
                    elif 'peaks' in df_calc.columns:
                        df_calc['signal_peaks'] = df_calc['peaks'].shift(peak_shift) * df_calc['Score']
                        df_calc['signal_bottoms'] = df_calc['valleys'].shift(peak_shift) * df_calc['Score']
                    
                    df_calc['time'] = df_calc['date'].dt.strftime('%Y-%m-%d')
                    df_json_cols = df_calc[['time','open','high','low','close','Score','signal_peaks','signal_bottoms']] 
                    df_json_cols = df_json_cols.iloc[:-2]
                    
                    df_bt = create_backtest_bier_df(df_json_cols)
                    stats = get_backtest_stats_dict(df_bt, allow_short)
                    
                    # Prepare Row Data for combination
                    combo_name = "_".join(combo)
                    row_data = {
                        'test_run': test_run_id,
                        'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                        'name': combo_name,
                        'start_date': start_date,
                        'end_date': end_date,
                        'signal_strategy': str(signal_strategy),
                        'nb_metrics': len(metrics_list),
                        'return': stats['return'],
                        'max_dd': abs(stats['max_dd']),
                        'sharpe': stats['sharpe'],
                        'sortino': stats['sortino'],
                        'calmar': stats['calmar']
                    }
                    
                    # Initialize all test metric columns to 0
                    for m in test_metrics:
                        row_data[m.lower()] = 0.0
                    
                    # Set all metrics in combination to 1.0
                    for metric in combo:
                        row_data[metric.lower()] = 1.0
                    
                    database.save_backtest_row(row_data)
                    results.append(f"{combo_str}: {stats['return']:.2f}%")
                    print(f"  Result: {stats['return']:.2f}% (Sharpe: {stats['sharpe']:.2f})")
                    
                except Exception as e:
                    print(f"  Error testing combination: {e}")
    else:
        print(f"\n=== PHASE 2: Skipped (only {len(profitable_metrics)} profitable metrics) ===")
            
    print("\nBatch Backtest Completed.")
    return f"Completed. Tested {len(test_metrics)} individual + {total_combos if len(profitable_metrics) >= 2 else 0} combinations. Top results in Backtest Evaluation tab."  
