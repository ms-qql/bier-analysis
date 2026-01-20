import anvil.server

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
from scipy.signal import savgol_filter, find_peaks

from backtesting import Backtest, Strategy
from backtesting.test import SMA, GOOG
from backtesting.lib import TrailingStrategy

from . import matrix_strategy 
from . import datatable



class Matrix(TrailingStrategy):
    stop_loss_pct = 90 / 100 # means not using it
    trailing_sl_no_atr = 50 #14 - means not using it
    invest_share = 70 / 100    

    def init(self, invest_share_pct=70):
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
  peak_shift = 1 # Shift peak by x bars to reflect delayed peak recognition 
  #scores_list = ['nvts', 'mvrv','reserve_risk','rhodl_ratio','nupl', 'macro_index']
  
  df_categories = datatable.read_categories() # read once for all categories
  metrics_list, metrics_list_norm = datatable.load_category_list(category, metric='', df=df_categories) 
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