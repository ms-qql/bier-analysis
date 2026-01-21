

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
import math
import time
import os
from io import StringIO
from scipy.signal import find_peaks, savgol_filter
#from findpeaks import findpeaks
from scipy.stats import zscore
#import yfinance as yf

from . import database 
from . import database
from . import matrix_bot

# --------------------------------------------------------------------------------------------------------------------------------------------------------------

itc_list = ['market_cap','risk_level','btc_dominance','btc_dominance_no_stables']


norm_lookback = 0 #4 * 360 # use 4 year lookback for normalization (equals to one BTC cycle)


# ------------------------------ Support functions ---------------------------------------------------

def calc_ohlc(df):
    try:
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    except:
      pass
    # Set 'date' as the index
    df.set_index('date', inplace=True)
    # Resample and convert to OHLC
    ohlc_data = df['close'].resample('D').ohlc()
    # Drop the original 'close' column
    df.drop('close', axis=1, inplace=True)
    # Join the OHLC data with the original DataFrame
    df = df.join(ohlc_data)
    # Reset the index to bring 'date' back as a column
    df.reset_index(inplace=True)
    try:
      df['date'] = df['date'].dt.strftime('%Y-%m-%d')  
    except:
      pass 
    return df

def calc_performance(df):
    """Calculate Equity and Drawdown for Strategy vs Buy & Hold"""
    df = df.copy()
    try:
        df['date'] = pd.to_datetime(df['date'])
    except:
        pass
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
    
    # Calculate additional metrics
    # Annualization factor (assuming daily data)
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    years = days / 365.25
    
    # Strategy metrics
    strat_total_return = (df['strat_equity'].iloc[-1] / 100 - 1)
    strat_ann_return = (1 + strat_total_return) ** (1 / years) - 1 if years > 0 else 0
    strat_volatility = df['strat_returns'].std() * np.sqrt(252)  # Annualized volatility
    strat_downside_returns = df['strat_returns'][df['strat_returns'] < 0]
    strat_downside_dev = strat_downside_returns.std() * np.sqrt(252) if len(strat_downside_returns) > 0 else 0.0001
    strat_max_dd = abs(df['strat_dd'].min())
    
    # Buy & Hold metrics
    bh_total_return = (df['bh_equity'].iloc[-1] / 100 - 1)
    bh_ann_return = (1 + bh_total_return) ** (1 / years) - 1 if years > 0 else 0
    bh_volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
    bh_downside_returns = df['returns'][df['returns'] < 0]
    bh_downside_dev = bh_downside_returns.std() * np.sqrt(252) if len(bh_downside_returns) > 0 else 0.0001
    bh_max_dd = abs(df['bh_dd'].min())
    
    # Calculate ratios
    strat_sharpe = strat_ann_return / strat_volatility if strat_volatility > 0 else 0
    bh_sharpe = bh_ann_return / bh_volatility if bh_volatility > 0 else 0
    
    strat_sortino = strat_ann_return / strat_downside_dev if strat_downside_dev > 0 else 0
    bh_sortino = bh_ann_return / bh_downside_dev if bh_downside_dev > 0 else 0
    
    strat_calmar = (strat_ann_return * 100) / strat_max_dd if strat_max_dd > 0 else 0
    bh_calmar = (bh_ann_return * 100) / bh_max_dd if bh_max_dd > 0 else 0
    
    # Store metrics in a dictionary
    metrics = {
        'strat_sharpe': strat_sharpe,
        'bh_sharpe': bh_sharpe,
        'strat_sortino': strat_sortino,
        'bh_sortino': bh_sortino,
        'strat_calmar': strat_calmar,
        'bh_calmar': bh_calmar,
        'strat_max_dd': strat_max_dd,
        'bh_max_dd': bh_max_dd
    }
    
    return df, metrics
 

def calc_metric(start_date = "2020-01-01", end_date = "2026-01-01", metric='risk_level'):  
  print(f'Input: Metric: {metric}')
  if metric.lower() in itc_list:
    table_name = 'itc' 
  else:
    table_name = 'capriole2'
  df_in = database.read_table_date_range_cloud(table_name, start_date, end_date)  
  df = df_in[['date','close',metric]]
  df = calc_ohlc(df, time_frame='D')
  #print('DF Calc: ', df, df.info())
  return df.to_json()


def calc_metric_all(start_date = "2020-01-01", end_date = "2026-01-01", metric='risk_level', asset='btc'):   

  print(f'Calc {metric} from {start_date} to {end_date} for {asset.upper()}')
  matrix = matrix_strategy()

  if asset.lower() == 'btc':
    df_itc = database.read_table_date_range_cloud('itc', start_date, end_date)  
    df_capriole = database.read_table_date_range_cloud('capriole2', start_date, end_date) 
    #print('DF Capriole:', df_capriole.info())  
    #print(df_capriole.tail()) #.to_string())
    #df_bmp = database.read_table_date_range_cloud('bmp', start_date, end_date)
    df_bmp_full = database.read_table_date_range_cloud('bmp2', start_date, end_date)  
    #print('DF BMP_Full:')  
    #print(df_bmp_full.tail().to_string())
    #df_bmp_full.to_csv('bmp_full.csv')
    df_manta = database.read_table_date_range_cloud('manta', start_date, end_date)
    df_manta.rename(columns = {'mi_correlation_norm' : 'correlation', 'mi_low_beta_norm' : 'beta'}, inplace = True)  
    df_manta['funding_rate_all'] = 50 + df_manta['funding_pos']/2 - df_manta['funding_neg']/2
    df_augmento = database.read_table_date_range_cloud('augmento', start_date, end_date)  
    #df_hyblock = database.read_table_date_range_cloud('hyblock', start_date, end_date) 
    #df_etf = database.read_table_date_range_cloud('etf', start_date, end_date) 
    #df_etf['flow'] = df_etf['flow'].replace(0, pd.NA).ffill() # replace NA from weekend values with previous days' value
    #df_etf.rename(columns = {'flow' : 'etf_flow'}, inplace = True)   
    df_tv = database.read_table_date_range_cloud('macro', start_date, end_date)
    df_tv.rename(columns = {'pmi_norm' : 'pmi', 'dxy_norm' : 'dxy', 'liq_norm' : 'liquidity_tv', 'dfg_norm' : 'dfg', 'usdt_dom_norm' : 'usdt_dom'}, inplace = True)      
    print('Macro: ', df_tv.tail(3))      


    #metrics_bmp_list = ['date','close','nupl', 'sopr', 'mvrv', 'liquidity', 'reserve_risk', 'bitcoin_sentiment', 'addresses_in_profit', 'rhodl_ratio', 'miner_fee_pct', 'realized_price_sth', 'sth_supply', 'vdd_multiple', 'fear_greed']
    """metrics_bmp_full_list = ['date', 'close', 'nupl', 'sopr', 'mvrv', 'liquidity', 'reserve_risk', 'bitcoin_sentiment', 'addresses_in_profit',
                    'rhodl_ratio', 'miner_fee_pct', 'realized_price_sth','sth_supply',  'vdd_multiple', 'fear_greed', 'funding_rate', 'nvt', 'sth_mvrv',
                    'lth_mvrv','financial_stress', 'high_yield_credit', 'm2_yoy_change', 'yield_spread', 'btc_etf_flows', 'cycle_capital_flows', 'bitcoin_cycle_master',
                    'onchain_prediction', 'realized_price_lth', 'pi_cycle_oscillator',  'everything_indicator'] 
    metrics_capriole_list = ['date','macro_index', 'heater', 'oi_pct_mcap', 'nvts', 'production_cost', 'miner_margin', 'supply_delta', 'cdd_trend', 'apparent_demand', 
                                'energy_value', 'active_more_2y_percent', 'active_more_1y_percent', 'active_more_6m_percent', 'lth_supply_percent', 'yardstick', 'institutional_buying_excess', 'mnav_enterprise_mean', 'treasury_cvd_Z', 'buy_sell_ratio_roc',
                                'companies_buying_per_day', 'ave_debt_to_equity', 'ave_debt_to_enterprise_value', 'ave_debt_to_bitcoin_treasury', 'yield_mean', 'days_cover_enterprise_mean', 'sentiment_spread', 'active_manager_sentiment', 'business_outlook', 'equity_fear_greed', 
                                'equity_premium', 'm2_yoy_less_rates', 'gsr', 'insider_trades', 'market_breadth', 'sp_pcr', 'us_liquidity_yoy', 'usd_positioning', 'bbb_credit_spread', 'xly_xlp', 
                                'como_spx', 'junk_treasuries', 'weight_node_strategy', 'weight_midas_strategy', 'weight_spectrum_strategy', 'macro_index_eth', 'apparent_demand_eth', 'heater_eth', 'oi_pct_mcap_eth', 'active_more_2y_pct_eth', 
                                'active_more_1y_pct_eth', 'active_more_6m_pct_eth', 'cdd_trend_eth', 'speculation_index_eth', 'crypto_breadth'] """

    df_categories = database.read_categories() # read once for all categories
    print('DF Categories: ', df_categories.tail(3))
    metrics_bmp_list, metrics_bmp_full_list_norm = database.load_category_list('bmp', metric='', df=df_categories)
    metrics_bmp_full_list = ['date', 'close'] + metrics_bmp_list    
    print("Metrics BMP: ", metrics_bmp_full_list) 
    metrics_capriole_list, metrics_capriole_full_list_norm = database.load_category_list('capriole', metric='', df=df_categories)
    metrics_capriole_full_list = ['date'] + metrics_capriole_list    
    #print("Metrics Capriole: ", metrics_capriole_full_list)     
    metrics_manta_list, metrics_manta_full_list_norm = database.load_category_list('manta', metric='', df=df_categories)
    metrics_manta_full_list = ['date'] + metrics_manta_list    
    print("Metrics Manta: ", metrics_manta_full_list)     
    metrics_itc_list, metrics_itc_full_list_norm = database.load_category_list('itc', metric='', df=df_categories)  
    metrics_itc_full_list = ['date'] + metrics_itc_list 
    #print("Metrics ITC: ", metrics_itc_full_list)         
    metrics_tv_list, metrics_tv_full_list_norm = database.load_category_list('tv', metric='', df=df_categories)      
    metrics_tv_full_list = ['date'] + metrics_tv_list
    #print("Metrics TV: ", metrics_tv_full_list)            
    metrics_augmento_full_list = ['date','augmento'] 

    df = df_bmp_full[metrics_bmp_full_list]     
    df = pd.merge(df, df_capriole[metrics_capriole_full_list],on='date', how="outer")        
    df = pd.merge(df, df_itc[metrics_itc_full_list],on='date', how="outer")  # itc      
    df = pd.merge(df, df_manta[metrics_manta_full_list],on='date', how="outer")         
    df = pd.merge(df, df_augmento[metrics_augmento_full_list],on='date', how="outer") 
    #print('DF Augmento: ', df.tail(3))         
    df = pd.merge(df, df_tv[metrics_tv_full_list],on='date', how="outer")        
    print('DF Metrics: ', df.tail(5))   
    print('DF MVRV, RR: ', df[['date','mvrv','reserve_risk']].tail(8))    

    #df['liquidity_tv_norm'] = df['liquidity_tv_norm'].shift(40) # Shift by 40 days
    df['roi'] = (df['close'] - df['close'].shift(90)) / df['close'] * 100 
    df['liquidity_change'] = (df['liquidity'] - df['liquidity'].shift(90)) / df['liquidity'] * 100 
    df['liquidity_change'] = matrix.double_hull_ma(df['liquidity_change'], 10, 10) 
    df['realized_price_delta_sth'] = (df['close'] - df['realized_price_sth']) / df['realized_price_sth'] * 100
    df['realized_price_ratio'] = (df['realized_price_sth'] - df['realized_price_lth']) / df['realized_price_lth'] * 100    
    df['active_ratio'] = (df['active_more_6m_percent'] - df['lth_supply_percent']) / df['lth_supply_percent'] * 100    

    # Merge NVT data
    nvt_cond = [df['date'] < '2016-03-06',  df['date'] >= '2016-03-06'] 
    nvt_category = [df['nvt'], df['nvts']]  # BMP for old data, Capriole for newer 
    df['nvt_combi'] = np.select(nvt_cond, nvt_category) 
    
    # Calc Z-score for some metrics
    #df['mvrv'] = matrix.calc_zscore(df['mvrv'])
    #df['reserve_risk'] = matrix.calc_zscore(df['reserve_risk'])    
    #print('DF MVRV, RR Z-Score: ', df[['date','mvrv','reserve_risk']].tail(8))       
    #df['production_margin'] = matrix.calc_zscore(df['production_margin'])  # got errror in calc_zscore
    df['financial_stress'] = matrix.calc_zscore(df['financial_stress'])
    df['bitcoin_sentiment'] = matrix.calc_zscore(df['bitcoin_sentiment'])  
    #df['rhodl_ratio'] = matrix.calc_zscore(df['rhodl_ratio'])
    #df['sth_supply'] = matrix.calc_zscore(df['sth_supply'])     

    all_metrics_list = metrics_bmp_full_list[1:] + metrics_capriole_list + metrics_itc_list + metrics_manta_list + metrics_tv_list + metrics_augmento_full_list[1:] + ['roi', 'liquidity_change', 'nvt_combi', 'realized_price_delta_sth','active_ratio','realized_price_ratio'] # All metrics without date    

    print('DF_Strategy: ', df.tail(5))#, df.info())  
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/strategy.csv')
    #print('DF Calc: ', df[['date', 'close']], df.info())   

    df = calc_categories(df.to_json(), metric, all_metrics_list)
    print('Calc all: ', df, df.columns)
    df = calc_ohlc(df)
    #print('OHLC:', df[['date', 'open', 'high', 'low', 'close']])
    #df.to_csv('strategy_ohlc.csv')
  else:
    print(f'Risk matrix for {asset}')
    non_btc_metrics_list = ['risk_level','roi']
    df_itc = database.read_table_date_range_cloud('itc', start_date, end_date) 
    symbol = asset.lower() + '_usd'
    symbol_risk = symbol + '_risk'
    df = df_itc[['date', symbol, symbol_risk]].copy()
    df = df.rename(columns={symbol: "close", symbol_risk:"risk_level"})
    df['roi'] = (df['close'] - df['close'].shift(90)) / df['close'] * 100  
    # Calc norm values
    for metric in non_btc_metrics_list:
      norm_name = metric + '_norm'
      df[norm_name] = matrix.calc_norm(df[metric], norm_lookback)       
    df = calc_ohlc(df)       
    df = df.fillna(0)
    #df = df.replace([np.inf, -np.inf], 0)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/strategy.csv')       

  return df.to_json()


def calc_categories(calc_json, single_metric, metrics_list):  
  df = pd.read_json(calc_json, orient='records') 
  #metrics_bmp_list = ['nupl', 'sopr', 'mvrv', 'liquidity_change', 'reserve_risk', 'bitcoin_sentiment', 'addresses_in_profit', 'rhodl_ratio', 'miner_fee_pct', 'realized_price_sth', 'sth_supply', 'vdd_multiple', 'fear_greed']

  '''metrics_bmp_full_list = ['nupl', 'sopr', 'mvrv', 'liquidity', 'reserve_risk', 'bitcoin_sentiment', 'addresses_in_profit',
                  'rhodl_ratio', 'miner_fee_pct', 'realized_price_sth','sth_supply',  'vdd_multiple', 'fear_greed', 'funding_rate', 'nvt', 'sth_mvrv',
                  'lth_mvrv','financial_stress', 'high_yield_credit', 'm2_yoy_change', 'yield_spread', 'btc_etf_flows', 'cycle_capital_flows', 'bitcoin_cycle_master',
                  'onchain_prediction', 'realized_price_lth', 'pi_cycle_oscillator',  'everything_indicator'] 
  metrics_capriole_list = ['macro_index', 'heater', 'oi_pct_mcap', 'nvts', 'production_cost', 'miner_margin', 'supply_delta', 'cdd_trend', 'apparent_demand', 
                            'energy_value', 'active_more_2y_percent', 'active_more_1y_percent', 'active_more_6m_percent', 'lth_supply_percent', 'yardstick', 'institutional_buying_excess', 'mnav_enterprise_mean', 'treasury_cvd_Z', 'buy_sell_ratio_roc',
                            'companies_buying_per_day', 'ave_debt_to_equity', 'ave_debt_to_enterprise_value', 'ave_debt_to_bitcoin_treasury', 'yield_mean', 'days_cover_enterprise_mean', 'sentiment_spread', 'active_manager_sentiment', 'business_outlook', 'equity_fear_greed', 
                            'equity_premium', 'm2_yoy_less_rates', 'gsr', 'insider_trades', 'market_breadth', 'sp_pcr', 'us_liquidity_yoy', 'usd_positioning', 'bbb_credit_spread', 'xly_xlp', 
                            'como_spx', 'junk_treasuries', 'weight_node_strategy', 'weight_midas_strategy', 'weight_spectrum_strategy', 'macro_index_eth', 'apparent_demand_eth', 'heater_eth', 'oi_pct_mcap_eth', 'active_more_2y_pct_eth', 
                            'active_more_1y_pct_eth', 'active_more_6m_pct_eth', 'cdd_trend_eth', 'speculation_index_eth', 'crypto_breadth']

  metrics_manta_list = ['correlation', 'beta', 'funding']  
  metrics_hyblock_list = ['volume_delta', 'whale_retail', 'user_bot_ratio', 'usdt_premium', 'bid_ask_ratio', 'top_traders_long', 'market_order_size', 'market_order_count', 'limit_order_count', 'bid_ask_delta', 'long_liquidations', 'short_liquidations', 'oi_delta', 'bvol', 'dvol']
  metrics_list = metrics_bmp_full_list + ['risk_level','roi'] + metrics_capriole_list + metrics_manta_list + ['augmento'] + metrics_hyblock_list + ['nvt_combi'] #+ ['etf_flow'] #.pop(0) '''
  matrix = matrix_strategy()

  for metric in metrics_list:
    norm_name = metric + '_norm'
    df[norm_name] = matrix.calc_norm(df[metric], norm_lookback) 
  single_norm = single_metric + '_norm'


  # Merge Macro Index / Risk data
  macro_cond = [df['date'] <= '2015-01-14',  df['date'] >= '2016-03-06'] # '2015-01-14'
  macro_category = [df['risk_level_norm'], df['macro_index_norm']]  # BMP for old data, Capriole for newer 
  df['macro_combi_norm'] = np.select(macro_cond, macro_category)
  df['macro_combi_norm'] = df['macro_combi_norm'].fillna(0)

  # Invert some metrices
  df['financial_stress_norm'] = 100 - df['financial_stress_norm']  
  df['gsr'] = 100 - df['gsr']  
  df['como_spx'] = 100 - df['como_spx']
  df['m2_yoy_blend'] = (df['m2_yoy_less_rates'] + df['m2_yoy_change']) / 2
  
  df['close_norm'] = matrix.calc_norm(df['close'], norm_lookback)

  df_categories = database.read_categories() # read once for all categories
  print('DF Categories: ', df_categories.tail(3))  

  # Load category lists
  category_list = df_categories.columns.tolist()[6:]  # all categories except 'metric' and source columns
  category_list = [item + '_cat' for item in category_list] # add _cat to each category name
  #category_list = df_categories['category'].unique().tolist()
  print('Categories: ', category_list)    

  custom_cat_list, custom_cat_list_norm = database.load_category_list('custom', metric='', df=df_categories)  
  bier_cat_list, bier_cat_list_norm = database.load_category_list('bier', metric='', df=df_categories)   
  test_cat_list, test_cat_list_norm = database.load_category_list('test', metric='', df=df_categories)   
  capriole_cat_list, capriole_cat_list_norm = database.load_category_list('capriole', metric='', df=df_categories)  
  bmp_cat_list, bmp_cat_list_norm = database.load_category_list('bmp', metric='', df=df_categories)
  manta_cat_list, manta_cat_list_norm = database.load_category_list('manta', metric='', df=df_categories)  
  itc_cat_list, itc_cat_list_norm = database.load_category_list('itc', metric='', df=df_categories)  
  tv_cat_list, tv_cat_list_norm = database.load_category_list('tv', metric='', df=df_categories)  
  bmp_cat_list, bmp_cat_list_norm = database.load_category_list('bmp', metric='', df=df_categories)  
  strategy_cat_list, strategy_cat_list_norm = database.load_category_list('strategy', metric='', df=df_categories)
  market_cat_list, market_cat_list_norm = database.load_category_list('market', metric='', df=df_categories)  
  mining_cat_list, mining_cat_list_norm = database.load_category_list('mining', metric='', df=df_categories)
  macro_cat_list, macro_cat_list_norm = database.load_category_list('macro', metric='', df=df_categories)   
  shortterm_cat_list, shortterm_cat_list_norm = database.load_category_list('shortterm', metric='', df=df_categories)
  sentiment_cat_list, sentiment_cat_list_norm = database.load_category_list('sentiment', metric='', df=df_categories)  
  hodl_cat_list, hodl_cat_list_norm = database.load_category_list('hodl', metric='', df=df_categories)
  treasury_cat_list, treasury_cat_list_norm = database.load_category_list('treasury', metric='', df=df_categories)   
  supply_demand_cat_list, supply_demand_cat_list_norm = database.load_category_list('supply_demand', metric='', df=df_categories)
  eth_cat_list, eth_cat_list_norm = database.load_category_list('eth', metric='', df=df_categories)  
  alts_cat_list, alts_cat_list_norm = database.load_category_list('alts', metric='', df=df_categories)  
  print('Market Metrics: ', market_cat_list_norm)
  print('Macro Metrics: ', macro_cat_list_norm)
  print('Shortterm Metrics: ', shortterm_cat_list_norm)  
  
  # Add calculated metrics to categories
  #market_cat_list_norm += ['roi_norm']
  #macro_cat_list_norm += ['liquidity_change_norm']
  #shortterm_cat_list_norm += ['nvt_combi_norm']
  #print('Market Metrics enhanced: ', market_cat_list_norm)

  df['custom_cat'] = df[custom_cat_list_norm].mean(axis=1)  
  df['bier_cat'] = df[bier_cat_list_norm].mean(axis=1)    
  df['test_cat'] = df[test_cat_list_norm].mean(axis=1)   
  df['capriole_cat'] = df[capriole_cat_list_norm].mean(axis=1)  
  df['bmp_cat'] = df[bmp_cat_list_norm].mean(axis=1)   
  df['manta_cat'] = df[manta_cat_list_norm].mean(axis=1)    
  df['mining_cat'] = df[itc_cat_list_norm].mean(axis=1)  
  df['macro_cat'] = df[tv_cat_list_norm].mean(axis=1)   
  df['strategy_cat'] = df[strategy_cat_list_norm].mean(axis=1)  
  df['market_cat'] = df[market_cat_list_norm].mean(axis=1)   
  df['mining_cat'] = df[mining_cat_list_norm].mean(axis=1)  
  df['macro_cat'] = df[macro_cat_list_norm].mean(axis=1)   
  df['shortterm_cat'] = matrix.double_hull_ma(df[shortterm_cat_list_norm].mean(axis=1) , 10, 10) 
  df['sentiment_cat'] = df[sentiment_cat_list_norm].mean(axis=1)    
  df['hodl_cat'] = df[hodl_cat_list_norm].mean(axis=1)  
  df['treasury_cat'] = df[treasury_cat_list_norm].mean(axis=1)  
  df['supply_demand_cat'] = df[supply_demand_cat_list_norm].mean(axis=1)   
  df['eth_cat'] = df[eth_cat_list_norm].mean(axis=1)    
  df['alts_cat'] = df[alts_cat_list_norm].mean(axis=1)    
   
  #df['single_cat'] = df[single_norm]
  #df['custom_cat'] = df[custom_cat_list].mean(axis=1)   

  #category_list = ['strategy_cat','market_cat', 'macro_cat', 'shortterm_cat', 'sentiment_cat', 'hodl_cat', 'manta_cat'] #,'single_cat']

  #print('DF_Cat: ', df_cat, df_cat.info())
  
  for category in category_list:
    category_ma = category + '_ma'
    df[category_ma] = matrix.double_hull_ma(df[category], 5, 5)

  return df


def calc_peaks_valleys(df, score, peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False):   
    matrix = matrix_strategy()
    df = matrix.find_peaks_valleys(df, score, peak_min, vert_dist, peak_dist, peak_width, peak_prominence, filt_double_extremes)      
    return df

def calc_strategy(df, peak_shift, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight):
  matrix = matrix_strategy()
                                     
  total_weight = sum((market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight))
  df['strategy_flex'] = (market_weight * df['market_cat'] + mining_weight * df['mining_cat'] + macro_weight * df['macro_cat'] + sentiment_weight * df['sentiment_cat'] + hodl_weight * df['hodl_cat'] + shortterm_weight * df['shortterm_cat'] + custom_weight * df['custom_cat'] + single_weight * df['single_cat']) / total_weight
  #total_weight = sum((risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight))
  #df['strategy_flex'] = (risk_weight * df['risk_cat'] + market_weight * df['market_cat'] + mining_weight * df['mining_cat'] + macro_weight * df['macro_cat'] + sentiment_weight * df['sentiment_cat'] + hodl_weight * df['hodl_cat'] + shortterm_weight * df['shortterm_cat'] + custom_weight * df['custom_cat'] + single_weight * df['single_cat']) / total_weight  
  df['strategy_ma'] = matrix.double_hull_ma(df['strategy_flex'], 5, 5) 

  df = calc_peaks_valleys(df, 'strategy_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)   
  #peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition    
  df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
  df['extremes'].replace(0, np.nan, inplace=True)
  df['extremes'] = df['extremes'].ffill(axis ='rows') 
  df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)
  
  return df


def calc_single_strategy(df_single, peak_shift, metric):
  matrix = matrix_strategy()

  invest_name = metric + '_invested'
  norm_name = metric + '_norm'
  df_single[norm_name] = matrix.calc_norm(df_single[metric], norm_lookback) 

  # Merge Macro Index / Risk data
  if metric == 'macro_index':
    macro_cond = [df_single['date'] <= '2015-02-09',  df_single['date'] >= '2016-03-06'] # '2015-01-14'
    macro_category = [df_single['risk_level_norm'], df_single['macro_index_norm']]  # BMP for old data, Capriole for newer 
    df_single['macro_index_norm'] = np.select(macro_cond, macro_category)
    df_single['macro_index_norm'] = df_single['macro_index_norm'].fillna(0)


  df_single['strategy_ma'] = matrix.double_hull_ma(df_single[norm_name], 8, 8) 
  #df_single['strategy_ma'] = matrix.double_hull_ma(df_single[norm_name], 5, 5) 
  df_single = calc_peaks_valleys(df_single, 'strategy_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)   
  df_single['extremes'] = df_single['peaks'].shift(peak_shift).fillna(0) - df_single['valleys'].shift(peak_shift).fillna(0)
  df_single['extremes'].replace(0, np.nan, inplace=True)
  df_single['extremes'] = df_single['extremes'].ffill(axis ='rows')  
  df_single[invest_name] = np.where((df_single['extremes'] < 0), 1, np.nan)
  df_single[invest_name] 
  df_single['trigger_short'] = df_single['peaks']
  df_single['trigger_long'] = df_single['valleys']
  os.makedirs('data', exist_ok=True)
  df_single.to_csv(f'data/calc_single_strategy_{metric}.csv')  
  
  return df_single



def calc_trade_status(df):
    trigger_level = 50  
    # Create the trigger_long and trigger_short columns
    df['trigger_long'] = ((df['invest_score'] > trigger_level) & (df['invest_score'].shift(1) <= trigger_level)).astype(int)
    df['trigger_short'] = ((df['invest_score'] < trigger_level) & (df['invest_score'].shift(1) >= trigger_level)).astype(int)

    # Create the invest column
    df['invested'] = np.nan
    last_trigger = np.nan  # Initial state is 0 (no position)
    for index, row in df.iterrows():
        if row['trigger_long'] == 1:
            last_trigger = 1
        elif row['trigger_short'] == 1:
            last_trigger = np.nan
        df.at[index, 'invested'] = last_trigger

    return df

def calc_multi_strategy(df, peak_shift, scores_list, use_signal=True):
  #scores_list = ['nvts', 'mvrv','reserve_risk','rhodl_ratio','nupl', 'macro_index']
  print('Scores_List: ', scores_list)
  if use_signal:
    # Use adding of scores method    
    invested_list = [] 
    # Run metrics in list
    for item in scores_list:
      item_invested = item + '_invested'
      invested_list.append(item_invested) # create invested list
      df = calc_single_strategy(df, peak_shift, item)
    df['invest_score'] = df[invested_list].sum(axis = 1)
    if len(scores_list) > 0:
        df['invest_score'] = df['invest_score'].fillna(0) * 100 / len(scores_list)  
        df = calc_trade_status(df)
        mid_score = 100 / len(scores_list) * round(len(scores_list) / 2)
        print(f'Nb metrics: {len(scores_list)} with mean: {mid_score}')
        df['range'] = np.where((df['invest_score'] == mid_score), 1, np.nan)    
    else:
        df['invest_score'] = 0
        df['invested'] = np.nan
        df['range'] = np.nan
    
  else: 
    # Use averaging method
    matrix = matrix_strategy()
    scores_list_norm = [item + '_norm' for item in scores_list]
    print('Scores_List: ', scores_list)
    if len(scores_list) > 0:
        df['strategy'] = df[scores_list_norm].mean(axis=1)   
    else:
        df['strategy'] = 0
    df['invest_score'] = matrix.double_hull_ma(df['strategy'], 5, 5) 
    df = calc_peaks_valleys(df, 'invest_score', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)   
    df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
    df['extremes'].replace(0, np.nan, inplace=True)
    df['extremes'] = df['extremes'].ffill(axis ='rows') 
    df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)    

  df['trigger_short'] = df['peaks']
  df['trigger_long'] = df['valleys']
  os.makedirs('data', exist_ok=True)
  df.to_csv('data/calc_multi_strategy.csv')   

  return df



class matrix_strategy():
  
    ''' Class for calculating matrix strategy.
    
    Attributes
    ============
    df_api - DataFrame of Binance API keys and secrets for all users
      
    Methods
    =======
    read_api_key_secret(self, user) - reads the api key and secret for specific user
    
    '''    
    
    def __init__(self):        
        self.results = None
        self.zscore_max = 2.5
        self.timeframe = '4h'      
        #self.get_data()

    def calc_score_multi_tf(self, start_date, end_date, asset_in, strategy_in, time_frame, hull_window):
      norm_window = 24 * 30 * 3 # sliding window for calc of normalization  
      asset = asset_in[0:3].lower()
      strategy = strategy_in.lower() 
      if asset == 'btc':
        table_name = 'hyblock_neo'
      else:
        table_name = 'hyblock_' + asset

      df_in = database.read_table_date_range_cloud(table_name, start_date, end_date)   
      df_in = df_in[['time','open','high','low','close','volume_delta']]
      #df_in = df_in.reset_index(drop=True)    
      #print('Time Frame: ', time_frame)
      #print('DF_Multi: ', df_in.tail(10), df_in.columns)
    
      # Adapt timeframe, if required
      if time_frame != '5Min':
        df = database.resample_df(df_in, time_frame=time_frame)
      else:
        df = df_in.copy()
      
      df = df.fillna(method='ffill')       
      hyblock_list = ['volume_delta']       
        
      for indicator in hyblock_list:         
        name_zscore = asset + '-' + indicator + '-zscore' 
        df[name_zscore] = zscore(df[indicator]).clip(-self.zscore_max, self.zscore_max)   # calc Z-Score  and limit to x standard deviations         
        name_norm = asset + '-' + indicator + '-norm'   
        df[name_norm] = 50 + 50 / self.zscore_max * df[name_zscore]   # norm value based on z-score                 

      #print('Strategy DF:', df.columns)
      # Calculation of the deep-dive category scores 
      vol_name_norm = asset + '-volume_delta-norm'         
      #hull_window = 40 #30     

      df['score_hyblock_raw'] = df[vol_name_norm]        
      df['score_hyblock'] = self.double_hull_ma(df['score_hyblock_raw'], hull_window, hull_window)  
      df['score_hyblock_multi'] = self.calc_norm(df['score_hyblock'], 300) # was 500 
      
      df.sort_values (by='time', inplace=True)
      return df
  
    def calc_stop_loss(self, df, direction, sl_pct: float = 0.03):
      if direction == 'short':
        stop_loss = df['close'].iloc[-1] * (1 + sl_pct)
      else:
        stop_loss = df['close'].iloc[-1] * (1 - sl_pct)        
      return stop_loss

    def calc_atr(self, df, periods: int = 100):
      """
      Set the lookback period for computing ATR. The default value
      of 100 ensures a _stable_ ATR.
      """
      df['Close'] = df['close']
      df['Open'] = df['Close'].shift(1)    
      df['Open'] = df['Open'].fillna(df['Close'])    
      df['High'] = df[['Open', 'Close']].max(axis=1)
      df['Low'] =  df[['Open', 'Close']].min(axis=1)
      hi, lo, c_prev = df.High, df.Low, pd.Series(df.Close).shift(1)
      tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
      atr = pd.Series(tr).rolling(periods).mean().bfill().values
      #print('ATR: ', atr)
      return atr[-1]  

    def calc_initial_trailing_sl(self, df, direction, nb_atr: int = 14):
      close = df['close'].iloc[-1]
      atr_sl = self.calc_atr(df) * nb_atr
      if direction == 'short':
        trailing_sl = close + atr_sl
      else:
        trailing_sl = close - atr_sl   
      return trailing_sl

    def set_initial_risk_mgmt(self, df, direction, sl_pct=0.03, nb_atr=14):
      sl = self.calc_stop_loss(df, direction, sl_pct)
      print('Stop Loss Long: ', sl)
      trailing_sl = self.calc_initial_trailing_sl(df, direction, nb_atr)
      database.delete_risk_mgmt_table('btc')      
      database.initial_row_risk_mgmt('btc', sl, trailing_sl)   
      return
      
    def update_trailing_sl(self, df, direction, nb_atr: int = 14):
      close = df['close'].iloc[-1]
      atr_sl = self.calc_atr(df) * nb_atr
      trade_sl = database.read_trailing_sl('btc')
      #print(f"ATR SL: {atr_sl}, Trade SL: {trade_sl} ")
      if direction == 'short':
        trailing_sl = min(close + atr_sl, trade_sl)
      else:
        trailing_sl = max(close - atr_sl, trade_sl)

      trail_sl_calc = close - atr_sl
      print(f'existing Trail-SL: {trade_sl}, close: {close}, atr_sl: {atr_sl} --> calc TSL: {trail_sl_calc}')
      if trailing_sl != trade_sl:
        sl = database.read_sl('btc')# read sl value
        database.update_risk_mgmt_table('btc', sl, trailing_sl)
      return trailing_sl

    def hull_ma(self, src, ma_length):
      # df column used for calculation is: input
      # results stored in added column 'hma_temp' and returned      
      #print(df[input_col])  
      src = src.fillna(0)  
      ema_half = src.ewm(span=ma_length/2, adjust=False).mean()  
      ema_length = src.ewm(span=ma_length, adjust=False).mean()    
      hma_calc = 2 * ema_half - ema_length
      hma = hma_calc.ewm(span=math.sqrt(ma_length), adjust=False).mean()
      return hma

    def double_hull_ma(self, src, window_1, window_2):
        # calc peaks
        return self.hull_ma(self.hull_ma(src, window_1),window_2)

    def filt_double_peaks_valleys(self, df, peak_col, valley_col):
      # New col with latest extreme status   
      df['extremes'] = df[peak_col].fillna(0) - df[valley_col].fillna(0)
      df['extremes'].replace(0, np.nan, inplace=True)
      df['extremes'] = df['extremes'].ffill(axis ='rows') 
      
      df[peak_col] = np.where((df[peak_col] > 0) & (df['extremes'].shift() < 1), 1, np.nan)
      df[valley_col] = np.where((df[valley_col] > 0) & (df['extremes'].shift() > -1), 1, np.nan)        
      return df
  
    def find_peaks_valleys(self, df, score, peak_min, vert_dist, peak_dist, peak_width, peak_prominence, filt_double_extremes):
        #display('DF Peaks',df)
        peak_col = 'peaks' #+timeframe # name of peak column in df for specific timeframe    
        valley_col = 'valleys' #+timeframe # name of valley column in df for specific timeframe
        valley_min = -(100 - peak_min)
        peaks, _ = find_peaks(x=df[score], height=peak_min, threshold=vert_dist, distance=peak_dist, width=peak_width, prominence=peak_prominence)
        valleys, _ = find_peaks(x=-df[score], height=valley_min, threshold=vert_dist, distance=peak_dist, width=peak_width, prominence=peak_prominence)
        df[peak_col] = np.nan
        df.loc[peaks, peak_col] = 1 # if peak then list value 1
        #df[peak_col] = df[score].iloc[peaks] # if peak then list score value
        df[valley_col] = np.nan
        df.loc[valleys, valley_col] = 1 # if valley then list value 1      
        #df[valley_col] = df[score].iloc[valleys] # if valley then list score value    
        if filt_double_extremes:
          self.filt_double_peaks_valleys(df, peak_col, valley_col)
        return df 

    def calc_zscore(self, src):
      zscore_max = 2.5
      src_z = zscore(src).clip(-zscore_max, zscore_max)   # calc Z-Score  and limit to x standard deviations
      return src_z
    
    def calc_rsi(self, df, price_column='close', period=14):
      """  Calculate Relative Strength Index (RSI) for a given price column in a DataFrame.
      
      Parameters:
      -----------
      price_column : str - The column name for the price data (default: 'close')
      period : int - The period for calculating RSI (default: 14)
      
      Returns:
      --------
      pandas.Series - Series containing RSI values   """
      
      # Create a DataFrame from the series
      print('RSI_DF: ', df, df.info())

      # Create a copy of the dataframe to avoid modifying the original
      df_copy = df.copy()
      # Calculate price change
      delta = df_copy[price_column].diff()
      
      # Calculate gains and losses
      gain = delta.copy()
      loss = delta.copy()
      gain[gain < 0] = 0
      loss[loss > 0] = 0
      loss = abs(loss)
      
      # Calculate average gain and average loss
      avg_gain = gain.rolling(window=period).mean()
      avg_loss = loss.rolling(window=period).mean()
      
      # Calculate RS (Relative Strength)
      rs = avg_gain / avg_loss
      
      # Calculate RSI
      rsi = 100 - (100 / (1 + rs))
      
      return rsi

    def calc_stoch_rsi(self, df, price_column='close', rsi_period=14, stoch_period=14, smoothK=3, smoothD=3):
        """
        Calculate Stochastic RSI for a given price column in a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame - DataFrame containing the price data
        price_column : str - The column name for the price data (default: 'close')
        rsi_period : int - The period for calculating RSI (default: 14)
        stoch_period : int - The period for calculating Stochastic RSI (default: 14)
        smoothK : int - The period for the %K smoothing (default: 3)
        smoothD : int - The period for the %D smoothing (default: 3)
        
        Returns:
        --------
        pandas.DataFrame - Original DataFrame with additional columns: 'rsi', 'stoch_rsi', 'stoch_rsi_k', 'stoch_rsi_d'      """

        # Create a copy of the dataframe to avoid modifying the original
        df_result = df.copy()
        
        # Calculate RSI using the separate function
        rsi = self.calc_rsi(df_result, price_column, rsi_period)
        df_result['rsi'] = rsi
        
        # Calculate Stochastic RSI
        stoch_rsi = (rsi - rsi.rolling(window=stoch_period).min()) / (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())
        df_result['stoch_rsi'] = stoch_rsi
        
        # Calculate smoothed Stochastic RSI %K
        df_result['stoch_rsi_k'] = stoch_rsi.rolling(window=smoothK).mean()
        
        # Calculate smoothed Stochastic RSI %D
        df_result['stoch_rsi_d'] = df_result['stoch_rsi_k'].rolling(window=smoothD).mean()
        
        return df_result

    def calc_norm(self, src, norm_window=0):
      if norm_window == 0: 
        value_min = src.min()
        value_max = src.max()  
      else: 
        value_min = src.rolling(norm_window).min()
        value_max = src.rolling(norm_window).max()
      
      src_norm = (src - value_min) / (value_max - value_min) * 100 
      return src_norm

# --------- Enhanced Dual Signal Investment Strategy ---------

def dual_signal_investment(score_series, lookback_window=50, momentum_window=10, value_base_weigth=0.3, value_extreme_weigth=0.3):
    """
    Enhanced version with adaptive momentum scaling and volatility adjustment.
    """
    
    score_series = pd.Series(score_series) if not isinstance(score_series, pd.Series) else score_series
    
    # Calculate rolling statistics
    rolling_std = score_series.rolling(window=lookback_window).std()
    score_momentum = score_series.diff(periods=momentum_window)
    
    # Value signal (same as basic version)
    value_signal = (score_series - 50) / 2.5
    
    # Adaptive momentum signal based on recent volatility
    # Higher volatility = lower momentum weight to reduce noise
    momentum_scaling = 10 / (1 + rolling_std.fillna(rolling_std.mean()))
    momentum_signal = np.clip(score_momentum * momentum_scaling, -10, 10)
    
    # Dynamic weights based on score extremes
    # At extremes, increase value weight; in middle range, increase momentum weight
    extreme_factor = np.abs(score_series - 50) / 50  # 0 to 1
    dynamic_value_weight = value_base_weigth + value_extreme_weigth * extreme_factor  # 0.3 to 0.8
    dynamic_momentum_weight = 1 - dynamic_value_weight
    
    # Combined signal
    combined_signal = (dynamic_value_weight * value_signal + 
                      dynamic_momentum_weight * momentum_signal)
    
    investment_level = np.clip(combined_signal, -20, 20)
    
    return investment_level


def create_forecast(df, score_column='strategy_score', momentum_window=10):
    """
    Apply dual signal strategy to a DataFrame with your data.
    """
    
    df = df.copy()
    df['forecast'] = dual_signal_investment(df[score_column], lookback_window=50, momentum_window=momentum_window)
    #df['position_change'] = df['forecast'].diff().abs()
    df['invested_ds'] = np.where(df['forecast'] > 0, 1, np.nan)  # Invested if forecast > 0
    
    # Add signal components for analysis
    df['value_signal'] = (df[score_column] - 50) / 2.5
    df['momentum_signal'] = np.clip(df[score_column].diff(momentum_window) * 10, -10, 10)
    
    return df


def store_strategy(start_date, end_date, metric, asset, category, signal_strategy=False): 
    peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition
    # Calc overall strategy
    calc_json = calc_metric_all(start_date, end_date, metric, asset) 
    df = pd.read_json(StringIO(calc_json), orient='records') 
    
    df_categories = database.read_categories() # read once for all categories
    metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 

    df = calc_multi_strategy(df, peak_shift, metrics_list, signal_strategy)

    df_bier = df[['date', 'close', 'invest_score', 'peaks', 'valleys', 'extremes', 'invested']].copy()
    df_bier = df_bier.fillna(0)
    os.makedirs('data', exist_ok=True)
    df_bier.to_csv('data/bier_strategy.csv')
    database.store_metric_table(df_bier, 'bier2') # store calculated metrics in database  
    print('Bier strategy stored: ', df_bier.tail(3)) #, df_bier.info())
