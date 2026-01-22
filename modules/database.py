

import psycopg2
import numpy as np
import pandas as pd
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
load_dotenv()

from . import date_time
import time


# This module follows the CRUD (Create, Read, Update, Delete) structure

DATABASE_URL = os.environ['BIER4_DB_URL']
#DATABASE_URL_HYBLOCK_1H = os.environ['BIER4_DB_URL_HYBLOCK']

CREATE_ITC_TABLE = """CREATE TABLE IF NOT EXISTS itc (id SERIAL, date TEXT, mcap REAL, mcap_risk REAL, close REAL, risk_level REAL, alt_mcap REAL, alt_mcap_risk REAL, dxy_usd REAL, dxy_usd_risk REAL, 
                      eth_usd REAL, eth_usd_risk REAL, bnb_usd REAL, bnb_usd_risk REAL, sol_usd REAL, sol_usd_risk REAL, ltc_usd REAL, ltc_usd_risk REAL);"""
CREATE_CAPRIOLE_TABLE = """CREATE TABLE IF NOT EXISTS capriole2 (id SERIAL, date TEXT, macro_index REAL, heater REAL, oi_pct_mcap REAL, nvts REAL, production_cost REAL, miner_margin REAL,
                       supply_delta REAL, cdd_trend REAL, apparent_demand REAL, energy_value REAL, active_more_2y_percent REAL, active_more_1y_percent REAL, active_more_6m_percent REAL, lth_supply_percent REAL,
                       yardstick REAL, institutional_buying_excess REAL, mnav_enterprise_mean REAL, treasury_cvd_Z REAL, buy_sell_ratio_roc REAL, companies_buying_per_day REAL, ave_debt_to_equity REAL,
                       ave_debt_to_enterprise_value REAL, ave_debt_to_bitcoin_treasury REAL, yield_mean REAL, days_cover_enterprise_mean REAL, sentiment_spread REAL, active_manager_sentiment REAL,
                       business_outlook REAL, equity_fear_greed REAL, equity_premium REAL, m2_yoy_less_rates REAL, gsr REAL, insider_trades REAL, market_breadth REAL, sp_pcr REAL, us_liquidity_yoy REAL,
                       usd_positioning REAL, bbb_credit_spread REAL, xly_xlp REAL, como_spx REAL, junk_treasuries REAL, weight_node_strategy REAL, weight_midas_strategy REAL, weight_spectrum_strategy REAL,
                       macro_index_eth REAL, apparent_demand_eth REAL, heater_eth REAL, oi_pct_mcap_eth REAL, active_more_2y_pct_eth REAL, active_more_1y_pct_eth REAL, active_more_6m_pct_eth REAL,
                       cdd_trend_eth REAL, speculation_index_eth REAL, crypto_breadth REAL);"""

CREATE_BMP_TABLE = """CREATE TABLE IF NOT EXISTS bmp (id SERIAL, date TEXT, close REAL, nupl REAL, sopr REAL, mvrv REAL, liquidity REAL, reserve_risk REAL, bitcoin_sentiment REAL,
                      addresses_in_profit REAL, rhodl_ratio REAL, miner_fee_pct REAL, realized_price_sth REAL, sth_supply REAL, vdd_multiple REAL, fear_greed REAL, funding_rate REAL);"""
CREATE_BMP2_TABLE = """CREATE TABLE IF NOT EXISTS bmp2 (id SERIAL, date TEXT, close REAL, nupl REAL, sopr REAL, mvrv REAL, liquidity REAL, reserve_risk REAL, bitcoin_sentiment REAL,
                      addresses_in_profit REAL, rhodl_ratio REAL, miner_fee_pct REAL, realized_price_sth REAL, sth_supply REAL, vdd_multiple REAL, fear_greed REAL, funding_rate REAL,
                      nvt REAL, sth_mvrv REAL, lth_mvrv REAL, financial_stress REAL, high_yield_credit REAL, m2_yoy_change REAL, yield_spread REAL, btc_etf_flows REAL, cycle_capital_flows REAL,
                      bitcoin_cycle_master REAL, onchain_prediction REAL, realized_price_lth REAL, pi_cycle_oscillator REAL,  everything_indicator REAL);""" # 7,8,9,5 = 29
CREATE_MANTA_TABLE = """CREATE TABLE IF NOT EXISTS manta_data (id SERIAL, date TEXT, price_usd_close REAL, itc_risk_level_norm REAL, mi_correlation_norm REAL, mi_low_beta_norm REAL, dxy REAL, dxy_norm REAL, funding_neg REAL, funding_pos REAL);"""
CREATE_AUGMENTO_TABLE = """CREATE TABLE IF NOT EXISTS augmento (id SERIAL, date TEXT, bitcointalk REAL, reddit REAL, twitter REAL, augmento REAL);"""
CREATE_HYBLOCK_TABLE = """CREATE TABLE IF NOT EXISTS hyblock (id SERIAL, date TEXT, volume_delta REAL, whale_retail REAL, user_bot_ratio REAL, usdt_premium REAL, bid_ask_ratio REAL, 
                          top_traders_long REAL, market_order_size REAL, market_order_count REAL, limit_order_count REAL, funding_rate REAL, fear_greed REAL, bid_ask_delta REAL, long_liquidations REAL, 
                          short_liquidations REAL , oi_delta REAL, bvol REAL, dvol REAL);"""
CREATE_ETF_TABLE = """CREATE TABLE IF NOT EXISTS etf_flow (id SERIAL, date TEXT, flow REAL, close REAL);"""
CREATE_BIER_TABLE = """CREATE TABLE IF NOT EXISTS bier (id SERIAL, date TEXT, close REAL, bier_invested REAL, bier_range REAL);"""
CREATE_BIER2_TABLE = """CREATE TABLE IF NOT EXISTS bier2 (id SERIAL, date TEXT, close REAL, invest_score REAL, peaks REAL, valleys REAL, extremes REAL, invested REAL);"""
CREATE_MACRO_TABLE = """CREATE TABLE IF NOT EXISTS macro_norm (id SERIAL, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, pmi_norm REAL, dxy_norm REAL, liq_norm REAL, dfg_norm REAL, usdt_dom_norm REAL);"""
CREATE_BIER_CATEGORIES_TABLE = """CREATE TABLE IF NOT EXISTS bier_categories (
    id SERIAL PRIMARY KEY,
    metric TEXT,
    capriole INTEGER,
    bmp INTEGER,
    manta INTEGER,
    itc INTEGER,
    tv INTEGER,
    strategy INTEGER,
    market INTEGER,
    mining INTEGER,
    macro INTEGER,
    shortterm INTEGER,
    sentiment INTEGER,
    hodl INTEGER,
    treasury INTEGER,
    supply_demand INTEGER,
    eth INTEGER,
    alts INTEGER,
    custom INTEGER,
    bier INTEGER,
    test INTEGER
);"""
CREATE_BIER_CATEGORY_WEIGHT_TABLE = """CREATE TABLE IF NOT EXISTS bier_category_weight (
    id SERIAL PRIMARY KEY,
    strategy INTEGER,
    market INTEGER,
    mining INTEGER,
    macro INTEGER,
    shortterm INTEGER,
    sentiment INTEGER,
    hodl INTEGER,
    treasury INTEGER,
    supply_demand INTEGER,
    eth INTEGER
);"""

SELECT_ALL_ROWS_ITC_TABLE = "SELECT * FROM itc ORDER BY date;"
SELECT_DATE_ROWS_ITC_TABLE = "SELECT * FROM itc WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_ITC_TABLE = "SELECT * FROM itc WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_ITC_TABLE = "SELECT * FROM itc ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_CAPRIOLE_TABLE = "SELECT * FROM capriole2 ORDER BY date;"
SELECT_DATE_ROWS_CAPRIOLE_TABLE = "SELECT * FROM capriole2 WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_CAPRIOLE_TABLE = "SELECT * FROM capriole2 WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_CAPRIOLE_TABLE = "SELECT * FROM capriole2 ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_BMP_TABLE = "SELECT * FROM bmp ORDER BY date;"
SELECT_DATE_ROWS_BMP_TABLE = "SELECT * FROM bmp WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_BMP_TABLE = "SELECT * FROM bmp WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_BMP_TABLE = "SELECT * FROM bmp ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_BMP2_TABLE = "SELECT * FROM bmp2 ORDER BY date;"
SELECT_DATE_ROWS_BMP2_TABLE = "SELECT * FROM bmp2 WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_BMP2_TABLE = "SELECT * FROM bmp2 WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_BMP2_TABLE = "SELECT * FROM bmp2 ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_MANTA_TABLE = "SELECT * FROM manta_data ORDER BY date;"
SELECT_FILTER_ROWS_MANTA_TABLE = "SELECT * FROM manta_data WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_DATE_RANGE_MANTA_TABLE = "SELECT * FROM manta_data WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_Last_X_ROWS_MANTA_TABLE = "SELECT * FROM manta_data ORDER BY id DESC LIMIT "
SELECT_Last_ROW_MANTA_TABLE = "SELECT * FROM manta_data WHERE (date <= %s) ORDER BY date DESC LIMIT 1;"

SELECT_ALL_ROWS_AUGMENTO_TABLE = "SELECT * FROM augmento ORDER BY date;"
SELECT_DATE_ROWS_AUGMENTO_TABLE = "SELECT * FROM augmento WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_AUGMENTO_TABLE = "SELECT * FROM augmento WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_AUGMENTO_TABLE = "SELECT * FROM augmento ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_HYBLOCK_TABLE_1H = "SELECT * FROM hyblock_data ORDER BY time;"
SELECT_DATE_ROWS_HYBLOCK_TABLE_1H = "SELECT * FROM hyblock_data WHERE (time >= %s) ORDER BY time;"
SELECT_DATE_RANGE_HYBLOCK_TABLE_1H = "SELECT * FROM hyblock_data WHERE (time >= %s and time <= %s) ORDER BY time;"
SELECT_Last_X_ROWS_HYBLOCK_TABLE_1H = "SELECT * FROM hyblock_data ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_HYBLOCK_TABLE = "SELECT * FROM hyblock ORDER BY date;"
SELECT_DATE_ROWS_HYBLOCK_TABLE = "SELECT * FROM hyblock WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_HYBLOCK_TABLE = "SELECT * FROM hyblock WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_Last_X_ROWS_HYBLOCK_TABLE = "SELECT * FROM hyblock ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_ETF_TABLE = "SELECT * FROM etf_flow ORDER BY date;"
SELECT_DATE_ROWS_ETF_TABLE = "SELECT * FROM etf_flow WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_ETF_TABLE = "SELECT * FROM etf_flow WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_ETF_TABLE = "SELECT * FROM etf_flow ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_BIER_TABLE = "SELECT * FROM bier ORDER BY date;"
SELECT_DATE_ROWS_BIER_TABLE = "SELECT * FROM bier WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_BIER_TABLE = "SELECT * FROM bier WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_BIER_TABLE = "SELECT * FROM bier ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_BIER2_TABLE = "SELECT * FROM bier2 ORDER BY date;"
SELECT_DATE_ROWS_BIER2_TABLE = "SELECT * FROM bier2 WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_BIER2_TABLE = "SELECT * FROM bier2 WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_BIER2_TABLE = "SELECT * FROM bier2 ORDER BY id DESC LIMIT "

SELECT_ALL_ROWS_MACRO_TABLE = "SELECT * FROM macro_norm ORDER BY date;"
SELECT_DATE_ROWS_MACRO_TABLE = "SELECT * FROM macro_norm WHERE (date >= %s) ORDER BY date;"
SELECT_DATE_RANGE_MACRO_TABLE = "SELECT * FROM macro_norm WHERE (date >= %s and date <= %s) ORDER BY date;"
SELECT_LAST_X_ROWS_MACRO_TABLE = "SELECT * FROM macro_norm ORDER BY id DESC LIMIT "
SELECT_ALL_ROWS_BIER_CATEGORIES_TABLE = "SELECT * FROM bier_categories;"
SELECT_ALL_ROWS_BIER_CATEGORY_WEIGHT_TABLE = "SELECT * FROM bier_category_weight;"
SELECT_ALL_ROWS_BIER_BACKTEST_TABLE = "SELECT * FROM bier_backtest ORDER BY id DESC;"

CREATE_BIER_BACKTEST_TABLE = """CREATE TABLE IF NOT EXISTS bier_backtest (
    id SERIAL PRIMARY KEY,
    test_run TEXT,
    date TEXT,
    name TEXT,
    start_date TEXT,
    end_date TEXT,
    signal_strategy TEXT,
    return REAL,
    max_dd REAL,
    sharpe REAL,
    sortino REAL,
    calmar REAL
);"""

ADD_ROW_ITC_TABLE = """INSERT INTO itc (date, mcap, mcap_risk, close, risk_level, alt_mcap, alt_mcap_risk, dxy_usd, dxy_usd_risk, eth_usd, eth_usd_risk, bnb_usd, bnb_usd_risk, sol_usd, sol_usd_risk, ltc_usd, ltc_usd_risk) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_CAPRIOLE_TABLE = """INSERT INTO capriole2 (date, macro_index, heater, oi_pct_mcap, nvts, production_cost, miner_margin, supply_delta, cdd_trend, apparent_demand, 
                                energy_value, active_more_2y_percent, active_more_1y_percent, active_more_6m_percent, lth_supply_percent, yardstick, institutional_buying_excess, mnav_enterprise_mean, treasury_cvd_Z, buy_sell_ratio_roc,
                                companies_buying_per_day, ave_debt_to_equity, ave_debt_to_enterprise_value, ave_debt_to_bitcoin_treasury, yield_mean, days_cover_enterprise_mean, sentiment_spread, active_manager_sentiment, business_outlook, equity_fear_greed, 
                                equity_premium, m2_yoy_less_rates, gsr, insider_trades, market_breadth, sp_pcr, us_liquidity_yoy, usd_positioning, bbb_credit_spread, xly_xlp, 
                                como_spx, junk_treasuries, weight_node_strategy, weight_midas_strategy, weight_spectrum_strategy, macro_index_eth, apparent_demand_eth, heater_eth, oi_pct_mcap_eth, active_more_2y_pct_eth, 
                                active_more_1y_pct_eth, active_more_6m_pct_eth, cdd_trend_eth, speculation_index_eth, crypto_breadth) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s);"""
ADD_ROW_BMP_TABLE = """INSERT INTO bmp (date, close, nupl, sopr, mvrv, liquidity, reserve_risk, bitcoin_sentiment,
                            addresses_in_profit, rhodl_ratio, miner_fee_pct, realized_price_sth, sth_supply, vdd_multiple, fear_greed, funding_rate) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_BMP2_TABLE = """INSERT INTO bmp2 (date, close, nupl, sopr, mvrv, liquidity, reserve_risk, bitcoin_sentiment, addresses_in_profit, 
                     rhodl_ratio, miner_fee_pct, realized_price_sth, sth_supply, vdd_multiple, fear_greed, funding_rate, nvt, sth_mvrv, 
                     lth_mvrv, financial_stress, high_yield_credit, m2_yoy_change, yield_spread, btc_etf_flows, cycle_capital_flows, bitcoin_cycle_master, 
                     onchain_prediction, realized_price_lth , pi_cycle_oscillator,  everything_indicator) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s);""" 

ADD_ROW_MANTA_TABLE = """INSERT INTO manta_data (date, price_usd_close, itc_risk_level_norm, mi_correlation_norm, mi_low_beta_norm, dxy, dxy_norm, funding_neg, funding_pos)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_AUGMENTO_TABLE = """INSERT INTO augmento (date, bitcointalk, reddit, twitter, augmento) 
                            VALUES (%s, %s, %s, %s, %s);"""
ADD_ROW_HYBLOCK_TABLE = """INSERT INTO hyblock (date, volume_delta, whale_retail, user_bot_ratio, usdt_premium, bid_ask_ratio, top_traders_long, market_order_size, market_order_count, limit_order_count, 
                            funding_rate, fear_greed, bid_ask_delta, long_liquidations, short_liquidations , oi_delta, bvol, dvol) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_ETF_TABLE = """INSERT INTO etf_flow (date, flow, close) VALUES (%s, %s, %s);"""
ADD_ROW_BIER_TABLE = """INSERT INTO bier (date, close, bier_invested, bier_range) VALUES (%s, %s, %s, %s);"""
ADD_ROW_BIER2_TABLE = """INSERT INTO bier2 (date, close, invest_score, peaks, valleys, extremes, invested) VALUES (%s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_MACRO_TABLE = """INSERT INTO macro_norm (date, open, high, low, close, volume, pmi_norm, dxy_norm, liq_norm, dfg_norm, usdt_dom_norm) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_BIER_CATEGORIES_TABLE = """INSERT INTO bier_categories (metric, capriole, bmp, manta, itc, tv, strategy, market, mining, macro, shortterm, sentiment, hodl, treasury, supply_demand, eth, alts, custom, bier, test)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
ADD_ROW_BIER_CATEGORY_WEIGHT_TABLE = """INSERT INTO bier_category_weight (strategy, market, mining, macro, shortterm, sentiment, hodl, treasury, supply_demand, eth)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""

DELETE_LAST_ROW_ITC_TABLE = "DELETE FROM itc WHERE id = (SELECT id FROM itc ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_CAPRIOLE_TABLE = "DELETE FROM capriole2 WHERE id = (SELECT id FROM capriole2 ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_BMP_TABLE = "DELETE FROM bmp WHERE id = (SELECT id FROM bmp ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_BMP2_TABLE = "DELETE FROM bmp2 WHERE id = (SELECT id FROM bmp2 ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_MANTA_TABLE = "DELETE FROM manta_data WHERE id = (SELECT id FROM manta_data ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_AUGMENTO_TABLE = "DELETE FROM augmento WHERE id = (SELECT id FROM augmento ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_HYBLOCK_TABLE = "DELETE FROM hyblock WHERE id = (SELECT id FROM hyblock ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_ETF_TABLE = "DELETE FROM etf_flow WHERE id = (SELECT id FROM etf_flow ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_BIER_TABLE = "DELETE FROM bier WHERE id = (SELECT id FROM bier ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_BIER2_TABLE = "DELETE FROM bier2 WHERE id = (SELECT id FROM bier2 ORDER BY id DESC LIMIT 1);"
DELETE_LAST_ROW_MACRO_TABLE = "DELETE FROM macro_norm WHERE id = (SELECT id macro_norm bier ORDER BY id DESC LIMIT 1);"

DELETE_LAST_X_ROWS_ITC_TABLE = "DELETE FROM itc WHERE id IN (SELECT id FROM itc ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_CAPRIOLE_TABLE = "DELETE FROM capriole2 WHERE id IN (SELECT id FROM capriole2 ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_BMP_TABLE = "DELETE FROM bmp WHERE id IN (SELECT id FROM bmp ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_BMP2_TABLE = "DELETE FROM bmp2 WHERE id IN (SELECT id FROM bmp2 ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_MANTA_TABLE = "DELETE FROM manta_data WHERE id IN (SELECT id FROM manta_data ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_AUGMENTO_TABLE = "DELETE FROM augmento WHERE id IN (SELECT id FROM augmento ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_HYBLOCK_TABLE = "DELETE FROM hyblock WHERE id IN (SELECT id FROM hyblock ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_ETF_TABLE = "DELETE FROM etf_flow WHERE id IN (SELECT id FROM etf_flow ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_BIER_TABLE = "DELETE FROM bier WHERE id IN (SELECT id FROM bier ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_BIER2_TABLE = "DELETE FROM bier2 WHERE id IN (SELECT id FROM bier2 ORDER BY id DESC LIMIT "
DELETE_LAST_X_ROWS_MACRO_TABLE = "DELETE FROM macro_norm WHERE id IN (SELECT id FROM macro_norm ORDER BY id DESC LIMIT "

DELETE_ALL_ROWS_BACKTEST_TABLE = "DELETE FROM bier_backtest;"

dict_keys_itc = ['id','date', 'mcap', 'mcap_risk', 'close', 'risk_level', 'alt_mcap', 'alt_mcap_risk', 'dxy_usd', 'dxy_usd_risk', 'eth_usd', 'eth_usd_risk', 'bnb_usd', 'bnb_usd_risk', 'sol_usd', 'sol_usd_risk', 'ltc_usd', 'ltc_usd_risk']

dict_keys_capriole = ['id','date', 'macro_index', 'heater', 'oi_pct_mcap', 'nvts', 'production_cost', 'miner_margin', 'supply_delta', 'cdd_trend', 'apparent_demand', 
                                'energy_value', 'active_more_2y_percent', 'active_more_1y_percent', 'active_more_6m_percent', 'lth_supply_percent', 'yardstick', 'institutional_buying_excess', 'mnav_enterprise_mean', 'treasury_cvd_Z', 'buy_sell_ratio_roc',
                                'companies_buying_per_day', 'ave_debt_to_equity', 'ave_debt_to_enterprise_value', 'ave_debt_to_bitcoin_treasury', 'yield_mean', 'days_cover_enterprise_mean', 'sentiment_spread', 'active_manager_sentiment', 'business_outlook', 'equity_fear_greed', 
                                'equity_premium', 'm2_yoy_less_rates', 'gsr', 'insider_trades', 'market_breadth', 'sp_pcr', 'us_liquidity_yoy', 'usd_positioning', 'bbb_credit_spread', 'xly_xlp', 
                                'como_spx', 'junk_treasuries', 'weight_node_strategy', 'weight_midas_strategy', 'weight_spectrum_strategy', 'macro_index_eth', 'apparent_demand_eth', 'heater_eth', 'oi_pct_mcap_eth', 'active_more_2y_pct_eth', 
                                'active_more_1y_pct_eth', 'active_more_6m_pct_eth', 'cdd_trend_eth', 'speculation_index_eth', 'crypto_breadth']

dict_keys_bmp = ['id','date', 'close', 'nupl', 'sopr', 'mvrv', 'lth_mvrv', 'liquidity', 'reserve_risk', 'bitcoin_sentiment', 
                 'addresses_in_profit', 'rhodl_ratio', 'miner_fee_pct', 'realized_price_sth','sth_supply', 'miner_price', 'vdd_multiple', 'fear_greed', 'funding_rate']
dict_keys_bmp2 = ['id','date', 'close', 'nupl', 'sopr', 'mvrv', 'liquidity', 'reserve_risk', 'bitcoin_sentiment', 'addresses_in_profit', #10
                  'rhodl_ratio', 'miner_fee_pct', 'realized_price_sth','sth_supply',  'vdd_multiple', 'fear_greed', 'funding_rate', 'nvt', 'sth_mvrv',#9
                  'lth_mvrv','financial_stress', 'high_yield_credit', 'm2_yoy_change', 'yield_spread', 'btc_etf_flows', 'cycle_capital_flows', 'bitcoin_cycle_master',#8
                  'onchain_prediction', 'realized_price_lth', 'pi_cycle_oscillator',  'everything_indicator'] #4

dict_keys_manta = ['id','date', 'price_usd_close', 'itc_risk_level_norm', 'mi_correlation_norm', 'mi_low_beta_norm', 'return_180d', 'return_180d_norm', 'dxy', 'dxy_norm', 
                   'funding_neg', 'funding_pos', 'capriole_macro', 'alts_speculation', 'augmento_reddit', 'augmento_twitter', 'augmento_btctalk']
dict_keys_augmento = ['id','date', 'bitcointalk', 'reddit', 'twitter', 'augmento']
dict_keys_hyblock_1h = ['id','time', 'volume_delta', 'whale_retail', 'user_bot_ratio', 'usdt_premium', 'bid_ask_ratio', 'top_traders_long','market_order_size','market_order_count','limit_order_count','funding_rate','fear_greed','bid_ask_delta','long_liquidations','short_liquidations','oi_delta','bvol','dvol']
dict_keys_hyblock = ['id','date', 'volume_delta', 'whale_retail', 'user_bot_ratio', 'usdt_premium', 'bid_ask_ratio', 'top_traders_long','market_order_size','market_order_count','limit_order_count','funding_rate','fear_greed','bid_ask_delta','long_liquidations','short_liquidations','oi_delta','bvol','dvol']
dict_keys_etf = ['id','date', 'flow']
dict_keys_bier = ['id','date', 'close', 'bier_invested', 'bier_range']
dict_keys_bier2 = ['id','date', 'close', 'invest_score', 'peaks', 'valleys', 'extremes', 'invested'] 
dict_keys_macro = ['id','date', 'open', 'high', 'low', 'close', 'volume', 'pmi_norm', 'dxy_norm', 'liq_norm', 'dfg_norm', 'usdt_dom_norm']
dict_keys_bier_categories = ['id', 'metric', 'capriole', 'bmp', 'manta', 'itc', 'tv', 'strategy', 'market', 'mining', 'macro', 'shortterm', 'sentiment', 'hodl', 'treasury', 'supply_demand', 'eth', 'alts', 'custom', 'bier', 'test']
dict_keys_bier_category_weight = ['id', 'strategy', 'market', 'mining', 'macro', 'shortterm', 'sentiment', 'hodl', 'treasury', 'supply_demand', 'eth']


connection = psycopg2.connect(DATABASE_URL)
connection_all = connection
#connection_hyblock_1h = psycopg2.connect(DATABASE_URL_HYBLOCK_1H)

# ----------------------------- Create & Load tables ----------------------------------------------------

def db_test():
    print('DB Test')


def create_tables():
    print('Bier2 table created')
    with connection:
        with connection.cursor() as cursor:      
            cursor.execute(CREATE_BIER2_TABLE)
            cursor.execute(CREATE_BIER_CATEGORIES_TABLE)
            cursor.execute(CREATE_BIER_CATEGORY_WEIGHT_TABLE)


# ----------------------------- Read tables -------------------------------------------------------------
# ----------------------- read ohlc price data in specified timeframe



def read_table_date_range_cloud(table_name, start_date, end_date): # combined query for all tables
  # possible selections: oi, funding, ls, liqs, vol, lending, price_spot, price_perp 
    if table_name == 'hyblock_1h':
       print('No Hyblock DB')
       return pd.DataFrame()
       #connection = connection_hyblock_1h
    else:
       connection = connection_all
    with connection:
        with connection.cursor() as cursor:                
          if table_name.lower() == 'itc':            
            db_command = SELECT_DATE_RANGE_ITC_TABLE
            dict_keys_selection = dict_keys_itc  
          elif table_name.lower() == 'capriole2':            
            db_command = SELECT_DATE_RANGE_CAPRIOLE_TABLE
            dict_keys_selection = dict_keys_capriole   
          elif table_name.lower() == 'bmp':            
            db_command = SELECT_DATE_RANGE_BMP_TABLE
            dict_keys_selection = dict_keys_bmp
          elif table_name.lower() == 'bmp2':            
            db_command = SELECT_DATE_RANGE_BMP2_TABLE
            dict_keys_selection = dict_keys_bmp2            
          elif table_name.lower() == 'manta':            
            db_command = SELECT_DATE_RANGE_MANTA_TABLE
            dict_keys_selection = dict_keys_manta         
          elif table_name.lower() == 'augmento':            
            db_command = SELECT_DATE_RANGE_AUGMENTO_TABLE
            dict_keys_selection = dict_keys_augmento
          elif table_name.lower() == 'hyblock_1h':            
            db_command = SELECT_DATE_RANGE_HYBLOCK_TABLE_1H
            dict_keys_selection = dict_keys_hyblock_1h    
          elif table_name.lower() == 'hyblock':            
            db_command = SELECT_DATE_RANGE_HYBLOCK_TABLE
            dict_keys_selection = dict_keys_hyblock      
          elif table_name.lower() == 'etf':            
            db_command = SELECT_DATE_RANGE_ETF_TABLE
            dict_keys_selection = dict_keys_etf       
          elif table_name.lower() == 'bier':            
            db_command = SELECT_DATE_RANGE_BIER_TABLE
            dict_keys_selection = dict_keys_bier        
          elif table_name.lower() == 'bier2':            
            db_command = SELECT_DATE_RANGE_BIER2_TABLE
            dict_keys_selection = dict_keys_bier2              
          elif table_name.lower() == 'macro':            
            db_command = SELECT_DATE_RANGE_MACRO_TABLE
            dict_keys_selection = dict_keys_macro                                                              
          else:
            return []
          #print(db_command)
          cursor.execute(db_command, (start_date, end_date))
          tuples = cursor.fetchall()
          list_dicts = []
          for i in range(len(tuples)):
            list_dicts.append(dict(zip(dict_keys_selection, tuples[i])))
          #print(list_dicts)
          df = pd.DataFrame(list_dicts)
          # Correct wrong date format in bmp table
          if (table_name.lower() == 'bmp') or (table_name.lower() == 'bmp_full'):  
            df['date'] = df['date'].str.slice(0, 10)
          return df
                   
  

def read_table_last_X_cloud(table_name, number): # combined query for all tables
  # possible selections: oi, funding, ls, liqs, vol, lending, price_spot, price_perp 
    #print('Table: ', table_name)
    if table_name == 'hyblock_1h':
       print('No Hyblock DB')
       return pd.DataFrame()
       #connection = connection_hyblock_1h  
    else:
       connection = connection_all       
    with connection:
        with connection.cursor() as cursor:
          if table_name.lower() == 'itc':    
            dict_keys_selection = dict_keys_itc   
            db_command = SELECT_LAST_X_ROWS_ITC_TABLE + str(number) + ";"  
          elif table_name.lower() == 'capriole2':            
            dict_keys_selection = dict_keys_capriole   
            db_command = SELECT_LAST_X_ROWS_CAPRIOLE_TABLE + str(number) + ";"     
          elif table_name.lower() == 'bmp':            
            dict_keys_selection = dict_keys_bmp   
            db_command = SELECT_LAST_X_ROWS_BMP_TABLE + str(number) + ";"   
          elif table_name.lower() == 'bmp2':            
            dict_keys_selection = dict_keys_bmp2   
            db_command = SELECT_LAST_X_ROWS_BMP2_TABLE + str(number) + ";"              
          elif table_name.lower() == 'manta':            
            dict_keys_selection = dict_keys_manta   
            db_command = SELECT_Last_X_ROWS_MANTA_TABLE + str(number) + ";"    
          elif table_name.lower() == 'augmento':            
            dict_keys_selection = dict_keys_augmento   
            db_command = SELECT_LAST_X_ROWS_AUGMENTO_TABLE + str(number) + ";"                 
          elif table_name.lower() == 'hyblock_1h':            
            dict_keys_selection = dict_keys_hyblock_1h   
            db_command = SELECT_Last_X_ROWS_HYBLOCK_TABLE_1H + str(number) + ";"           
          elif table_name.lower() == 'hyblock':            
            dict_keys_selection = dict_keys_hyblock  
            db_command = SELECT_Last_X_ROWS_HYBLOCK_TABLE + str(number) + ";"     
          elif table_name.lower() == 'etf':            
            dict_keys_selection = dict_keys_etf  
            db_command = SELECT_LAST_X_ROWS_ETF_TABLE + str(number) + ";"    
          elif table_name.lower() == 'bier':            
            dict_keys_selection = dict_keys_bier  
            db_command = SELECT_LAST_X_ROWS_BIER_TABLE + str(number) + ";" 
          elif table_name.lower() == 'bier2':            
            dict_keys_selection = dict_keys_bier2  
            db_command = SELECT_LAST_X_ROWS_BIER2_TABLE + str(number) + ";"             
          elif table_name.lower() == 'macro':            
            dict_keys_selection = dict_keys_macro 
            db_command = SELECT_LAST_X_ROWS_MACRO_TABLE + str(number) + ";"                                                            
          else:
            return []
          print(db_command)
          cursor.execute(db_command)
          tuples = cursor.fetchall()
          list_dicts = []
          for i in range(len(tuples)):
            list_dicts.append(dict(zip(dict_keys_selection, tuples[i])))
          print(list_dicts)
          return list_dicts            


# ----------------------------- Update tables -------------------------------------------------------------
# add row to table

def add_row_itc(table_name, date, mcap, mcap_risk, close, risk_level, alt_mcap, alt_mcap_risk, dxy_usd, dxy_usd_risk, eth_usd, eth_usd_risk, bnb_usd, bnb_usd_risk, sol_usd, sol_usd_risk, ltc_usd, ltc_usd_risk):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'itc':
              db_command = ADD_ROW_ITC_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, mcap, mcap_risk, close, risk_level, alt_mcap, alt_mcap_risk, dxy_usd, dxy_usd_risk, eth_usd, eth_usd_risk, bnb_usd, bnb_usd_risk, sol_usd, sol_usd_risk, ltc_usd, ltc_usd_risk))
    return


def add_df_itc(df, table_name):
    for d in df.to_dict(orient="records"):
      #add_row_itc(table_name, d['date'], d['market_cap'], d['close'], d['risk_level'], d['btc_dominance'], d['btc_dominance_no_stables'])
      add_row_itc(table_name, d['date'], d['mcap'], d['mcap_risk'], d['close'], d['risk_level'], d['alt_mcap'], d['alt_mcap_risk'], d['dxy_usd'], d['dxy_usd_risk'],  
                  d['eth_usd'], d['eth_usd_risk'], d['bnb_usd'], d['bnb_usd_risk'], d['sol_usd'], d['sol_usd_risk'], d['ltc_usd'], d['ltc_usd_risk'])
    return



def add_df_capriole(df, table_name):
    for d in df.to_dict(orient="records"):
      """add_row_capriole(table_name, d['date'], d['macro_index'], d['heater'], d['oi_pct_mcap'], d['nvts'], d['production_cost'], d['miner_margin'], d['supply_delta'], d['cdd_trend'], d['apparent_demand'], 
                      d['energy_value'], d['active_more_2y_percent'], d['active_more_1y_percent'], d['active_more_6m_percent'], d['lth_supply_percent'], d['yardstick'], d['institutional_buying_excess'], d['mnav_enterprise_mean'], d['treasury_cvd_Z'], d['buy_sell_ratio_roc'],
                      d['companies_buying_per_day'], d['ave_debt_to_equity'], d['ave_debt_to_enterprise_value'], d['ave_debt_to_bitcoin_treasury'], d['yield_mean'], d['days_cover_enterprise_mean'], d['sentiment_spread'], d['active_manager_sentiment'], d['business_outlook'], d['equity_fear_greed'],
                      d['equity_premium'], d['m2_yoy_less_rates'], d['gsr'], d['insider_trades'], d['market_breadth'], d['sp_pcr'], d['us_liquidity_yoy'], d['usd_positioning'], d['bbb_credit_spread'], d['xly_xlp'],  
                      d['como_spx'], d['junk_treasuries'], d['weight_node_strategy'], d['weight_midas_strategy'], d['weight_spectrum_strategy'], d['macro_index_eth'], d['apparent_demand_eth'], d['heater_eth'], d['oi_pct_mcap_eth'], d['active_more_2y_pct_eth'], 
                      d['active_more_1y_pct_eth'], d['active_more_6m_pct_eth'], d['cdd_trend_eth'], d['speculation_index_eth'], d['crypto_breadth']) """
      # Remove treasuries, liquidity
      add_row_capriole(table_name, d['date'], d['macro_index'], d['heater'], d['oi_pct_mcap'], d['nvts'], d['production_cost'], d['miner_margin'], d['supply_delta'], d['cdd_trend'], d['apparent_demand'], 
                      d['energy_value'], d['active_more_2y_percent'], d['active_more_1y_percent'], d['active_more_6m_percent'], d['lth_supply_percent'], d['yardstick'], 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d['sentiment_spread'], d['active_manager_sentiment'], d['business_outlook'], d['equity_fear_greed'],
                      d['equity_premium'], 0.0, d['gsr'], d['insider_trades'], d['market_breadth'], d['sp_pcr'], d['us_liquidity_yoy'], d['usd_positioning'], d['bbb_credit_spread'], d['xly_xlp'],  
                      d['como_spx'], d['junk_treasuries'], d['weight_node_strategy'], d['weight_midas_strategy'], d['weight_spectrum_strategy'], d['macro_index_eth'], d['apparent_demand_eth'], d['heater_eth'], d['oi_pct_mcap_eth'], d['active_more_2y_pct_eth'], 
                      d['active_more_1y_pct_eth'], d['active_more_6m_pct_eth'], d['cdd_trend_eth'], d['speculation_index_eth'], d['crypto_breadth'])      
    return



# add row to table

def add_row_capriole(table_name, date, macro_index, heater, oi_pct_mcap, nvts, production_cost, miner_margin, supply_delta, cdd_trend, apparent_demand, 
                                energy_value, active_more_2y_percent, active_more_1y_percent, active_more_6m_percent, lth_supply_percent, yardstick, institutional_buying_excess, mnav_enterprise_mean, treasury_cvd_Z, buy_sell_ratio_roc,
                                companies_buying_per_day, ave_debt_to_equity, ave_debt_to_enterprise_value, ave_debt_to_bitcoin_treasury, yield_mean, days_cover_enterprise_mean, sentiment_spread, active_manager_sentiment, business_outlook, equity_fear_greed, 
                                equity_premium, m2_yoy_less_rates, gsr, insider_trades, market_breadth, sp_pcr, us_liquidity_yoy, usd_positioning, bbb_credit_spread, xly_xlp, 
                                como_spx, junk_treasuries, weight_node_strategy, weight_midas_strategy, weight_spectrum_strategy, macro_index_eth, apparent_demand_eth, heater_eth, oi_pct_mcap_eth, active_more_2y_pct_eth, 
                                active_more_1y_pct_eth, active_more_6m_pct_eth, cdd_trend_eth, speculation_index_eth, crypto_breadth):
                    
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'capriole2':
              db_command = ADD_ROW_CAPRIOLE_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, macro_index, heater, oi_pct_mcap, nvts, production_cost, miner_margin, supply_delta, cdd_trend, apparent_demand, 
                                energy_value, active_more_2y_percent, active_more_1y_percent, active_more_6m_percent, lth_supply_percent, yardstick, institutional_buying_excess, mnav_enterprise_mean, treasury_cvd_Z, buy_sell_ratio_roc,
                                companies_buying_per_day, ave_debt_to_equity, ave_debt_to_enterprise_value, ave_debt_to_bitcoin_treasury, yield_mean, days_cover_enterprise_mean, sentiment_spread, active_manager_sentiment, business_outlook, equity_fear_greed, 
                                equity_premium, m2_yoy_less_rates, gsr, insider_trades, market_breadth, sp_pcr, us_liquidity_yoy, usd_positioning, bbb_credit_spread, xly_xlp, 
                                como_spx, junk_treasuries, weight_node_strategy, weight_midas_strategy, weight_spectrum_strategy, macro_index_eth, apparent_demand_eth, heater_eth, oi_pct_mcap_eth, active_more_2y_pct_eth, 
                                active_more_1y_pct_eth, active_more_6m_pct_eth, cdd_trend_eth, speculation_index_eth, crypto_breadth))
               
    return

# add row to table

def add_row_bmp(table_name, date, close, nupl, sopr, mvrv, liquidity, reserve_risk, bitcoin_sentiment, addresses_in_profit, rhodl_ratio, miner_fee_pct, realized_price_sth, sth_supply, vdd_multiple, fear_greed, funding_rate):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'bmp':
              db_command = ADD_ROW_BMP_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, close, nupl, sopr, mvrv, liquidity, reserve_risk, bitcoin_sentiment, addresses_in_profit, rhodl_ratio, miner_fee_pct, realized_price_sth, sth_supply, vdd_multiple, fear_greed, funding_rate))
    return


def add_df_bmp(df, table_name):
    print('ADD DF BMP')
    for d in df.to_dict(orient="records"):
      add_row_bmp(table_name, d['date'], d['close'], d['nupl'], d['sopr'], d['mvrv'], d['liquidity'], d['reserve_risk'], d['bitcoin_sentiment'], d['addresses_in_profit'], d['rhodl_ratio'], d['miner_fee_pct'], d['realized_price_sth'], d['sth_supply'], d['vdd_multiple'], d['fear_greed'], d['fear_greed'])
      #add_row_bmp(table_name, d['date'], d['close'], d['nupl'], d['sopr'], d['mvrv'], d['liquidity'], d['reserve_risk'], d['bitcoin_sentiment'], d['addresses_in_profit'], d['rhodl_ratio'], d['miner_fee_pct'], d['realized_price_sth'], d['sth_supply'], d['vdd_multiple'], d['fear_greed'], d['funding_rate'])
    return


# add row to table

def add_row_bmp2(table_name, date, close, nupl, sopr, mvrv, liquidity, reserve_risk, bitcoin_sentiment, addresses_in_profit, 
                     rhodl_ratio, miner_fee_pct, realized_price_sth, sth_supply, vdd_multiple, fear_greed, funding_rate, nvt, sth_mvrv, 
                     lth_mvrv, financial_stress, high_yield_credit, m2_yoy_change, yield_spread, btc_etf_flows, cycle_capital_flows, bitcoin_cycle_master,
                     onchain_prediction, realized_price_lth, pi_cycle_oscillator,  everything_indicator):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'bmp2':
              db_command = ADD_ROW_BMP2_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, close, nupl, sopr, mvrv, liquidity, reserve_risk, bitcoin_sentiment, addresses_in_profit, 
                     rhodl_ratio, miner_fee_pct, realized_price_sth, sth_supply, vdd_multiple, fear_greed, funding_rate, nvt, sth_mvrv, 
                     lth_mvrv, financial_stress, high_yield_credit, m2_yoy_change, yield_spread, btc_etf_flows, cycle_capital_flows, bitcoin_cycle_master,
                     onchain_prediction, realized_price_lth, pi_cycle_oscillator,  everything_indicator))
    return


def add_df_bmp2(df, table_name):
    print('ADD DF BMP2')

    # Define expected columns
    expected_columns = [
        'date', 'close', 'nupl', 'sopr', 'mvrv', 'liquidity', 'reserve_risk', 
        'bitcoin_sentiment', 'addresses_in_profit', 'rhodl_ratio', 'miner_fee_pct', 
        'realized_price_sth', 'sth_supply', 'vdd_multiple', 'fear_greed', 
        'funding_rate', 'nvt', 'sth_mvrv', 'lth_mvrv', 'financial_stress', 
        'high_yield_credit', 'm2_yoy_change', 'yield_spread', 'btc_etf_flows', 
        'cycle_capital_flows', 'bitcoin_cycle_master', 'onchain_prediction', 
        'realized_price_lth', 'pi_cycle_oscillator', 'everything_indicator']
    
    # Check for missing columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing Columns: {missing_columns}")
        return

    for d in df.to_dict(orient="records"):
      add_row_bmp2(table_name, d['date'], d['close'], d['nupl'], d['sopr'], d['mvrv'], d['liquidity'], d['reserve_risk'], d['bitcoin_sentiment'], d['addresses_in_profit'], 
                       d['rhodl_ratio'], d['miner_fee_pct'], d['realized_price_sth'], d['sth_supply'], d['vdd_multiple'], d['fear_greed'], d['funding_rate'], d['nvt'], d['sth_mvrv'],
                       d['lth_mvrv'], d['financial_stress'], d['high_yield_credit'], d['m2_yoy_change'], d['yield_spread'], d['btc_etf_flows'], d['cycle_capital_flows'], d['bitcoin_cycle_master'], 
                       d['onchain_prediction'], d['realized_price_lth'], d['pi_cycle_oscillator'],  d['everything_indicator'])
    return



   
def add_row_manta(table_name, date, price_usd_close, itc_risk_level_norm, mi_correlation_norm, mi_low_beta_norm, dxy, dxy_norm, funding_neg, funding_pos):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
              if table_name.lower() == 'manta':            
                db_command = ADD_ROW_MANTA_TABLE
              else:
                return []            
              #print(db_command)                
              cursor.execute(db_command, (date, price_usd_close, itc_risk_level_norm, mi_correlation_norm, mi_low_beta_norm, dxy, dxy_norm, funding_neg, funding_pos))
    return
 

def add_df_manta(df, table_name):
    for d in df.to_dict(orient="records"):
      add_row_manta(table_name, d['date'], d['price_usd_close'], d['itc_risk_level_norm'], d['mi_correlation_norm'], d['mi_low_beta_norm'], d['dxy'], d['dxy_norm'], d['funding_neg'], d['funding_pos'], )       
    return


def add_row_augmento(table_name, date, bitcointalk, reddit, twitter, augmento):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'augmento':
              db_command = ADD_ROW_AUGMENTO_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, bitcointalk, reddit, twitter, augmento))
    return


def add_df_augmento(df, table_name):
    for d in df.to_dict(orient="records"):
      add_row_augmento(table_name, d['date'], d['bitcointalk'], d['reddit'], d['twitter'], d['augmento'])
    return



def add_row_hyblock(table_name, date, volume_delta, whale_retail, user_bot_ratio, usdt_premium, bid_ask_ratio, top_traders_long, market_order_size, market_order_count, limit_order_count, funding_rate, fear_greed, bid_ask_delta, long_liquidations, short_liquidations , oi_delta, bvol, dvol):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'hyblock':
              db_command = ADD_ROW_HYBLOCK_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, volume_delta, whale_retail, user_bot_ratio, usdt_premium, bid_ask_ratio, top_traders_long, market_order_size, market_order_count, limit_order_count, funding_rate, fear_greed, bid_ask_delta, long_liquidations, short_liquidations , oi_delta, bvol, dvol))
    return


def add_df_hyblock(df, table_name):
    for d in df.to_dict(orient="records"):
      add_row_hyblock(table_name, d['date'], d['volume_delta'], d['whale_retail'], d['user_bot_ratio'], d['usdt_premium'], d['bid_ask_ratio'], d['top_traders_long'], d['market_order_size'], d['market_order_count'], d['limit_order_count'], d['funding_rate'], d['fear_greed'], d['bid_ask_delta'], d['long_liquidations'], d['short_liquidations'], d['oi_delta'], d['bvol'], d['dvol'])
    return


def add_row_etf(table_name, date, flow, close):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'etf':
              db_command = ADD_ROW_ETF_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, flow, close))
    return


def add_df_etf(df, table_name):
    for d in df.to_dict(orient="records"):
      add_row_etf(table_name, d['date'], d['flow'], d['close'])
    return


def add_row_bier(table_name, date, close, bier_invested, bier_range):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'bier':
              db_command = ADD_ROW_BIER_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, close, bier_invested, bier_range))
    return


def add_df_bier(df, table_name):
    for d in df.to_dict(orient="records"):
      add_row_bier(table_name, d['date'], d['close'], d['bier_invested'], d['bier_range'])
    return



def add_row_bier2(table_name, date, close, invest_score, peaks, valleys, extremes, invested):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'bier2':
              db_command = ADD_ROW_BIER2_TABLE       
            else:
              return []            
            print(db_command)
            print(date, close, invest_score, peaks, valleys, extremes, invested)
            cursor.execute(db_command, (date, close, invest_score, peaks, valleys, extremes, invested))
    return


def add_df_bier2(df, table_name):
    df = df.replace({np.nan: None})
    #df['date'] = df['date'].strftime("%Y-%m-%d")
    for d in df.to_dict(orient="records"):
      add_row_bier2(table_name, str(d['date'])[:10], d['close'], d['invest_score'], d['peaks'], d['valleys'], d['extremes'], d['invested'])
    return


def add_row_macro(table_name, date, open, high, low, close, volume, pmi_norm, dxy_norm, liq_norm, dfg_norm, usdt_dom_norm):
    with connection:
        with connection.cursor() as cursor:
            #print('Add row: ',table_name, time)
            if table_name.lower() == 'macro':
              db_command = ADD_ROW_MACRO_TABLE       
            else:
              return []            
            #print(db_command)
            cursor.execute(db_command, (date, open, high, low, close, volume, pmi_norm, dxy_norm, liq_norm, dfg_norm, usdt_dom_norm))
    return


def add_df_macro(df, table_name):
    for d in df.to_dict(orient="records"):
      add_row_macro(table_name, d['date'], d['open'], d['high'], d['low'], d['close'], d['volume'], d['pmi_norm'], d['dxy_norm'], d['liq_norm'], d['dfg_norm'], d['usdt_dom_norm'])
    return


def add_row_bier_categories(metric, capriole, bmp, manta, itc, tv, strategy, market, mining, macro, shortterm, sentiment, hodl, treasury, supply_cap_eth, alts, custom, bier, test):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(ADD_ROW_BIER_CATEGORIES_TABLE, (metric, capriole, bmp, manta, itc, tv, strategy, market, mining, macro, shortterm, sentiment, hodl, treasury, supply_cap_eth, alts, custom, bier, test))
    return


def add_df_bier_categories(df):
    for d in df.to_dict(orient="records"):
        add_row_bier_categories(d['metric'], d['capriole'], d['bmp'], d['manta'], d['itc'], d['tv'], d['strategy'], d['market'], d['mining'], d['macro'], d['shortterm'], d['sentiment'], d['hodl'], d['treasury'], d['supply_cap_eth'], d['alts'], d['custom'], d['bier'], d['test'])
    return


def create_bier_categories_table():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_BIER_CATEGORIES_TABLE)
    return


def add_row_bier_category_weight(strategy, market, mining, macro, shortterm, sentiment, hodl, treasury, supply_demand, eth):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(ADD_ROW_BIER_CATEGORY_WEIGHT_TABLE, (strategy, market, mining, macro, shortterm, sentiment, hodl, treasury, supply_demand, eth))
    return


def add_df_bier_category_weight(df):
    for d in df.to_dict(orient="records"):
        add_row_bier_category_weight(d['strategy'], d['market'], d['mining'], d['macro'], d['shortterm'], d['sentiment'], d['hodl'], d['treasury'], d['supply_demand'], d['eth'])
    return


def create_bier_category_weight_table():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_BIER_CATEGORY_WEIGHT_TABLE)
    return


def read_bier_categories():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_ALL_ROWS_BIER_CATEGORIES_TABLE)
            tuples = cursor.fetchall()
            list_dicts = []
            for t in tuples:
                list_dicts.append(dict(zip(dict_keys_bier_categories, t)))
            return list_dicts


def read_bier_category_weight():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_ALL_ROWS_BIER_CATEGORY_WEIGHT_TABLE)
            tuples = cursor.fetchall()
            list_dicts = []
            for t in tuples:
                list_dicts.append(dict(zip(dict_keys_bier_category_weight, t)))
            return list_dicts


# ----------------------------- Backtest Table Functions --------------------------------------------------

# ----------------------------- Backtest Table Functions --------------------------------------------------

def create_bier_backtest_table():
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_BIER_BACKTEST_TABLE)
    return

def get_test_metrics():
    """
    Get list of metrics where test = 1 in bier_categories table.
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT metric FROM bier_categories WHERE test = 1;")
            tuples = cursor.fetchall()
            return [t[0] for t in tuples]

def sync_backtest_columns(metrics_list):
    """
    Checks if columns for metrics exist in bier_backtest, adds them if not.
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            # Get existing columns
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'bier_backtest';")
            existing_columns = [t[0] for t in cursor.fetchall()]
            
            for metric in metrics_list:
                # Sanitize metric name to be safe for column name (lowercase)
                col_name = metric.lower()
                if col_name not in existing_columns:
                    print(f"Adding column {col_name} to bier_backtest")
                    cursor.execute(f"ALTER TABLE bier_backtest ADD COLUMN {col_name} REAL DEFAULT 0;")

def save_backtest_row(data_dict):
    """
    Dynamically inserts a row into bier_backtest.
    data_dict keys must match column names.
    """
    if not data_dict:
        return

    columns = list(data_dict.keys())
    values = list(data_dict.values())
    
    placeholders = ",".join(["%s"] * len(values))
    columns_str = ",".join(columns)
    
    query = f"INSERT INTO bier_backtest ({columns_str}) VALUES ({placeholders});"
    
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(values))

def read_backtest_table():
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(SELECT_ALL_ROWS_BIER_BACKTEST_TABLE)
            columns = [desc[0] for desc in cursor.description]
            tuples = cursor.fetchall()
            list_dicts = []
            for t in tuples:
                list_dicts.append(dict(zip(columns, t)))
            return pd.DataFrame(list_dicts)

def delete_bier_backtest_table_content():
    """
    Deletes all rows from the bier_backtest table using a fresh connection with autocommit.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(DELETE_ALL_ROWS_BACKTEST_TABLE)
        conn.close()
        print("Backtest table deleted successfully.")
    except Exception as e:
        print(f"Error deleting backtest table: {e}")
    return


def get_all_metrics():
    """
    Get list of all unique metrics from bier_categories table.
    Returns: list: List of all metric names.
    """
    df = read_categories()
    return sorted(df['metric'].tolist())


def update_custom_metrics(selected_metrics):
    """
    Update the 'custom' column in bier_categories table.
    Set to 1 for selected metrics, 0 for all others.
    
    Args:
        selected_metrics (list): List of metric names to mark as custom
    """
    with connection:
        with connection.cursor() as cursor:
            # First, set all custom values to 0
            cursor.execute("UPDATE bier_categories SET custom = 0;")
            
            # Then set selected metrics to 1
            if selected_metrics:
                placeholders = ','.join(['%s'] * len(selected_metrics))
                query = f"UPDATE bier_categories SET custom = 1 WHERE metric IN ({placeholders});"
                cursor.execute(query, tuple(selected_metrics))
    return




def store_metric_table(df, table_name):
  print('Table Name: ', table_name)

  if table_name.lower() == 'itc':  
    #delete_table_last_row_cloud(table_name)
    pass
  elif table_name.lower() == 'capriole2':    
    delete_table_last_x_rows_cloud(table_name, 22) # delayed metrices e.g. macro_index only updated after 21 days
    pass
  if table_name.lower() == 'bmp':  
    delete_table_last_row_cloud(table_name)
  if table_name.lower() == 'bmp_2':  
    print('Delete last row in BMP2')
    #delete_table_last_row_cloud(table_name)    
  if table_name.lower() == 'manta':  
    delete_table_last_row_cloud(table_name)   
  if table_name.lower() == 'augmento':  
    delete_table_last_row_cloud(table_name)  
  if table_name.lower() == 'hyblock':  
    delete_table_last_row_cloud(table_name)     
  if table_name.lower() == 'etf':  
    delete_table_last_row_cloud(table_name)   
  if table_name.lower() == 'bier':  
    delete_table_last_row_cloud(table_name)      
  if table_name.lower() == 'macro':  
    delete_table_last_row_cloud(table_name)       

  last_row = read_table_last_X_cloud(table_name,1)
  print('Last row: ', last_row)  

  last_date = str(last_row[0]['date']) # check last date in database
  print('Last date: ', last_date)
  #print(df.head(), df.info())
  filt_time = df['date'] > last_date
  df_new = df.loc[filt_time]
  print('DF new: ', df_new.tail().to_string())
  if table_name.lower() == 'itc':
    add_df_itc(df_new, table_name)
  elif table_name.lower() == 'capriole2':
    add_df_capriole(df_new, table_name)
  elif table_name.lower() == 'bmp':
    add_df_bmp(df_new, table_name)
  elif table_name.lower() == 'bmp2':
    print('Add DF BMP2')
    add_df_bmp2(df_new, table_name)    
  elif table_name.lower() == 'manta':
    add_df_manta(df_new, table_name)
  elif table_name.lower() == 'augmento':
    add_df_augmento(df_new, table_name)   
  elif table_name.lower() == 'hyblock':
    add_df_hyblock(df_new, table_name)    
  elif table_name.lower() == 'etf':
    add_df_etf(df_new, table_name)  
  elif table_name.lower() == 'bier':
    add_df_bier(df_new, table_name)     
  elif table_name.lower() == 'bier2':
    add_df_bier2(df_new, table_name)       
  elif table_name.lower() == 'macro':
    add_df_macro(df_new, table_name)         

  return  

# ------------------------ Delete rows --------------------------------------------------------------------
   

def delete_table_last_row_cloud(table_name): # combined query for all tables to delete last X rows of table
    with connection:
        with connection.cursor() as cursor: 
          if table_name.lower() == 'itc':            
            db_command = DELETE_LAST_ROW_ITC_TABLE       
          elif table_name.lower() == 'capriole2':            
            db_command = DELETE_LAST_ROW_CAPRIOLE_TABLE     
          elif table_name.lower() == 'bmp':            
            db_command = DELETE_LAST_ROW_BMP_TABLE  
          elif table_name.lower() == 'bmp2':            
            db_command = DELETE_LAST_ROW_BMP2_TABLE              
          elif table_name.lower() == 'manta':            
            db_command = DELETE_LAST_ROW_MANTA_TABLE       
          elif table_name.lower() == 'augmento':            
            db_command = DELETE_LAST_ROW_AUGMENTO_TABLE 
          elif table_name.lower() == 'hyblock':            
            db_command = DELETE_LAST_ROW_HYBLOCK_TABLE      
          elif table_name.lower() == 'etf':            
            db_command = DELETE_LAST_ROW_ETF_TABLE    
          elif table_name.lower() == 'bier':            
            db_command = DELETE_LAST_ROW_BIER_TABLE       
          elif table_name.lower() == 'macro':            
            db_command = DELETE_LAST_ROW_MACRO_TABLE                                                             
          else:
            return []
          #print(db_command)
          cursor.execute(db_command)
          #time.sleep(1) # wait for x seconds to finish the close market order
          return 
   
#@anvil.server.callable
def delete_table_last_x_rows_cloud(table_name, number):
    with connection:
        with connection.cursor() as cursor:
            if table_name.lower() == 'itc':
                db_command = DELETE_LAST_X_ROWS_ITC_TABLE + str(number) + ")"
            elif table_name.lower() == 'capriole2':
                db_command = DELETE_LAST_X_ROWS_CAPRIOLE_TABLE + str(number) + ")"
            elif table_name.lower() == 'bmp':
                db_command = DELETE_LAST_X_ROWS_BMP_TABLE + str(number) + ")"
            elif table_name.lower() == 'bmp2':
                db_command = DELETE_LAST_X_ROWS_BMP2_TABLE + str(number) + ")"                
            elif table_name.lower() == 'manta':
                db_command = DELETE_LAST_X_ROWS_MANTA_TABLE + str(number) + ")"   
            elif table_name.lower() == 'augmento':
                db_command = DELETE_LAST_X_ROWS_AUGMENTO_TABLE + str(number) + ")"      
            elif table_name.lower() == 'hyblock':
                db_command = DELETE_LAST_X_ROWS_HYBLOCK_TABLE + str(number) + ")"         
            elif table_name.lower() == 'etf':
                db_command = DELETE_LAST_X_ROWS_ETF_TABLE + str(number) + ")"       
            elif table_name.lower() == 'bier':
                db_command = DELETE_LAST_X_ROWS_BIER_TABLE + str(number) + ")"         
            elif table_name.lower() == 'macro':
                db_command = DELETE_LAST_X_ROWS_MACRO_TABLE + str(number) + ")"                                                                            
            else:
                return []
            print(db_command)
            cursor.execute(db_command)
            return


#@anvil.server.callable
#def upload_macro_table(file, table_name):
  # Functionality removed due to Anvil dependency removal
#    pass 

#@anvil.server.callable
#def upload_itc_table(file, table_name):
  # Functionality removed due to Anvil dependency removal
#    pass  

# ----------------------------- Category Helper Functions (moved from datatable.py) -----------------------------

def read_categories():
    """
    Load metric data from the Postgres table 'bier_categories' and return as a DataFrame.
    Returns: pandas.DataFrame: DataFrame with categories data.
    """
    dict_categories = read_bier_categories()
    df_categories = pd.DataFrame.from_dict(dict_categories)    
    if 'id' in df_categories.columns:
        df_categories = df_categories.drop(columns=['id'])
    return df_categories  


def load_category_list(category: str, metric='', df=None) -> list:
    """
    Load the category list from the database and return the metrics for the specified category.
    
    Args:
        category (str): The category to filter by (e.g., 'capriole', 'bmp', etc.)
        metric (str): Optional metric name for single metric mode
        df (DataFrame): Optional pre-loaded categories DataFrame
        
    Returns:
        tuple: (category_list, category_norm_list) - Lists of metric names for the specified category
    """
    try:
      category_base = category.split('_')[0]
    except:
      category_base = category

    if category_base == 'single': # use single metric
        category_list = [metric]
        category_norm_list = [f'{metric}_norm']
    else:    
        if df is None:
          df = read_categories()
        # Filter and get the metrics where the category column is True
        df_true = df[df[category] == True]
        category_list = df_true['metric'].tolist()    
        category_norm_list = [f"{metric}_norm" for metric in category_list]  

    return category_list, category_norm_list


def read_category_weight_table():
    """
    Read category weights from the database and return as a DataFrame.
    Returns: pandas.DataFrame: DataFrame with category weights.
    """
    dict_weights = read_bier_category_weight()
    df_weights = pd.DataFrame.from_dict(dict_weights)    
    if 'id' in df_weights.columns:
        df_weights = df_weights.drop(columns=['id'])
    return df_weights 


def update_custom_metrics(selected_metrics):
    """
    Update the custom column in bier_categories table.
    Set custom=1 for selected metrics, custom=0 for all others.
    
    Args:
        selected_metrics (list): List of metric names to mark as custom
    """
    from psycopg2 import sql
    
    try:
        # Use the global connection
        with connection.cursor() as cur:
            # First, set all custom values to 0
            cur.execute("UPDATE bier_categories SET custom = 0")
            
            # Then, set custom=1 for selected metrics
            if selected_metrics:
                query = sql.SQL("UPDATE bier_categories SET custom = 1 WHERE metric IN ({})").format(
                    sql.SQL(', ').join(sql.Placeholder() * len(selected_metrics))
                )
                cur.execute(query, selected_metrics)
            
            # Commit changes
            connection.commit()
            
            print(f"Updated custom metrics: {selected_metrics}")
        
    except Exception as e:
        print(f"Error updating custom metrics: {e}")
        connection.rollback()
        raise
  