import anvil.email
import anvil.users
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server

import datetime
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import http.client
from bs4 import BeautifulSoup
import json
from io import StringIO
from dotenv import load_dotenv
import os
load_dotenv()

from . import database 
from . import datatable
from . import matrix_strategy 
from . import date_time


BASE = "https://www.bitcoinmagazinepro.com"
LOGIN_URL = f"{BASE}/accounts/login/"
WELCOME_URL = f"{BASE}/welcome/"

BMP_USERNAME = os.environ['BMP_USERNAME']
BMP_PASSWORD = os.environ['BMP_PASSWORD']


# ----------------------------- Session Login -------------------------------------------------------------


def get_csrf_from_form(html):
    soup = BeautifulSoup(html, "html.parser")
    inp = soup.find("input", {"name": "csrfmiddlewaretoken"})
    return inp["value"] if inp and inp.get("value") else None

def ensure_csrf(session):
    # Prefer cookie 'csrftoken' if present
    token = session.cookies.get("csrftoken")
    if token:
        return token
    # Fallback: do a GET to a same-origin page that sets/refreshes CSRF cookie
    r = session.get(WELCOME_URL, timeout=30)
    r.raise_for_status()
    token = session.cookies.get("csrftoken")
    if not token:
        # Last fallback: try to parse a hidden input from the page
        token = get_csrf_from_form(r.text)
    if not token:
        raise RuntimeError("Unable to obtain CSRF token.")
    return token

def login_and_get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0.0.0 Safari/537.36")
    })

    # 1) GET login page to get CSRF
    r = s.get(LOGIN_URL, timeout=30)
    r.raise_for_status()
    csrf = get_csrf_from_form(r.text) or s.cookies.get("csrftoken")
    if not csrf:
        raise RuntimeError("Could not find CSRF token on login page.")

    # 2) POST credentials with CSRF and same-origin headers
    payload = {
        "username": BMP_USERNAME,
        "password": BMP_PASSWORD,
        "csrfmiddlewaretoken": csrf,
        "next": "/welcome/",
    }
    headers = {
        "Referer": LOGIN_URL,
        "Origin": BASE,
    }
    r2 = s.post(LOGIN_URL, data=payload, headers=headers, timeout=30, allow_redirects=True)
    # Basic login check (adjust as needed)
    if r2.status_code not in (200, 302):
        raise RuntimeError(f"Login failed: status={r2.status_code}")

    # Touch a known page (welcome) to finalize session and set cookies
    r3 = s.get(WELCOME_URL, headers={"Referer": WELCOME_URL}, timeout=30)
    if r3.status_code not in (200, 302):
        raise RuntimeError(f"Post-login check failed: status={r3.status_code}")

    return s

# ----------------------------- Bitcoinmagazine Pro -------------------------------------------------------------

# ----------------------------- Auto Import BMP Data -------------------------------------------------------------

@anvil.server.background_task
@anvil.server.callable
def auto_import_bmp_data():

    session = login_and_get_session()


    metrics = [ 'sopr', 'mvrv', 'liquidity','reserve_risk', 'bitcoin_sentiment', 'addresses_in_profit', 'rhodl_ratio', 
               'miner_fee_pct', 'realized_price_sth', 'sth_supply', 'vdd_multiple', 'fear_greed', 'funding_rate', 'nvt', 
               'sth_mvrv', 'lth_mvrv', 'financial_stress', 'high_yield_credit', 'm2_yoy_change', 'yield_spread', 
               'btc_etf_flows', 'cycle_capital_flows', 'bitcoin_cycle_master', 'onchain_prediction', 'realized_price_lth',
               'pi_cycle_oscillator', 'everything_indicator'             
               ]

    #metrics = ['lth_mvrv']

    df = query_bmp_metric(session, 'nupl') # create DF with dates and NUPL metric
    #print('BMP specific - NUPL: ',df, df.info()) 

    for metric in metrics:
        print('Metric to run: ',metric)
        if metric != 'nupl':
            df_metric = query_bmp_metric(session, metric)
            print(f'BMP specific - {metric}: ',df_metric.tail(), df_metric.info())
            df = pd.merge(df, df_metric[['time', metric]],on='time', how="outer")    
            print(f'BMP merge - {metric}: ',df.tail(), df.info())

    """"        

    df.drop('realized_price_sth_y', axis=1, inplace=True)
    df = df.rename(columns={'price': 'close', 'time': 'date', 'realized_price_sth_x': 'realized_price_sth'}) """

    try:
        df['btc_etf_flows'] = df['btc_etf_flows'].fillna(0)
    except:
        pass

    df = df.rename(columns={'price': 'close'})
    df['date'] = df['time'].dt.strftime('%Y-%m-%d')
    df = df.drop('time', axis=1)    
    df = df.fillna(method='ffill')
    # Dropping last row using drop
    df.drop(df.tail(1).index, inplace = True)
    df.to_csv('bmp_data.csv')
    print('BMP: ', df.info())

    # add imported data to database
    database.store_metric_table(df,'bmp2')

    return


def query_bmp_metric(session: requests.Session, metric):
    # Works for following metrics:
    # Liquidity, STH_Supply, NUPL, Reserve-Risk, Bitcoin-Sentiment, Adresses in Profit, rhodl_ratio, puell_multiple,
    # realized_price_sth, realized_price_lth, miner fee pct, vdd_multiple, Fear&Gread
    
    # Works not for:
    # MVRV, SOPR, Funding_rate, bitcoin_cycle_master, onchain_prediction, 
    
    #data = query_bmp_data_full(metric)
    data_json, figure = query_bmp_data(session, metric)

    if metric == 'sopr':
        data_date = data_json['response']['chart']['figure']['data'][1]['x']
        data_metric = data_json['response']['chart']['figure']['data'][1]['y']  
        print(f'{metric}: ',data_date[:50], data_metric[:50])      
    
    elif metric == 'mvrv':
        data_date = data_json['response']['chart']['figure']['data'][7]['x']
        data_metric = data_json['response']['chart']['figure']['data'][7]['customdata']      
        print(f'{metric}: ',data_date[:50], data_metric[:50])
    
    elif metric == 'lth_mvrv':
        data_date = data_json['response']['chart']['figure']['data'][4]['x']
        data_metric = data_json['response']['chart']['figure']['data'][4]['customdata']      
        print(f'{metric}: ',data_date[:50], data_metric[:50])

    elif metric == 'pi_cycle_oscillator':
        data_date = data_json['response']['chart']['figure']['data'][4]['x']
        data_metric = data_json['response']['chart']['figure']['data'][7]['y']        
        print(f'{metric}: ',data_date[:50], data_metric[:50])

    elif metric == 'cycle_capital_flows':
        data_price = data_json['response']['chart']['figure']['data'][0]['customdata']    
        data_date = data_json['response']['chart']['figure']['data'][1]['x']
        data_short_term = data_json['response']['chart']['figure']['data'][1]['y']
        data_long_term = data_json['response']['chart']['figure']['data'][2]['y']  
    
    elif metric == 'fear_greed':
        data_metric = data_json['response']['chart']['figure']['data'][1]['customdata']
        columns = ['time_str', 'price', metric, 'regime', 'signal']
        df_metric = pd.DataFrame(data_metric, columns=columns) 
        df_metric['time'] = pd.to_datetime(df_metric['time_str'], format = 'ISO8601')                  
        df = df_metric[['time', metric]].copy()        
        #print('F&G: ', df.tail(3), df.info())

    elif metric == 'funding_rate':
        data_metric = data_json['response']['chart']['figure']['data'][0]['customdata']
        columns = ['nb','time_str', 'regime', metric, 'funding2', 'time2', 'price']
        df_metric = pd.DataFrame(data_metric, columns=columns) 
        df_metric['time'] = pd.to_datetime(df_metric['time_str'], format = 'ISO8601')                  
        df = df_metric[['time', metric]].copy()        
        #print('Funding: ', df.tail(3), df.info())  
        # Convert to daily sum
        df['time_str'] = df['time'].dt.strftime('%Y-%m-%d')
        df = df.groupby('time_str')['funding_rate'].sum().reset_index() 
        df['time'] = pd.to_datetime(df['time_str'], format = 'ISO8601')         
        print('Daily Funding: ', df.tail(3), df.info())            

    else:
        try:
            data_date = data_json['response']['chart']['figure']['data'][0]['x']
            data_price = data_json['response']['chart']['figure']['data'][0]['y']        
            data_metric = data_json['response']['chart']['figure']['data'][0]['customdata']
        except:
            data_metric = []

    try:
        print('Length BMP data: ', len(data_metric)) # data_metric[:500], 
    except:
        pass

    # Create a DataFrame with time and price
    if metric not in ['fear_greed', 'funding_rate']:
        df_time = pd.DataFrame({'time_str': data_date})
        df_time['time'] = pd.to_datetime(df_time['time_str'], format = 'ISO8601') 
        print('BMP time: ', df_time.tail(2))

    # Merge the dataframes
    if metric in ['sopr','mvrv','lth_mvrv', 'pi_cycle_oscillator']:

        """# 5) Build DataFrame; let pandas infer datetime and coerce bad rows to NaT/NaN
        df = pd.DataFrame({
            'time': pd.to_datetime(data_date, errors='coerce'),
            metric: pd.to_numeric(data_metric, errors='coerce')
        })"""

        df_metric = pd.DataFrame({metric: data_metric})
        print(f'BMP {metric} raw DF: ', df_metric.tail(3), df_metric.info())
        if metric in ['mvrv','lth_mvrv']:
            df_metric[metric] = df_metric[metric].apply(lambda x: x[0])
        print(f'BMP {metric} DF: ', df_metric.tail(3), df_metric.info())
        df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric[[metric]]], axis=1)
        print('BMP metric combi: ', metric, df.tail(3)) #, df.info())

    elif metric in ['cycle_capital_flows']:           
        df_metric = pd.DataFrame({'short_term_cycle': data_short_term, 'long_term_cycle': data_long_term})
        df_metric[metric] = (0.5 + (df_metric['short_term_cycle'] - df_metric['long_term_cycle']) / 2) * 100 # Merge into one metric
        df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric], axis=1)

    elif metric in ['fear_greed', 'funding_rate']:
        # Fully handled in dedicated section above  
        pass         

    else:   
        df_metric = pd.DataFrame({'price': data_price, metric: data_metric})
        df_metric[metric] = df_metric[metric].apply(lambda x: x[0])        
        df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric[['price', metric]]], axis=1)

    print('BMP metric: ', metric, df.tail(3)) #, df.info())
    
    if (metric == 'sth_mvrv'):
        name_ratio = metric + '_ratio'
        df[name_ratio] = df['price'] / df[metric] 

    elif (metric == 'm2_yoy_change'): 
        # Delta to the M2 365 rows (days) earlier
        m2_yoy_change = df['m2_yoy_change'].copy()
        #df['m2_yoy_change'] = m2_yoy_change - m2_yoy_change.shift(365) # absoute change
        df['m2_yoy_change'] = m2_yoy_change.pct_change(periods=365) # relative change

    #elif (metric == 'yield_spread'):
    #    # swap column names
    #    df.rename(columns = {"price": "price_bu", metric : "price"}, inplace = True) 
    #    df.rename(columns = {"price_bu" : metric}, inplace = True) 

    print('BMP metric: ', metric, df.tail(3)) #, df.info())

    return df

# ----------------------------- Fetch data -------------------------------------------------------------


def query_bmp_data(session: requests.Session, metric):
    if metric.lower() == 'liquidity':
        metric_header = "global-liquidity/"
        metric_endpoint = "global_liquidity"
        data, figure = query_bmp_data_liquidity(session, metric_header, metric_endpoint)
        return data, figure
        
    elif metric.lower() == 'nupl':
        metric_header = "relative-unrealized-profit--loss/"
        metric_endpoint = "unrealised_profit_loss"

    elif metric.lower() == 'mvrv':    
        metric_header = "mvrv-zscore"
        metric_endpoint = "mvrv_zscore"

    elif metric.lower() == 'reserve_risk':
        metric_header = "reserve-risk/"
        metric_endpoint = "reserve_risk"

    elif metric.lower() == 'bitcoin_sentiment': 
        # Active Adress Sentiment Indicator = Bitcoin Sentiment
        metric_header = "active-address-sentiment-indicator/"
        metric_endpoint = "bitcoin_sentiment"

    elif metric.lower() == 'addresses_in_profit': 
        metric_header = "percent-addresses-in-profit/"
        metric_endpoint = "addresses_in_profit"
        
    elif metric.lower() == 'rhodl_ratio': 
        metric_header = "rhodl-ratio/"
        metric_endpoint = "rhodl_ratio"  

    elif metric.lower() == 'puell_multiple': 
        metric_header = "puell-multiple/"
        metric_endpoint = "puell_multiple"

    elif metric.lower() == 'miner_fee_pct': 
        # Percentage of Miner Fees as % of overall Miner Revenues
        metric_header = "bitcoin-miner-revenue-fees-vs-rewards/"
        metric_endpoint = "miner_revenue_fees_pct"

    elif metric.lower() == 'sopr': 
        metric_header = "sopr-spent-output-profit-ratio/"
        metric_endpoint = "sopr"
        
    elif metric.lower() == 'funding_rate': 
        metric_header = "bitcoin-funding-rates"
        metric_endpoint = "funding_rates"
        data, figure = query_bmp_data_funding(session, metric_header, metric_endpoint)
        return data, figure        
        
    elif metric.lower() == 'fear_greed': 
        metric_header = "bitcoin-fear-and-greed-index"
        metric_endpoint = "fear_and_greed"
        data, figure = query_bmp_data_fear_greed(session, metric_header, metric_endpoint)
        return data, figure        
        
    elif metric.lower() == 'bitcoin_cycle_master': 
        metric_header = "bitcoin-cycle-master"
        metric_endpoint = "bitcoin_cycle_master"
        
    elif metric.lower() == 'onchain_prediction': 
        metric_header = "bitcoin-price-prediction"
        metric_endpoint = "price_prediction"        
        
    elif metric.lower() == 'realized_price_sth': 
        metric_header = "short-term-holder-realized-price"
        metric_endpoint = "realized_price_sth"
        
    elif metric.lower() == 'realized_price_lth': 
        metric_header = "long-term-holder-realized-price"
        metric_endpoint = "realized_price_lth"        
        
    elif metric.lower() == 'sth_supply': 
        metric_header = "short-term-holder-supply"
        metric_endpoint = "sth_supply"
        data, figure = query_bmp_data_sth_supply(session, metric_header, metric_endpoint)
        return data, figure        
       
    elif metric.lower() == 'vdd_multiple': 
        metric_header = "value-days-destroyed-multiple"
        metric_endpoint = "vdd_multiple"
              
    elif metric.lower() == 'nvt': 
        metric_header = "advanced-nvt-signal"
        metric_endpoint = "nvts"
              
    elif metric.lower() == 'pi_cycle_oscillator': 
        metric_header = "pi-cycle-top-bottom-indicator"
        metric_endpoint = "pi_cycle_top_bottom_indicator"
               
    elif metric.lower() == 'sth_mvrv': 
        metric_header = "short-term-holder-mvrv"
        metric_endpoint = "mvrv_sth"

    elif metric.lower() == 'lth_mvrv': 
        metric_header = "long-term-holder-mvrv"
        metric_endpoint = "mvrv_lth"
      
    elif metric.lower() == 'financial_stress': 
        metric_header = "financial-stress-index-vs-btc"
        metric_endpoint = "financial_stress_index_vs_btc"
              
    elif metric.lower() == 'high_yield_credit': 
        metric_header = "bitcoin-cycles-vs-high-yield-credit-cycles"
        metric_endpoint = "high_yield_credit_vs_btc"
  
    elif metric.lower() == 'm2_yoy_change': 
        metric_header = "m2-global-vs-btc"
        metric_endpoint = "m2_global_vs_btc"
        
    elif metric.lower() == 'yield_spread': 
        metric_header = "10yr-2yr-yield-spread"
        metric_endpoint = "us_2y_10y_spread"
        data, figure = query_bmp_data_yield_spread(session, metric_header, metric_endpoint)
        return data, figure          
        
    elif metric.lower() == 'btc_etf_flows': 
        metric_header = "bitcoin-etf-daily-flows-usd/"
        metric_endpoint = "etf_daily_flows_usd"
        data, figure = query_bmp_data_btc_etf_flow(session, metric_header, metric_endpoint)
        return data, figure         
    
    elif metric.lower() == 'cycle_capital_flows': 
        metric_header = "cycle-capital-flows"
        metric_endpoint = "cycle_capital_flows"

    elif metric.lower() == 'everything_indicator': 
        metric_header = "everything-indicator"
        metric_endpoint = "everything_indicator"  

    else:
        print('wrong metric')

    data, figure = query_bmp_data_base(session, metric_header, metric_endpoint)
    
    return data, figure


def query_bmp_data_liquidity(session: requests.Session, metric_header, metric_endpoint): 
    """ adds special payload for liquidity and calls query function """
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "..chart.figure...recent-data-button-desktop.hidden...recent-data-button-mobile.hidden..",
        "outputs": [{"id": "chart", "property": "figure"}, 
                    {"id": "recent-data-button-desktop", "property": "hidden"}, 
                    {"id": "recent-data-button-mobile", "property": "hidden"}],
        "inputs": [{"id": "url", "property": "pathname", "value": ""},
                   {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"]}
    
    data, figure = query_bmp_data_special(session, metric_header, metric_endpoint, payload)
    
    return data, figure


def query_bmp_data_fear_greed(session: requests.Session, metric_header, metric_endpoint): 
    """ adds special payload for Fear & Greed and calls query function """
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "..chart.figure...indicator.figure...last_update.children...now.children...now.style..",
        "outputs": [{"id": "chart", "property": "figure"}, 
                    {"id": "indicator", "property": "figure"}, 
                    {"id": "last_update", "property": "children"},
                    {"id": "now", "property": "children"},        
                    {"id": "now", "property": "style"}],         
        "inputs": [{"id": "url", "property": "pathname", "value": ""},
                   {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"]}
    
    data, figure = query_bmp_data_special(session, metric_header, metric_endpoint, payload)
    
    return data, figure

def query_bmp_data_funding(session: requests.Session, metric_header, metric_endpoint): 
    """ adds special payload for Funfing Rate and calls query function """
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "..chart.figure...exchange.options...resolution.disabled...resolution.value...exchange.value..",
        "outputs": [{"id": "chart", "property": "figure"}, 
                    {"id": "exchange", "property": "options"}, 
                    {"id": "resolution", "property": "disabled"},
                    {"id": "resolution", "property": "value"},        
                    {"id": "exchange", "property": "value"}],         
        "inputs": [{"id": "currency", "property": "value", "value": "funding_rate_usd"},
                   {"id": "exchange", "property": "value", "value": "average"},
                   {"id": "resolution", "property": "value", "value": "1h"},                   
                   {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"]}
    
    data, figure = query_bmp_data_special(session, metric_header, metric_endpoint, payload)
    
    return data, figure

def query_bmp_data_sth_supply(session: requests.Session, metric_header, metric_endpoint): 
    """ adds special payload for STH Supply and calls query function """
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "chart.figure",
        "outputs": {"id": "chart", "property": "figure"},        
        "inputs": [{"id": "url", "property": "pathname", "value": ""},
                   {"id": "period", "property": "value", "value": "all"},                 
                   {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"]}
    
    data, figure = query_bmp_data_special(session, metric_header, metric_endpoint, payload)
    
    return data, figure

def query_bmp_data_yield_spread(session: requests.Session, metric_header, metric_endpoint): 
    """ adds special payload for Yield Spread and calls query function """
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "chart.figure",
        "outputs": {"id": "chart", "property": "figure"},        
        "inputs": [{"id": "url", "property": "pathname", "value": ""},
                   {"id": "period", "property": "value", "value": "5y"},                 
                   {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"]}
    
    data, figure = query_bmp_data_special(session, metric_header, metric_endpoint, payload)
    
    return data, figure

def query_bmp_data_btc_etf_flow(session: requests.Session, metric_header, metric_endpoint): 
    """ adds special payload for Yield Spread and calls query function """
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "chart.figure",
        "outputs": {"id": "chart", "property": "figure"},        
        "inputs": [{"id": "url", "property": "pathname", "value": ""},
                   {"id": "etf", "property": "value", "value": "total"},                 
                   {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"]}
    
    data, figure = query_bmp_data_special(session, metric_header, metric_endpoint, payload)
    
    return data, figure
    
def query_bmp_data_special(session: requests.Session, metric_header, metric_endpoint, payload):  
    """ Query special BMPro cases, where payload is provided as parameter """

    CHART_URL = f"{BASE}/charts/{metric_header}/"
    DASH_ENDPOINT = f"{BASE}/django_plotly_dash/app/{metric_endpoint}/_dash-update-component"
    
    # Visit the chart page (helps set any page-specific cookies/context)
    session.get(CHART_URL, headers={"Referer": CHART_URL}, timeout=30)

    # Ensure we have a CSRF token (cookie preferred)
    csrf = ensure_csrf(session)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": BASE,
        "Referer": CHART_URL,
        "X-CSRFToken": csrf,  # CSRF from cookie
        # Optional but harmless:
        "User-Agent": session.headers.get("User-Agent", ""),
    }

    resp = session.post(DASH_ENDPOINT, data=json.dumps(payload), headers=headers, timeout=45)
    if resp.status_code >= 400:
        raise RuntimeError(f"Dash endpoint error: {resp.status_code}\n{resp.text[:1000]}")

    # The response is a Dash JSON payload. Typically, figure will be in response[0]['response']['props']['figure']
    data = resp.json()

    # Try to extract the figure safely (structure can vary)
    figure = None
    try:
        # Dash often returns {"response": [{"props": {"figure": {...}}, "id": "chart", "property": "figure"}], ...}
        resp_list = data.get("response") or data.get("responses")
        if isinstance(resp_list, list) and resp_list:
            item = resp_list[0]
            props = item.get("props", {})
            figure = props.get("figure")
    except Exception:
        pass

    return data, figure

def query_bmp_data_base(session: requests.Session, metric_header, metric_endpoint):    

    CHART_URL = f"{BASE}/charts/{metric_header}/"
    DASH_ENDPOINT = f"{BASE}/django_plotly_dash/app/{metric_endpoint}/_dash-update-component"
    
    # Visit the chart page (helps set any page-specific cookies/context)
    session.get(CHART_URL, headers={"Referer": CHART_URL}, timeout=30)

    # Ensure we have a CSRF token (cookie preferred)
    csrf = ensure_csrf(session)
    
    # Dash update payload (mirroring what the page sends)
    payload = {
        "output": "chart.figure",
        "outputs": {"id": "chart", "property": "figure"},
        "inputs": [
            {"id": "url", "property": "pathname", "value": f"/charts/{metric_header}/"},
            {"id": "display", "property": "children", "value": "xl 1251px"}],
        "changedPropIds": ["url.pathname", "display.children"],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": BASE,
        "Referer": CHART_URL,
        "X-CSRFToken": csrf,  # CSRF from cookie
        # Optional but harmless:
        "User-Agent": session.headers.get("User-Agent", ""),
    }

    resp = session.post(DASH_ENDPOINT, data=json.dumps(payload), headers=headers, timeout=45)
    if resp.status_code >= 400:
        raise RuntimeError(f"Dash endpoint error: {resp.status_code}\n{resp.text[:1000]}")

    # The response is a Dash JSON payload. Typically, figure will be in response[0]['response']['props']['figure']
    data = resp.json()

    # Try to extract the figure safely (structure can vary)
    figure = None
    try:
        # Dash often returns {"response": [{"props": {"figure": {...}}, "id": "chart", "property": "figure"}], ...}
        resp_list = data.get("response") or data.get("responses")
        if isinstance(resp_list, list) and resp_list:
            item = resp_list[0]
            props = item.get("props", {})
            figure = props.get("figure")
    except Exception:
        pass

    return data, figure
