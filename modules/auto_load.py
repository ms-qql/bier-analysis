

import datetime
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import os
import requests
import http.client
import json
from io import StringIO

from . import database 

from . import matrix_strategy 
from . import date_time
from . import auto_load_bmp
from . import auto_load_capriole

hyblock_list = ['volume_delta','whale_retail','user_bot_ratio','usdt_premium','bid_ask_ratio','top_traders_long','market_order_size','market_order_count','limit_order_count','funding_rate','fear_greed','bid_ask_delta','long_liquidations','short_liquidations','oi_delta','bvol','dvol']


def date_today():
    # returns today's date as string
    now = datetime.now() # current date and time
    today = now.strftime("%Y-%m-%d %H:%M")
    return today

def date_tomorrow():
    # returns tomorrow's date as string
    now = datetime.now() # current date and time
    tomorrow = now + timedelta(days = 1) # Calculate yesterdays' date
    tomorrow_str = tomorrow.strftime("%Y-%m-%d %H:%M")
    return tomorrow_str

def date_yesterday():
    # returns tomorrow's date as string
    now = datetime.now() # current date and time
    yesterday = now - timedelta(days = 1) # Calculate yesterdays' date
    yesterday_str = yesterday.strftime("%Y-%m-%d %H:%M")
    return yesterday_str  

def iso_to_unix(iso_time_str):
    # Define the format of the input ISO time string
    iso_format = '%Y-%m-%d %H:%M'
    # Parse the ISO time string into a datetime object
    dt = datetime.strptime(iso_time_str, iso_format)
    # Convert the datetime object to Unix timestamp
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp


# ----------------------------- All data -------------------------------------------------------------


def auto_import_all_data():
    # Load all data
    try:
      auto_load_capriole.auto_import_capriole_data()
      print('Capriole autoload successful')      
    except:
      print('Capriole autoload failed')
    try:
      auto_load_bmp.auto_import_bmp_data()  
      print('BMP autoload successful')      
    except:
      print('BMP autoload failed')
    try:
      auto_import_augmento_data()
      print('Augmento autoload successful')          
    except:
      print('Augmento autoload failed')
    try:
      auto_import_itc_data() 
      print('ITC autoload successful')      
    except:
      print('ITC autoload failed')
    #anvil.server.call('auto_import_etf_data')    
    #anvil.server.call('auto_import_hyblock_data') # ERROR  
    
    return

# ----------------------------- ITC -------------------------------------------------------------


def auto_import_itc_data():

    base_url ='https://app.intothecryptoverse.com/api/v2/'
    feature_url = 'risk-models/price-based/crypto'
    api_key = 'tzTzrkCdKnhtKzTDJJt6b'
    url = base_url + feature_url + '?apikey=' + api_key
    response = requests.get(url)
    data = response.json()['data']
    print('ITC Data: ', date_time.date_today(), data['BTC'], data['ETH'], data['BNB'], data['SOL'], data['LTC'])
    database.add_row_itc('itc', date_time.date_today(), 0.0, 0.0, 0.0, data['BTC'], 0.0, 0.0, 0.0, 0.0, 0.0, data['ETH'], 0.0, data['BNB'], 0.0, data['SOL'], 0.0, data['LTC'])

    """

    conn = http.client.HTTPSConnection("fn.intothecryptoverse.com")
    payload = json.dumps({})

    headers = {
    'accept': 'application/json',
    'accept-language': 'en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7',
<<<<<<< HEAD
    'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Ijg1NzA4MWNhOWNiYjM3YzIzNDk4ZGQzOTQzYmYzNzFhMDU4ODNkMjgiLCJ0eXAiOiJKV1QifQ.eyJ0aWVyIjoicHJvIiwiaXNzIjoiaHR0cHM6Ly9zZWN1cmV0b2tlbi5nb29nbGUuY29tL2NyeXB0b3ZlcnNlLWRlNTIyIiwiYXVkIjoiY3J5cHRvdmVyc2UtZGU1MjIiLCJhdXRoX3RpbWUiOjE2OTI2OTAwNjAsInVzZXJfaWQiOiJIalVyQ0huREswYnNvM2RxbjlGU0dYdDZuazMyIiwic3ViIjoiSGpVckNIbkRLMGJzbzNkcW45RlNHWHQ2bmszMiIsImlhdCI6MTc0NDcwNDY4NSwiZXhwIjoxNzQ0NzA4Mjg1LCJlbWFpbCI6Im1hbmZyZWRAYWlrcS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJtYW5mcmVkQGFpa3EuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiY3VzdG9tIn19.Hy2xSk5sSyswO8vT1d2qSLFVK_rYCJYZ2gzs1AF_twX1VAEnhUrnko5YUYjLRppx5kMYukB3YGeiftX0v08oAU2FY_dt658ZesI1N3kvt12EAlsYVrDQLPYgn_EM6N4Lp6EckkajFMzW-23rQF9zF6VGXLXle9IMyhy60OoVX_7ec3rDi9N3eXUyQsbPAGaRLby28mbSKeNtz0fCnw2cHG-7fQvT8BPaANifhjgdv-7Ull3lKF5ID9BHvl5fjbI5jdcGeYxY7rIQjM0O3J8EC5mVldiXPJzwyrOMcZqdfKBs1I3VLPnncq2QLGZbiYkPpwf_VowhUdq5MUKkL_SFVA',
=======
    'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjMwYjIyMWFiNjU2MTdiY2Y4N2VlMGY4NDYyZjc0ZTM2NTIyY2EyZTQiLCJ0eXAiOiJKV1QifQ.eyJ0aWVyIjoicHJvIiwiaXNzIjoiaHR0cHM6Ly9zZWN1cmV0b2tlbi5nb29nbGUuY29tL2NyeXB0b3ZlcnNlLWRlNTIyIiwiYXVkIjoiY3J5cHRvdmVyc2UtZGU1MjIiLCJhdXRoX3RpbWUiOjE2OTI2OTAwNjAsInVzZXJfaWQiOiJIalVyQ0huREswYnNvM2RxbjlGU0dYdDZuazMyIiwic3ViIjoiSGpVckNIbkRLMGJzbzNkcW45RlNHWHQ2bmszMiIsImlhdCI6MTc0MjgwNTAwNSwiZXhwIjoxNzQyODA4NjA1LCJlbWFpbCI6Im1hbmZyZWRAYWlrcS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJtYW5mcmVkQGFpa3EuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiY3VzdG9tIn19.Cb9imDOuvsnpS4mk1XuEfIaxFGVjYKVzht4il_mLx70axWRZ2o65ESmvFIwdPl5NgqhMOugx0enH-N5AYY7qjVGsNwOLxuErzxJuR4NLai8d4wUWORH0JIDINprpScpZywmK-rzgB7iPXuvXvndyqDOYsLi6NKs1pGWoGh2bn5SGY8BhvWRRRqFnRgRpr5MV7x8aPPfEbA3-fLSA2fiHB3klmsHq6PkxNLZfnmyBbF9iuvYS8Af8ggHOmTZIftILH7VA6oLFVJgW-2fZHntdDd0P8bwbL7HIq6ssL_0iihdA7ZDzCecDS5YIVai_1McWALv430JcgrwDgk1BhF--wA',
    'content-type': 'application/json',
    'origin': 'https://app.intothecryptoverse.com',
>>>>>>> e762b0d6c83507277333929b2ee2b32a1a921d64
    'priority': 'u=1, i',
    'referer': 'https://app.intothecryptoverse.com/charts/risk',
    'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
    'Cookie': '_ga=GA1.1.1203265883.1692689987; cf_clearance=xkC52sQ905YI8xdvfyhCPLmegolnrFlOoyWEAk5BB5k-1732547831-1.2.1.1-p2s9GhLVl_X0amYfqiNRO0lSGqy.pNQ28NMSU0RRYGSR_bnfq4wloLpy7vvqphYAzOjVZUtcDPn.uM_AiW.BxV9BWjCEXYlmglY8f2nb2RISgAZtCSsNLiRJh0McGBiy0Ogypf2Vkr20Hgh7Ko9fUSdrv2BSElFItWFBme12ldp89QpQ5lcr9gB4bkvCczpTfmv55Yy0YjCrZXc.l.5MryKiJVNyWbtkswm4JeIJSDmB7KT1Umf67SBm.fwYIOg2Q43rITAJ1sXB7900qE0reppDvPGXU0TU0uCmoMRU8gZv3AA3jHYbC7E5eEG.GUToEpItUyxWXGiWi.uO_SkvspZpad_NrezT8LdNqEUVh94eAxA8JNJ5uoixQ0b.uilo; _ga_RKER7WEQGN=GS1.1.1744704654.58.1.1744704680.0.0.0; ph_phc_NrYHi5vjwe6TVaCCG0Wu2ppydeJ4pFfR7Mp8SuV1aXD_posthog=%7B%22distinct_id%22%3A%22019593b9-eb7a-74a9-84b4-435cdf97e675%22%2C%22%24sesid%22%3A%5B1744704681730%2C%2201963880-5195-7550-9608-722a3afbf7b2%22%2C1744704655765%5D%2C%22%24epp%22%3Atrue%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Fapp.intothecryptoverse.com%2Fcharts%2Frisk%22%7D%7D'
    }
                
    headers_old = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7',
    'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjhkMjUwZDIyYTkzODVmYzQ4NDJhYTU2YWJhZjUzZmU5NDcxNmVjNTQiLCJ0eXAiOiJKV1QifQ.eyJ0aWVyIjoicHJvIiwiaXNzIjoiaHR0cHM6Ly9zZWN1cmV0b2tlbi5nb29nbGUuY29tL2NyeXB0b3ZlcnNlLWRlNTIyIiwiYXVkIjoiY3J5cHRvdmVyc2UtZGU1MjIiLCJhdXRoX3RpbWUiOjE2OTI2OTAwNjAsInVzZXJfaWQiOiJIalVyQ0huREswYnNvM2RxbjlGU0dYdDZuazMyIiwic3ViIjoiSGpVckNIbkRLMGJzbzNkcW45RlNHWHQ2bmszMiIsImlhdCI6MTczOTM0NzI1NSwiZXhwIjoxNzM5MzUwODU1LCJlbWFpbCI6Im1hbmZyZWRAYWlrcS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJtYW5mcmVkQGFpa3EuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiY3VzdG9tIn19.aTWngXH4k7z8XKiRWfo5YuFcdibQysXHd0LvdOaOnOlGssmX9-8SG3-tFRevtmt7q7gBE60QCv9HnibiwCSGE_aOyzGgSuhWYGByh0ZopBc7vO8tZj7hOsXt5TEaBxZpfaSiU_3XtOkW1A3wdoNl1xP7ZC2AI3Fp8GAV8u2BF0eadHBY1xuPN9JUMY3GSblOggbRCOk0egctIlTYDhDi9F7Ry1LNOXLJzSoqzIFsmUIjY1AfAJ7nU233gJfN2ghToely0gjqZwXsnl38HgVJMsSMxsdb4I1gqYPXB73rPy0OKCOOLpIhN6wi_y8LeD3Jee3SqShgH45X2RrCqrskpg',
    'content-type': 'application/json',
    'origin': 'https://app.intothecryptoverse.com',
    'priority': 'u=1, i',
    'referer': 'https://app.intothecryptoverse.com/',
    'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
    }

    #conn.request("POST", "/extApi/v1/chart/historical", payload, headers)
    query = "/extApi/v1/chart/historical" 
    conn.request("POST", query, payload, headers)

    res = conn.getresponse()
    data = res.read().decode("utf-8")
    itc_data = json.loads(data)
    print('ITC Data: ', itc_data)

    # Extract the series list
    series = itc_data['data']['series']#[0]

    df = pd.DataFrame(series)
    df = df.rename(columns={'d': 'date', 'btc_usd': 'close', 'btc_risk': 'risk_level', 'alt_mcap_usd': 'alt_mcap', 'dxy_risk': 'dxy_usd_risk', 'eth_risk': 'eth_usd_risk', 'bnb_risk': 'bnb_usd_risk', 'sol_risk': 'sol_usd_risk', 'ltc_risk': 'ltc_usd_risk'})
    #print(df.info())

    df_itc = df[['date', 'mcap', 'mcap_risk', 'close', 'risk_level', 'alt_mcap', 'alt_mcap_risk', 'dxy_usd', 'dxy_usd_risk', 'eth_usd', 'eth_usd_risk', 'bnb_usd', 'bnb_usd_risk', 'sol_usd', 'sol_usd_risk', 'ltc_usd', 'ltc_usd_risk']].copy()
    df_itc.fillna(method="ffill", inplace=True)  
    #print('DF ITC: ', df_itc.tail()) 
    #df_itc.to_csv('itc_risk_levels.csv', index=False)

    # add imported data to database
    database.store_metric_table(df_itc, 'itc') 

    def add_df_itc(df, table_name):
    for d in df.to_dict(orient="records"):
      #add_row_itc(table_name, d['date'], d['market_cap'], d['close'], d['risk_level'], d['btc_dominance'], d['btc_dominance_no_stables'])
      add_row_itc(table_name, d['date'], d['mcap'], d['mcap_risk'], d['close'], d['risk_level'], d['alt_mcap'], d['alt_mcap_risk'], d['dxy_usd'], d['dxy_usd_risk'],  
                  d['eth_usd'], d['eth_usd_risk'], d['bnb_usd'], d['bnb_usd_risk'], d['sol_usd'], d['sol_usd_risk'], d['ltc_usd'], d['ltc_usd_risk'])"""

    return



def auto_import_itc_oi_data():
  # Load Macro Index
  df_data = query_itc_oi_data('future') # Load OI Future data
  print('OI Future: ', df_data, df_data.info())
  return

import http.client
import json

conn = http.client.HTTPSConnection("fn.intothecryptoverse.com")
payload = json.dumps({})
headers = {
  'accept': 'application/json, text/plain, */*',
  'accept-language': 'en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7',
  'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjFlNTIxYmY1ZjdhNDAwOGMzYmQ3MjFmMzk2OTcwOWI1MzY0MzA5NjEiLCJ0eXAiOiJKV1QifQ.eyJ0aWVyIjoicHJvIiwiaXNzIjoiaHR0cHM6Ly9zZWN1cmV0b2tlbi5nb29nbGUuY29tL2NyeXB0b3ZlcnNlLWRlNTIyIiwiYXVkIjoiY3J5cHRvdmVyc2UtZGU1MjIiLCJhdXRoX3RpbWUiOjE2OTI2OTAwNjAsInVzZXJfaWQiOiJIalVyQ0huREswYnNvM2RxbjlGU0dYdDZuazMyIiwic3ViIjoiSGpVckNIbkRLMGJzbzNkcW45RlNHWHQ2bmszMiIsImlhdCI6MTczMTkyNTU5OSwiZXhwIjoxNzMxOTI5MTk5LCJlbWFpbCI6Im1hbmZyZWRAYWlrcS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJtYW5mcmVkQGFpa3EuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiY3VzdG9tIn19.hfuaJrzCCKazwNDQjonP0JoglyBDFUwlW8BU_65cGuXQapzmzar_7PshZ43_EgMi2UYULxgL6rTWCdJnUATH99RMjw6uqqDwR02BzrRv_UvRmNSm4IDo1ABkUUPm2plTPLS2aUFiP31m-cWQwRj_oDX4NGIslWWwjjh-4mRmM2ywAQRex4LkHrOyeje16wgWikcO7MXLQ1ItfzDn9Qh7a1_gJrW9mVn7IMp_LaXzmiMrZnsklbLyUg0dUeZ0uOBED5EEez-W2xleGEOliF4TGNOygi9AdMd9rnbRBKJ4Zj41zQGGDnIPC5Kb578Rr71yZrXIH6STT-Jzf3RQHJ54Cw',
  'content-type': 'application/json',
  'origin': 'https://app.intothecryptoverse.com',
  'priority': 'u=1, i',
  'referer': 'https://app.intothecryptoverse.com/',
  'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sec-fetch-dest': 'empty',
  'sec-fetch-mode': 'cors',
  'sec-fetch-site': 'same-site',
  'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
}
conn.request("POST", "/extApi/v1/chart/historical", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
def query_itc_oi_data(market = 'future'):
    conn = http.client.HTTPSConnection("app.intothecryptoverse.com")
  
    payload = ""
    headers = {
        'accept': "application/json",
        'accept-language': "en-GB,en;q=0.8",
        'authorization': "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImUwM2E2ODg3YWU3ZjNkMTAyNzNjNjRiMDU3ZTY1MzE1MWUyOTBiNzIiLCJ0eXAiOiJKV1QifQ.eyJ0aWVyIjoicHJvIiwiaXNzIjoiaHR0cHM6Ly9zZWN1cmV0b2tlbi5nb29nbGUuY29tL2NyeXB0b3ZlcnNlLWRlNTIyIiwiYXVkIjoiY3J5cHRvdmVyc2UtZGU1MjIiLCJhdXRoX3RpbWUiOjE2MjYzNjc0MjUsInVzZXJfaWQiOiJIalVyQ0huREswYnNvM2RxbjlGU0dYdDZuazMyIiwic3ViIjoiSGpVckNIbkRLMGJzbzNkcW45RlNHWHQ2bmszMiIsImlhdCI6MTcyNjQyMzA1NSwiZXhwIjoxNzI2NDI2NjU1LCJlbWFpbCI6Im1hbmZyZWRAYWlrcS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJtYW5mcmVkQGFpa3EuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiY3VzdG9tIn19.kkLatS4TawuLjclH8_mCDrVP0P7RC2qeoc_JVeONaPVOSuiQBTknsYBoys_kjWAR3YFevW5d4UhKBpW832HC40WQdqL0upe1SiW_wH0Fo0bfmHbJde1P74CgWPnHHrWxbYEqp9HPQkL5-d75S9lbsN5fwnoxiJ7HTEHM4Fkh3wyuxZdzpK-mpzeuIqCH6e3929zCQ9rGLyXDVzTcA5bD3V3OezjwZ3rs_ISzMPA5YfqnI4kE_qmmMOPU5Ody7DDdhQemLsnLSw0jyYKfZWc4mYxWen9Ny_Q6UF5y7Ab19kAWfZX6GbiSLIrR1EnieFOk1BeiOScRM_4OHTeVptVwuw",
        'cookie': "cf_clearance=12asTsMDm5NHt8FycUQzHv.v48T_wKOc4EezVUyxx1Y-1725901990-1.2.1.1-JDGrX9A3rbNaPMdqdsrdPPWY5nQdsGkJ53dje1Fjj5OG..uf9obssbnozhci3WXMQEf0hevPSN0Y4rv.x3Z5mo4_6hTiI_lMSV_1lHu3uJplEZeXT0tFNnkEXH5kcLSV2AJqi.7.0bqmxsYRkVusdo9B59Xjiwlfg8dNQqTgFk.wgD60IgFIvJvRbPUTQhg_rxQ..UztnY0OIYDxNFWl0lYfB2M1UxdE1ZToPuQtmx2_Pr.Vwe69Lkgonm2v71n_F3QjU4canL4d0kVb1ifisZRlJzz9SunXvxVDdy66fqF2vUvdAO6x.94LjQ368w80KAvxCgxCCl5YJFQjaHbvslwtV97oMMNp..0ZUi.H2dXyGL3ya3JWk_y2O9zkjJhpM.ej6NsW.MtsaKlDvPHsfg",
        'priority': "u=1, i",
        'referer': "https://app.intothecryptoverse.com/charts/futures-open-interest",
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': "?0",
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': "empty",
        'sec-fetch-mode': "cors",
        'sec-fetch-site': "same-origin",
        'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
    
    if market.lower() == 'future':
        query = "/api/data/coinglass?rnd=0.9213785584140022&uri=futures%2FopenInterest%2Fexchange-history-chart%3Fsymbol%3DBTC%26range%3Dall%26unit%3DUSD"
    elif market.lower() == 'option':
        query = "/api/data/coinglass?rnd=0.4601994715454105&uri=option%2Fexchange-oi-history%3Fsymbol%3DBTC%26currency%3DUSD"
    else:
        return 'Wrong Market'
    
    conn.request("GET", query, payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    
    data_json = json.loads(data)
    print('JSON: ', data_json)
    if market.lower() == 'option':
        itc_data = data_json[0]
        itc_data['timeList'] = itc_data.pop('dateList')
    else:
        itc_data = data_json
        
    # Convert timeList to ISO format
    iso_times = [datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M') for ts in itc_data['timeList']]
    # Create a DataFrame with time and price
    df = pd.DataFrame({'Time': iso_times, 'Price': itc_data['priceList']})

    # Add columns from dataMap
    for exchange, values in itc_data['dataMap'].items():
        df[exchange] = values

    #df.fillna(0, inplace=True)
    df['Total'] = df.drop(columns=['Time', 'Price']).sum(axis=1)
        
    return df


# ----------------------------- capriole_btc -------------------------------------------------------------

'''

def auto_import_capriole_data():

  # Load Macro Index
  df_data = get_capriole_metric(0) # 0 = Macro Index
  df_data['macro_index'] = df_data[['contraction', 'expansion', 'expansion_atr','recovery', 'slowdown']].max(axis=1)
  df_data['date_str'] = df_data['date'].str.slice(0, 10)
  df_data['date'] = pd.to_datetime(df_data['date_str'])    
  df_capriole = df_data[['date','macro_index']]
  #print('DF capriole_btc: ', df_capriole.tail(), df_capriole.info())

  # Load Heater
  df_data = get_capriole_metric(1) # 1 = Heater
  df_data['date_str'] = df_data['date'].str.slice(0, 10)
  df_data['date'] = pd.to_datetime(df_data['date_str'])     
  #print('DF Data: ', df_data.tail(), df_data.info())    
  df_capriole = pd.merge(df_capriole, df_data[['date','price_usd_close','heater','heat_perps','heat_futures','heat_options','oi_pct_mcap']],on='date', how="outer")

  # Dynamic Range NVT
  df_data = get_capriole_metric(2) # 2 = Dynamic Range NVT
  df_data['date_str'] = df_data['date'].str.slice(0, 10)
  df_data['date'] = pd.to_datetime(df_data['date_str'])     
  #print('DF Data: ', df_data.tail(), df_data.info()) 
  df_capriole = pd.merge(df_capriole, df_data[['date','nvts','nvts_low','nvts_high']],on='date', how="outer")

  # Bitcoin Production Cost
  df_data = get_capriole_metric(3) # 3 = Bitcoin Production Cost
  df_data['date_str'] = df_data['date'].str.slice(0, 10)
  df_data['date'] = pd.to_datetime(df_data['date_str'])     
  #print('DF Data: ', df_data.tail(), df_data.info()) 
  df_capriole = pd.merge(df_capriole, df_data[['date','Electrical Cost','Production Cost','Miner Price', 'margin']],on='date', how="outer")

  # Hash Ribbons
  df_data = get_capriole_metric(4) # 4 = Hash Ribbons
  df_data['date_str'] = df_data['date'].str.slice(0, 10)
  df_data['date'] = pd.to_datetime(df_data['date_str'])     
  #print('DF Data: ', df_data.tail(), df_data.info()) 
  df_capriole = pd.merge(df_capriole, df_data[['date','hr','hr30','hr60','miner_capitulation','hash_ribbon_buy','capitulation','buy']],on='date', how="outer")  
  
  df_capriole['date'] = df_capriole['date'].dt.strftime('%Y-%m-%d')
  df_capriole = df_capriole.rename(columns={'price_usd_close' : 'close', 'Electrical Cost': 'electrical_cost', 'Production Cost': 'production_cost', 'Miner Price': 'miner_price', 'margin': 'production_margin', 'capitulation': 'hr_capitulation', 'buy': 'hr_buy'})  
  df_capriole.fillna(method="ffill", inplace=True)
  print('DF capriole_btc: ', df_capriole, df_capriole.info())  

  # add imported data to database
  database.store_metric_table(df_capriole, 'capriole_btc')
  
  return
  
def get_capriole_metric(metric_nb):
  # Tags of capriole_btc data query (via Chrome Development Tool)
  capriole_metrics = {
      'name': ['capriole_btc Bitcoin Macro Index', 'Bitcoin Heater', 'Dynamic Range NVT', 'Bitcoin Production Cost', 'Hash Ribbons'],
      'file_name': ['YocJ0.csv', 'wQkNA.csv', 'l2ZFL.csv', 'S77Ci.csv', 'dSp0L.csv'],
      'v_nb': ['1739197320000', '1725356280000', '1725356280000', '1725356280000', '1725356280000']}
    #'name': ['capriole_btc Bitcoin Macro Index', 'Bitcoin Heater', 'Dynamic Range NVT', 'Bitcoin Production Cost', 'Hash Ribbons'],
    #'file_name': ['4AKp4.csv', 'wQkNA.csv', 'l2ZFL.csv', 'S77Ci.csv', 'dSp0L.csv'],
    #'v_nb': ['1725356340000', '1725356280000', '1725356280000', '1725356280000', '1725356280000']}
  df_capriole_metrics = pd.DataFrame(capriole_metrics)
  row = df_capriole_metrics.iloc[metric_nb]
  file_name = row['file_name']
  v_nb = row['v_nb']
  
  capriole_data = StringIO(query_capriole_data(file_name, v_nb))
  print(capriole_data)   
  # Convert the data to a pandas DataFrame
  df_data = pd.read_csv(capriole_data, sep=',') 
  try:
    df_data.rename(columns ={'Unnamed: 0': 'date'}, inplace = True)
  except:
    pass
  print('capriole_btc Metric: ', metric_nb, df_data)    
  return df_data


def query_capriole_data(file_name, v_nb):
  conn = http.client.HTTPSConnection("static.dwcdn.net")

  payload = ""

  headers = {
      'accept': "*/*",
      'accept-language': "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
      'origin': "https://datawrapper.dwcdn.net",
      'priority': "u=1, i",
      'referer': "https://datawrapper.dwcdn.net/",
      '^sec-ch-ua': "^\^Chromium^^;v=^\^128^^, ^\^Not"}

  query = "/data/" + file_name + "?v=" + v_nb +"/"
  #print(query)

  conn.request("GET", query, payload, headers)
  #conn.request("GET", "/data/wQkNA.csv?v=1725361740000", payload, headers)    

  res = conn.getresponse()
  data = res.read().decode("utf-8")
  return data '''

# ----------------------------- Augmento -------------------------------------------------------------

def auto_import_augmento_data():
  # Load Macro Index
  df_augmento = query_augmento_data() # Load Augmento data
  print('Augmento: ', df_augmento, df_augmento.info())    

  # add imported data to database
  database.store_metric_table(df_augmento, 'augmento')
  return 


def query_augmento_data():
  conn = http.client.HTTPSConnection("bitcoin-sentiment.augmento.ai")

  payload = ""

  headers = {
    '^accept': "application/json^",
    '^accept-language': "en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7^",
    '^content-type': "application/json^",
    '^cookie': "_gid=GA1.2.2033930949.1729332980; _gat_gtag_UA_61144449_1=1; _ga_4NTC2B7HVV=GS1.1.1729337601.8.1.1729337602.0.0.0; _ga=GA1.1.741727049.1707981225^",
    '^priority': "u=1, i^",
    '^referer': "https://bitcoin-sentiment.augmento.ai/graph/^",
    '^sec-ch-ua': "^\^Chromium^^;v=^\^130^^, ^\^Google",
    '^sec-ch-ua-mobile': "?0^",
    '^sec-ch-ua-platform': "^\^Windows^^^",
    '^sec-fetch-dest': "empty^",
    '^sec-fetch-mode': "cors^",
    '^sec-fetch-site': "same-origin^",
    '^user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36^",
    '^x-csrftoken': "undefined^"
      }

  query = "/graph/_dash-layout"
  #print(query)

  conn.request("GET", query, payload, headers)
  #conn.request("GET", "/data/wQkNA.csv?v=1725361740000", payload, headers)    

  res = conn.getresponse()
  data = res.read().decode("utf-8")
  #print(data)
  data_json = json.loads(data)
  data_path = data_json['props']['children'][0]['props']['children'][0]['props']['figure']['data']

  # 0 = XBTUSD, 1 = Bitcointalk, 2 = Reddit, 3 = Twitter
  df = pd.DataFrame({'date': data_path[1]['x'], 'bitcointalk': data_path[1]['y']})  
  df_reddit = pd.DataFrame({'date': data_path[2]['x'], 'reddit': data_path[2]['y']})  
  df = pd.merge(df, df_reddit[['date','reddit']],on='date', how="outer")    
  df_twitter = pd.DataFrame({'date': data_path[3]['x'], 'twitter': data_path[3]['y']})  
  df = pd.merge(df, df_twitter[['date','twitter']],on='date', how="outer")
  df['date'] = pd.to_datetime(df['date'])
  df['date'] = df['date'].dt.strftime('%Y-%m-%d')

  df['augmento'] = df[['bitcointalk','reddit','twitter']].mean(axis=1) * 100 

  return df


# ----------------------------- Bitcoinmagazine Pro -------------------------------------------------------------

'''
def query_bmp_data_full(metric):
    conn = http.client.HTTPSConnection("www.bitcoinmagazinepro.com")
       
    # Individual variables for the payload components
    pl_output = "chart.figure"
    pl_outputs_id = "chart"
    pl_outputs_property = "figure"
    
    pl_inputs_url_id = "url"
    pl_inputs_url_property = "pathname"
    pl_inputs_url_value = "" #/charts/relative-unrealized-profit--loss/"
    pl_inputs_display_id = "display"
    pl_inputs_display_property = "children"
    pl_inputs_display_value = "xl 1251px"
    
    pl_changedPropIds = ["url.pathname", "display.children"]       
    
  
    if metric.lower() == 'liquidity':
        metric_header = "global-liquidity/"
        metric_query = "global_liquidity"
        pl_output = "..chart.figure...recent-data-button-desktop.hidden...recent-data-button-mobile.hidden.."
        pl_outputs_id_2 = "recent-data-button-desktop"
        pl_outputs_property_2 = "hidden"   
        pl_outputs_id_3 = "recent-data-button-mobile"
        pl_outputs_property_3 = "hidden"         
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": [{"id": pl_outputs_id, "property": pl_outputs_property}, {"id": pl_outputs_id_2, "property": pl_outputs_property_2}, {"id": pl_outputs_id_3, "property": pl_outputs_property_3}], 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],
                              "changedPropIds": pl_changedPropIds}) 
        

    elif metric.lower() == 'nupl':
        metric_header = "relative-unrealized-profit--loss/"
        metric_query = "unrealised_profit_loss"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 


    elif metric.lower() == 'mvrv':
        metric_header = "mvrv-zscore/"
        metric_query = "mvrv_zscore"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
 
    elif metric.lower() == 'reserve_risk':
        metric_header = "reserve-risk/"
        metric_query = "reserve_risk"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 

        
    elif metric.lower() == 'bitcoin_sentiment': 
        # Active Adress Sentiment Indicator = Bitcoin Sentiment
        metric_header = "active-address-sentiment-indicator/"
        metric_query = "bitcoin_sentiment"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 


    elif metric.lower() == 'addresses_in_profit': 
        metric_header = "percent-addresses-in-profit/"
        metric_query = "addresses_in_profit"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 

        
    elif metric.lower() == 'rhodl_ratio': 
        metric_header = "rhodl-ratio/"
        metric_query = "rhodl_ratio"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 


    elif metric.lower() == 'puell_multiple': 
        metric_header = "puell-multiple/"
        metric_query = "puell_multiple"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
 

    elif metric.lower() == 'miner_fee_pct': 
        # Percentage of Miner Fees as % of overall Miner Revenues
        metric_header = "bitcoin-miner-revenue-fees-vs-rewards/"
        metric_query = "miner_revenue_fees_pct"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 


    elif metric.lower() == 'sopr': 
        metric_header = "sopr-spent-output-profit-ratio/"
        metric_query = "sopr"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
        
    elif metric.lower() == 'funding_rate': 
        metric_header = "bitcoin-funding-rates"
        metric_query = "funding_rates"
        
        # Output string
        pl_output = "..chart.figure...exchange.options...resolution.disabled...resolution.value...exchange.value.."

        # Outputs list components
        pl_outputs_list = [{"id": "chart", "property": "figure"}, {"id": "exchange", "property": "options"}, 
                           {"id": "resolution", "property": "disabled"}, {"id": "resolution", "property": "value"},
                           {"id": "exchange", "property": "value"}]

        # Inputs components
        pl_inputs_currency_id = "currency"
        pl_inputs_currency_property = "value"
        pl_inputs_currency_value = "funding_rate_usd"

        pl_inputs_exchange_id = "exchange"
        pl_inputs_exchange_property = "value"
        pl_inputs_exchange_value = "average"

        pl_inputs_resolution_id = "resolution"
        pl_inputs_resolution_property = "value"
        pl_inputs_resolution_value = "1h"
        
        pl_changedPropIds = ["display.children"]
    
        pl_inputs_url_value = "/charts/" + metric_header  
        payload = json.dumps({"output": pl_output, "outputs": pl_outputs_list, 
                              "inputs": [{"id": pl_inputs_currency_id, "property": pl_inputs_currency_property, "value": pl_inputs_currency_value},
                                         {"id": pl_inputs_exchange_id, "property": pl_inputs_exchange_property, "value": pl_inputs_exchange_value},
                                         {"id": pl_inputs_resolution_id, "property": pl_inputs_resolution_property, "value": pl_inputs_resolution_value},
                                         {"id": pl_inputs_display_id, "property": pl_inputs_display_property, "value": pl_inputs_display_value}],
                              "changedPropIds": pl_changedPropIds})

        
    elif metric.lower() == 'fear_greed': 
        metric_header = "bitcoin-fear-and-greed-index"
        metric_query = "fear_and_greed"

        pl_output = "..chart.figure...indicator.figure...last_update.children...now.children...now.style.."
        pl_outputs_id_2 = "indicator"
        pl_outputs_property_2 = "figure"   
        pl_outputs_id_3 = "last_update"
        pl_outputs_property_3 = "children"     
        pl_outputs_id_4 = "now"
        pl_outputs_property_4 = "children"   
        pl_outputs_id_5 = "now"
        pl_outputs_property_5 = "style"          
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": [{"id": pl_outputs_id, "property": pl_outputs_property}, 
                                          {"id": pl_outputs_id_2, "property": pl_outputs_property_2}, 
                                          {"id": pl_outputs_id_3, "property": pl_outputs_property_3}, 
                                          {"id": pl_outputs_id_4, "property": pl_outputs_property_4}, 
                                          {"id": pl_outputs_id_5, "property": pl_outputs_property_5}], 
                                "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                           {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
    
        
    elif metric.lower() == 'bitcoin_cycle_master': 
        metric_header = "bitcoin-cycle-master"
        metric_query = "bitcoin_cycle_master"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 

        
    elif metric.lower() == 'onchain_prediction': 
        metric_header = "bitcoin-price-prediction"
        metric_query = "price_prediction"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
         
        
    elif metric.lower() == 'realized_price_sth': 
        metric_header = "short-term-holder-realized-price"
        metric_query = "realized_price_sth"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
 
        
    elif metric.lower() == 'realized_price_lth': 
        metric_header = "long-term-holder-realized-price"
        metric_query = "realized_price_lth"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, 
                                          "value": pl_inputs_url_value}, {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
         
        
    elif metric.lower() == 'sth_supply': 
        metric_header = "short-term-holder-supply"
        metric_query = "sth_supply"
        
        pl_inputs_url_value = "/charts/" + metric_header           
        pl_inputs_id_2 = "period"
        pl_inputs_property_2 = "value"          
        pl_inputs_value_2 = "all"      
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_id_2, "property": pl_inputs_property_2, "value": pl_inputs_value_2}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],
                              "changedPropIds": pl_changedPropIds}) 
        
       
    elif metric.lower() == 'vdd_multiple': 
        metric_header = "value-days-destroyed-multiple"
        metric_query = "vdd_multiple"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
              
    elif metric.lower() == 'nvt': 
        metric_header = "advanced-nvt-signal"
        metric_query = "nvts"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
              
    elif metric.lower() == 'pi_cycle_oscillator': 
        metric_header = "pi-cycle-top-bottom-indicator"
        metric_query = "pi_cycle_top_bottom_indicator"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
               
    elif metric.lower() == 'sth_mvrv': 
        metric_header = "short-term-holder-mvrv"
        metric_query = "mvrv_sth"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
              
    elif metric.lower() == 'lth_mvrv': 
        metric_header = "long-term-holder-mvrv"
        metric_query = "mvrv_lth"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 

      
    elif metric.lower() == 'financial_stress': 
        metric_header = "financial-stress-index-vs-btc"
        metric_query = "financial_stress_index_vs_btc"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
              
    elif metric.lower() == 'high_yield_credit': 
        metric_header = "bitcoin-cycles-vs-high-yield-credit-cycles"
        metric_query = "high_yield_credit_vs_btc"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
  
    elif metric.lower() == 'm2_yoy_change': 
        metric_header = "m2-vs-btc-yoy-change"
        metric_query = "m2_vs_btc_yoy"
        pl_inputs_url_value = "/charts/" + metric_header         
        payload = json.dumps({"output": pl_output, 
                              "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],"changedPropIds": pl_changedPropIds}) 
        
  
    elif metric.lower() == 'yield_spread': 
        metric_header = "10yr-2yr-yield-spread"
        metric_query = "us_2y_10y_spread"
        
        pl_inputs_url_value = "/charts/" + metric_header           
        pl_inputs_id_2 = "period"
        pl_inputs_property_2 = "value"          
        pl_inputs_value_2 = "5y"      
        payload = json.dumps({"output": pl_output, "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
                              "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                                         {"id": pl_inputs_id_2, "property": pl_inputs_property_2, "value": pl_inputs_value_2}, 
                                         {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],
                              "changedPropIds": pl_changedPropIds})             
                   
        
    elif metric.lower() == 'btc_etf_flows': 
        metric_header = "bitcoin-etf-daily-flows-usd/"
        metric_query = "etf_daily_flows_usd"
        
        pl_inputs_url_value = "/bitcoin-portfolio/" + metric_header +"/"
        pl_changedPropIds = ["url.pathname"]  
        
        pl_inputs_id_2 = "etf"
        pl_inputs_property_2 = "value"          
        pl_inputs_value_2 = "total"  
        
        payload = json.dumps({
            "output": pl_output, 
            "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
            "inputs": [
                {"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                {"id": pl_inputs_id_2, "property": pl_inputs_property_2, "value": pl_inputs_value_2}, 
                {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],
            "changedPropIds": pl_changedPropIds })                
         
    
    elif metric.lower() == 'cycle_capital_flows': 
        metric_header = "cycle-capital-flows"
        metric_query = "cycle_capital_flows"
        pl_inputs_url_value = "/bitcoin-portfolio/" + metric_header + "/"    
        
        payload = json.dumps({
            "output": pl_output, 
            "outputs": {"id": pl_outputs_id, "property": pl_outputs_property}, 
            "inputs": [{"id": pl_inputs_url_id, "property": pl_inputs_url_property, "value": pl_inputs_url_value}, 
                       {"id": pl_inputs_display_id,"property": pl_inputs_display_property, "value": pl_inputs_display_value}],
            "changedPropIds": pl_changedPropIds})      
        
    else:
        return "Wrong metric"

    #cookie = "_ga=GA1.1.1902202101.1727269776; _ga_LFVGV1TW01=GS1.1.1729588439.4.1.1729588455.0.0.0; csrftoken=nV8l9ipmx3ByGYjY6ho9FiTxTnHvN0yY; sessionid=i6c4w39nph536dx6nahpd7fb587qgruj; sessionid=i6c4w39nph536dx6nahpd7fb587qgruj"  
    #cookie = "_ga=GA1.1.1902202101.1727269776; csrftoken=nV8l9ipmx3ByGYjY6ho9FiTxTnHvN0yY; sessionid=i6c4w39nph536dx6nahpd7fb587qgruj; _ga_LFVGV1TW01=GS1.1.1735310088.5.1.1735310126.0.0.887455580; csrftoken=nV8l9ipmx3ByGYjY6ho9FiTxTnHvN0yY; sessionid=i6c4w39nph536dx6nahpd7fb587qgruj" 
    cookie = "_ga=GA1.1.1902202101.1727269776; csrftoken=dqqRKFoyFm4XVpIc2vtW8hSWZCTfqVOj; sessionid=64spyjq602vii1v0hl4674ho7jcpktf8; _ga_LFVGV1TW01=GS2.1.s1757082727^$o13^$g1^$t1757082780^$j7^$l0^$h2144554367; csrftoken=nV8l9ipmx3ByGYjY6ho9FiTxTnHvN0yY; sessionid=64spyjq602vii1v0hl4674ho7jcpktf8"

    headers = {
        'accept': 'application/json',
        'accept-language': 'en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7',
        'content-type': "application/json",
        'origin': "https://www.bitcoinmagazinepro.com",
        'priority': "u=1, i",
        'referer': "https://www.bitcoinmagazinepro.com/charts/"+metric_header,
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': "?0",
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': "empty",
        'sec-fetch-mode': "cors",
        'sec-fetch-site': "same-origin",
        'Cookie': '_ga=GA1.1.1902202101.1727269776; csrftoken=dqqRKFoyFm4XVpIc2vtW8hSWZCTfqVOj; sessionid=64spyjq602vii1v0hl4674ho7jcpktf8; _ga_LFVGV1TW01=GS2.1.s1757082727^$o13^$g1^$t1757082780^$j7^$l0^$h2144554367; csrftoken=nV8l9ipmx3ByGYjY6ho9FiTxTnHvN0yY; sessionid=64spyjq602vii1v0hl4674ho7jcpktf8'
        }


    """    'cookie': cookie,
        'accept': "application/json",
        'accept-language': "en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7",
        'content-type': "application/json",
        'origin': "https://www.bitcoinmagazinepro.com",
        'priority': "u=1, i",
        'referer': "https://www.bitcoinmagazinepro.com/charts/"+metric_header,
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': "?0",
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': "empty",
        'sec-fetch-mode': "cors",
        'sec-fetch-site': "same-origin",
        'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        'x-csrftoken': "undefined"} """
    
    query = "/django_plotly_dash/app/" + metric_query + "/_dash-update-component"
    
    conn.request("POST", query, payload, headers)

    res = conn.getresponse()
    data = res.read().decode("utf-8")
    print(metric, 'data length: ', len(data))
    print(data[:500])  # print the first 500 characters of the response for debugging
    return data

def query_bmp_metric(metric):
    # Works for following metrics:
    # Liquidity, STH_Supply, NUPL, Reserve-Risk, Bitcoin-Sentiment, Adresses in Profit, rhodl_ratio, puell_multiple,
    # realized_price_sth, realized_price_lth, miner fee pct, vdd_multiple, Fear&Gread
    
    # Works not for:
    # MVRV, SOPR, Funding_rate, bitcoin_cycle_master, onchain_prediction, 
    
    data = query_bmp_data_full(metric)
    try:
        data_json = json.loads(data)
    except:
        print('Error loading JSON data for metric: ', metric)
        return pd.DataFrame({})
    try:
        data_date = data_json['response']['chart']['figure']['data'][0]['x']
        data_price = data_json['response']['chart']['figure']['data'][0]['y']        
        data_metric = data_json['response']['chart']['figure']['data'][0]['customdata']
    except:
        data_metric = []

    # Create a DataFrame with time and price
    df_time = pd.DataFrame({'time_str': data_date})
    df_time['time'] = pd.to_datetime(df_time['time_str'], format = 'ISO8601') 
    
    df_metric = pd.DataFrame({'price': data_price, metric: data_metric})
    df_metric[metric] = df_metric[metric].apply(lambda x: x[0])

    # Merge the dataframes
    df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric[['price', metric]]], axis=1)
    
    if (metric == 'sth_mvrv'):
        name_ratio = metric + '_ratio'
        df[name_ratio] = df['price'] / df[metric] 
    elif (metric == 'yield_spread'):
        # swap column names
        df.rename(columns = {"price": "price_bu", metric : "price"}, inplace = True) 
        df.rename(columns = {"price_bu" : metric}, inplace = True) 

    return df

   
def query_bmp_complex(metric):

    data = query_bmp_data_full(metric) 
    #print(data)
    data_json = json.loads(data)
    print(metric, data_json)
    
    if metric.lower() == 'fear_greed':
        data_complex = data_json['response']['chart']['figure']['data'][1]['customdata']
        columns = ['time_str', 'price', metric, 'regime', 'signal']
    
    elif metric.lower() == 'funding_rate':
        data_complex = data_json['response']['chart']['figure']['data'][0]['customdata']
        columns = ['nb','time_str', 'regime', metric, 'funding2', 'time2', 'price']
        
    else:
        return 'Wrong metric!'

    
    df_complex = pd.DataFrame(data_complex, columns=columns)
    #display(df_complex)
    df_complex['time'] = pd.to_datetime(df_complex['time_str'], format = 'ISO8601') 
    
    return df_complex

def query_bmp_sopr():
    # Works for following metrics: SOPR
    metric = 'sopr'
    data = query_bmp_data_full(metric)
    data_json = json.loads(data)
    data_date = data_json['response']['chart']['figure']['data'][1]['x']
    data_metric = data_json['response']['chart']['figure']['data'][1]['y']

    # Create a DataFrame with time and price
    df_time = pd.DataFrame({'time_str': data_date})
    df_time['time'] = pd.to_datetime(df_time['time_str'], format = 'ISO8601') 
    
    df_metric = pd.DataFrame({metric: data_metric})

    # Merge the dataframes
    df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric[[metric]]], axis=1)

    return df

def query_bmp_mvrv():
    # Works for following metrics: MVRV
    metric = 'mvrv'
    data = query_bmp_data_full(metric)
    data_json = json.loads(data)
    data_date = data_json['response']['chart']['figure']['data'][7]['x']
    data_metric = data_json['response']['chart']['figure']['data'][7]['customdata']

    # Create a DataFrame with time and price
    df_time = pd.DataFrame({'time_str': data_date})
    df_time['time'] = pd.to_datetime(df_time['time_str'], format = 'ISO8601') 
    
    df_metric = pd.DataFrame({metric: data_metric})
    df_metric[metric] = df_metric[metric].apply(lambda x: x[0])

    # Merge the dataframes
    df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric[[metric]]], axis=1)

    return df

def query_bmp_lth_mvrv():
    # Works for following metrics: Longterm Holder MVRV
    metric = 'lth_mvrv'
    data = query_bmp_data_full(metric)
    data_json = json.loads(data)
    data_date = data_json['response']['chart']['figure']['data'][4]['x']
    data_metric = data_json['response']['chart']['figure']['data'][4]['customdata']

    # Create a DataFrame with time and price
    df_time = pd.DataFrame({'time_str': data_date})
    df_time['time'] = pd.to_datetime(df_time['time_str'], format = 'ISO8601') 
    
    df_metric = pd.DataFrame({metric: data_metric})
    df_metric[metric] = df_metric[metric].apply(lambda x: x[0])

    # Merge the dataframes
    df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric[[metric]]], axis=1)

    return df


def query_bmp_cycle_capital_flows():
    # Works for following metrics: cycle_capital_flows
    metric = 'cycle_capital_flows'
    data = query_bmp_data_full(metric)
    data_json = json.loads(data)

    data_price = data_json['response']['chart']['figure']['data'][0]['customdata']    
    data_date = data_json['response']['chart']['figure']['data'][1]['x']
    data_short_term = data_json['response']['chart']['figure']['data'][1]['y']
    data_long_term = data_json['response']['chart']['figure']['data'][2]['y']    
    
    # Create a DataFrame with time and price
    df_time = pd.DataFrame({'time_str': data_date})
    df_time['time'] = pd.to_datetime(df_time['time_str'], format = 'ISO8601') 
    
    df_metric = pd.DataFrame({'short_term_cycle': data_short_term, 'long_term_cycle': data_long_term})
    df_metric[metric] = (0.5 + (df_metric['short_term_cycle'] - df_metric['long_term_cycle']) / 2) * 100 # Merge into one metric

    # Merge the dataframes
    df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_metric], axis=1)

    return df

"""
@anvil.server.background_task
@anvil.server.callable
def auto_import_bmp_data():
    #list_metrics_simple = ['liquidity', 'reserve_risk', 'nvt', 'bitcoin_sentiment', 'addresses_in_profit', 'rhodl_ratio',   
    list_metrics_simple = ['reserve_risk', 'nvt', 'bitcoin_sentiment', 'addresses_in_profit', 'rhodl_ratio',   
                        'miner_fee_pct', 'realized_price_sth', 'realized_price_sth', 'sth_supply','vdd_multiple','sth_mvrv',
                        'financial_stress', 'high_yield_credit', 'm2_yoy_change', 'yield_spread', 'btc_etf_flows']
    list_metrics_complex = ['fear_greed'] #, 'funding_rate']

    df = query_bmp_metric('nupl') # create DF with dates and NUPL metric
    print('BMP specific - NUPL: ',df, df.info())  

    # Add specific metrices
    df_sopr = query_bmp_sopr() # query SOPR data
    df = pd.merge(df, df_sopr[['time','sopr']],on='time', how="outer")
    print('BMP specific - SOPR: ',df, df.info())     

    df_mvrv = query_bmp_mvrv() # query MVRV-Z data
    df = pd.merge(df, df_mvrv[['time','mvrv']],on='time', how="outer")
    print('BMP specific - MVRV: ',df, df.info())    

    print('Metric to run: lth_mvrv')
    df_lth_mvrv = query_bmp_lth_mvrv() # query Longterm Holder MVRV data
    df = pd.merge(df, df_lth_mvrv[['time','lth_mvrv']],on='time', how="outer")

    print('Metric to run: cycle capital flows')
    df_cycle_capital = query_bmp_cycle_capital_flows() # query Shortterm & Longterm cycle flow & calc integrated Osciillator
    df = pd.merge(df, df_cycle_capital[['time','short_term_cycle','long_term_cycle','cycle_capital_flows']],on='time', how="outer")
    print('BMP specific - Cycle Capital: ',df, df.info())
  
    # Add simple metrices
    for metric in list_metrics_simple:
        print('Metric to run: ',metric)
        df_simple = query_bmp_metric(metric)
        df = pd.merge(df, df_simple[['time',metric]],on='time', how="outer")  
    print('BMP simple: ',df, df.info())          


    # Add complex metrices
    for metric in list_metrics_complex:
        print('Metric to run: ',metric)        
        df_complex = query_bmp_complex(metric)
        df = pd.merge(df, df_complex[['time',metric]],on='time', how="outer")
    #print('BMP complex: ',df, df.info())          

    df.drop('realized_price_sth_y', axis=1, inplace=True)
    df = df.rename(columns={'price': 'close', 'time': 'date', 'realized_price_sth_x': 'realized_price_sth'})
      
    df = df.fillna(method='ffill')
    print('BMP: ', df.info())

    # add imported data to database
    database.store_metric_table(df,'bmp_full')

    return """
'''

# ----------------------------- BTC ETF (https://members.delphidigital.io/) -------------------------------------------------------------
@anvil.server.background_task
@anvil.server.callable
def auto_import_etf_data():
  # Load ETF Data
  df_etf = query_etf_data() # Load ETF data
  print('ETF Flow: ', df_etf, df_etf.info())    

  # add imported data to database
  database.store_metric_table(df_etf, 'etf')
  return


def get_etf_data():
    conn = http.client.HTTPSConnection("members.delphidigital.io")
    
    payload = "{\"output\":\"..default_loader_btc_flows_beta.style...output_btc_flows_beta.children..\",\"outputs\":[{\"id\":\"default_loader_btc_flows_beta\",\"property\":\"style\"},{\"id\":\"output_btc_flows_beta\",\"property\":\"children\"}],\"inputs\":[{\"id\":\"location_btc_flows_beta\",\"property\":\"pathname\",\"value\":\"/scope/portal/BTC_ETF_Flows\"}],\"changedPropIds\":[]}"

    headers = {
        'accept': 'application/json',
        'accept-language': 'en,de-DE;q=0.9,de;q=0.8,en-US;q=0.7',
        'content-type': 'application/json',
        'cookie': 'rl_anonymous_id=RS_ENC_v3_IjE2ZTkyZjYwLWIzZGUtNGE4Ny1iMDVmLTU3YmE1YjEzMmU0MSI%3D; rl_page_init_referrer=RS_ENC_v3_IiRkaXJlY3Qi; stream_unread_latest=1731978231000; stream_latest=1727074833000; rl_session=RS_ENC_v3_eyJpZCI6MTczMjUyODg4NTM5OCwiZXhwaXJlc0F0IjoxNzMyNTMwNjg1Mzk4LCJ0aW1lb3V0IjoxODAwMDAwLCJhdXRvVHJhY2siOnRydWV9; _iub_cs-78158090=%7B%22timestamp%22%3A%222024-10-22T09%3A05%3A46.390Z%22%2C%22version%22%3A%221.67.1%22%2C%22purposes%22%3A%7B%221%22%3Atrue%2C%224%22%3Afalse%7D%2C%22id%22%3A78158090%2C%22cons%22%3A%7B%22rand%22%3A%224a369c%22%7D%7D; usprivacy=%7B%22uspString%22%3A%221YY-%22%2C%22firstAcknowledgeDate%22%3A%222024-10-22T09%3A05%3A40.849Z%22%2C%22optOutDate%22%3A%222024-10-22T09%3A05%3A46.390Z%22%7D; ph_phc_kyQTybnQwt6dqvv99ZLl52YFoIcG2WrnUQ0pLIWY2Cv_posthog=%7B%22distinct_id%22%3A%22019362c5-1440-7400-8bc0-382597f16802%22%2C%22%24sesid%22%3A%5B1732528903907%2C%22019362c5-1481-751f-9ce4-ca2592719af3%22%2C1732528903297%5D%7D',
        'origin': 'https://members.delphidigital.io',
        'priority': 'u=1, i',
        'referer': 'https://members.delphidigital.io/scope/portal/BTC_ETF_Flows',
        'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'x-csrftoken': 'undefined'
    }

    query = "/scope/_dash-update-component"
    
    conn.request("POST", query, payload, headers)

    res = conn.getresponse()
    data = res.read().decode("utf-8")
    return data

def query_etf_data():
    data = get_etf_data()
    data_json = json.loads(data)
    data_date = data_json['response']['output_btc_flows_beta']['children'][0]['props']['children'][1]['props']['children'][0]['props']['figure']['data'][0]['x']
    data_price = data_json['response']['output_btc_flows_beta']['children'][0]['props']['children'][1]['props']['children'][0]['props']['figure']['data'][0]['y']
    data_flow = data_json['response']['output_btc_flows_beta']['children'][0]['props']['children'][1]['props']['children'][0]['props']['figure']['data'][1]['y']

    #display(data_price, len(data_price))
    #display(data_flow, len(data_flow))


    # Create a DataFrame with time and flow
    df_time = pd.DataFrame({'time_str': data_date})
    df_time['date'] = pd.to_datetime(df_time['time_str']) 

    df_price = pd.DataFrame({'close': data_price})
    df_flow = pd.DataFrame({'flow': data_flow})
    #df_metric[metric] = df_metric[metric].apply(lambda x: x[0])

    # Merge the dataframes
    #df = pd.concat([df_time['time'].iloc[:len(df_metric)], df_price['price'], df_flow['flow']], axis=1)
    df = pd.concat([df_time['date'], df_flow, df_price], axis=1)
    #display(df.head(20), df.info())

    # Generate a complete date range
    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

    # Identify missing dates
    missing_dates = full_date_range[~full_date_range.isin(df['date'])]

    # Create new rows for the identified missing dates and fill them with previous data
    fill_data = []
    for date in missing_dates:
        #previous_date = df[df['time'] < date].iloc[-1]
        #fill_data.append({'time': date, 'flow': None, 'price': previous_date['price']})
        fill_data.append({'date': date, 'flow': None, 'close': None})    

    missing_df = pd.DataFrame(fill_data)

    # Concatenate original DataFrame with the new missing dates DataFrame
    final_df = pd.concat([df, missing_df]).sort_values(by='date').reset_index(drop=True)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    #display(final_df.head(20), final_df.info())
    
    return final_df

# -----------------------------------------------------------------------------------------

@anvil.server.background_task
@anvil.server.callable
def auto_import_hyblock_data(start_date = "2020-01-01", end_date = "2025-01-01"):
    df_hyblock_1h = database.read_table_date_range_cloud('hyblock_1h', start_date, end_date) 
    print('Hyblock 1h: ', df_hyblock_1h, df_hyblock_1h.info())

    df_hyblock = resample_df_data(df_hyblock_1h, tf='1D')
    print('Hyblock: ', df_hyblock, df_hyblock.info())    

    # add imported data to database
    database.store_metric_table(df_hyblock, 'hyblock_1h')   

    return




@anvil.server.callable
def store_bier_result(calc_json, signal_strategy, category, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight, scores_list):  
  df = pd.read_json(calc_json, orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 
  
  df = matrix_strategy.calc_multi_strategy(df, peak_shift, scores_list, signal_strategy)    
  #print('Price Graph: ', df.tail(10), df.info())
  df = df.replace([np.inf, -np.inf], 0)
  os.makedirs('data', exist_ok=True)
  df.to_csv('data/store_bier.csv')
  df_bier = df[['date', 'close', 'invested', 'range']].fillna(0)
  try:
    df_bier.rename(columns ={'invested': 'bier_invested', 'range': 'bier_range'}, inplace = True)
  except:
    pass
  print('BIER DB: ', df_bier.tail(10))  
  
  # add imported data to database
  database.store_metric_table(df_bier, 'bier') 

  return



def resample_df_data(df, tf='1D'): 

  try:
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M') # mixed
  except:
    print('Error in date converting')   
  df = df.set_index('time')
  columns_dict = {'volume_delta': 'sum', 'whale_retail': 'last', 'user_bot_ratio': 'last', 'usdt_premium': 'last', 'bid_ask_ratio': 'last', 'top_traders_long': 'last', 'market_order_size': 'sum', 'market_order_count': 'sum', 'limit_order_count': 'sum', 'funding_rate': 'last','fear_greed': 'last', 'bid_ask_delta' : 'last', 'long_liquidations' : 'sum', 'short_liquidations' : 'sum', 'oi_delta' : 'sum', 'bvol' : 'last', 'dvol' : 'last'} #, 'top_traders_leverage_delta': 'last'      
  print('Columns dict: ', columns_dict)

  df_resample = df.resample(tf).agg(columns_dict)

  df_resample.reset_index(inplace=True)
  df_resample['date'] = df_resample['time'].dt.strftime('%Y-%m-%d')
  df_resample = df_resample.drop('time', axis = 1)

  return df_resample
