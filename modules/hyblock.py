

import requests
import json
import pandas as pd
from datetime import datetime, timedelta, timezone

#import inspect 
base_url = "https://api1.hyblockcapital.com/v1"

client_id = "2bugucpn21oodiirg40qdc8r03"
client_secret = "vpotfcnelcq0i2cn313c62qmng2pkm290qr06bbkvgvjeglhjkk"
api_key = "IEdaPRm7ql7iElBvaDaR4a5JxWwEhnPv1HimDUUG"

grant_type = "client_credentials" #this is constant
amazon_auth_url = 'https://auth-api.hyblockcapital.com/oauth2/token'
auth_data = {
     "grant_type": grant_type,
     "client_id": client_id,
     "client_secret": client_secret,}

def iso_to_unix(iso_time_str):  
    iso_format = '%Y-%m-%d %H:%M' # Define the format of the input ISO time string
    dt = datetime.strptime(iso_time_str, iso_format) # Parse the ISO time string into a datetime object
    unix_timestamp = int(dt.timestamp()) # Convert the datetime object to Unix timestamp
    return unix_timestamp

def update_access_token():
       auth_response = requests.post(
         amazon_auth_url, data=auth_data, headers={'Content-Type': "application/x-www-form-urlencoded"})
       return auth_response.json()

def get_data(path, base_url, query_params):
       auth_response_json = update_access_token()
       auth_token_header = {
         "Authorization": "Bearer %s" % auth_response_json["access_token"],
         "x-api-key": api_key }

       url = base_url + path
       response = requests.get(url, params=query_params, headers=auth_token_header)
       #print(response.json())
       return response.json()

def get_data_hyblock(data_type, timeframe='5m', asset='BTC', exchange='Binance', start_time='2023-06-26 00:00', limit=20):
    #print('Start Time func: ', start_time)
    start_time_unix = iso_to_unix(start_time)  
    
    if data_type == 'volume_delta':
        orderbook_path = '/volumeDelta'                  
    else:
        orderbook_path = '/klines'        

    #print('Path: ', orderbook_path)
    query = {"timeframe": timeframe, "coin": asset.lower(), "exchange": exchange, "startTime" : start_time_unix, "limit": limit}
    #query = {"timeframe": timeframe, "coin": asset.lower(), "exchange": exchange, "limit": 1000}    

    data = get_data(orderbook_path, base_url, query)
    #print('Data: ', data)
    df_total = pd.DataFrame.from_dict(data['data'])
    df_data = df_total #['data']
    df_data['time'] = pd.to_datetime(df_data['openDate'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
    df_data = df_data.drop(['openDate'], axis=1)
    return df_data