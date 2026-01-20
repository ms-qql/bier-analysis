import http.client
import json
import numpy as np
import pandas as pd

conn = http.client.HTTPSConnection("fn.intothecryptoverse.com")
payload = json.dumps({})
headers = {
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
conn.request("POST", "/extApi/v1/chart/historical", payload, headers)
res = conn.getresponse()
data = res.read().decode("utf-8")
itc_data = json.loads(data)

# Extract the series list
series = itc_data['data']['series']#[0]

df = pd.DataFrame(series)
df = df.rename(columns={'d': 'date', 'btc_usd': 'close', 'btc_risk': 'risk_level', 'alt_mcap_usd': 'alt_mcap', 'dxy_risk': 'dxy_usd_risk', 'eth_risk': 'eth_usd_risk', 'bnb_risk': 'bnb_usd_risk', 'sol_risk': 'sol_usd_risk', 'ltc_risk': 'ltc_usd_risk'})
#print(df.info())

df_itc = df[['date', 'mcap', 'mcap_risk', 'close', 'risk_level', 'alt_mcap', 'alt_mcap_risk', 'dxy_usd', 'dxy_usd_risk', 'eth_usd', 'eth_usd_risk', 'bnb_usd', 'bnb_usd_risk', 'sol_usd', 'sol_usd_risk', 'ltc_usd', 'ltc_usd_risk']]
#df_itc.to_csv('itc_risk_levels.csv', index=False)
#print(df_itc)

