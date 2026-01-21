
import json

import numpy as np
import pandas as pd
import datetime
import dateutil.parser
from datetime import datetime, date, timedelta
import time
import plotly.graph_objects as go
#import math
from . import date_time
#from . import datatable
from . import database



import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO

from . import matrix_strategy 
#from . import matrix_bot


#from modules.database import read_table_from_date 



######## graphs ############################################################################################################

# chart titles
charttitle_price = 'Metrics & Price (USD)' 
charttitle_strategy = 'Strategy Score'
charttitle_category = 'Categories Score'
charttitle_norm = 'Norm / Raw Data'
charttitle_signals = 'Signals'

# ------ Support Functions ---------------------------------------------------------------------------------------

norm_lookback = 0 #4 * 360 # use 4 year lookback for normalization (equals to one BTC cycle)

# ------ Show Graphs ---------------------------------------------------------------------------------------



def update_price_chart(calc_json, signal_strategy, category, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 
  
  df_categories = database.read_categories() # read once for all categories
  metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 

  df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list, signal_strategy)
  y_axis_invested = df['invest_score']       
  
  print('Price Graph: ', df.tail(3), df.info())

  df = df.iloc[norm_lookback:-1] # use only data after lookback period & without last incomplete bar
  
  x_axis = df['date']
  y_axis_price = df['close']
  y_axis_invest_graph = df['invested'] * df['close']

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          line = dict(color='red'),   
          name = 'Price'))

  fig.add_trace( go.Scatter(  
          x = x_axis,   
          y = y_axis_invest_graph, 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER invested'))
  
  if signal_strategy:
        y_axis_range_graph = df['range'] * df['close']
        fig.add_trace( go.Scatter(  
                x = x_axis,   
                y = y_axis_range_graph, 
                mode = 'markers',
                marker = dict(color='yellow'),    
                name = 'Range phase')) 

  fig.add_trace(go.Scatter(
          x = x_axis, 
          y = y_axis_invested,
          #y = y_axis_metric,    
          mode = 'lines',
          line = dict(color='blue'),        
          name = 'Score',
          yaxis='y2'))
    
  fig.update_layout(
      title = charttitle_price,
      yaxis=dict(title='Price'),
      yaxis2=dict(title='Score', overlaying='y', side='right'),    
      legend={'orientation':'h'},
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver', type='log')

  return fig.to_json() 




def update_category_chart(calc_json, signal_strategy, category):  
  matrix = matrix_strategy.matrix_strategy()
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 
  
  df_categories = database.read_categories() # read once for all categories
  #print('Categories DF: ', df_categories, df_categories.columns)
  df_weights = database.read_category_weight_table() # read weights of all categories
  weights = df_weights.iloc[0].values # get weights from the first row  
  #print('Weights DF: ', df_weights, df_weights.columns)
  categories = list(df_weights.columns)
  # DF to collect category scores
  result_df = None

  # Create empty figure
  fig = go.Figure()
 
  for category in categories:
       weight_cat = df_weights[category].iloc[0]
       #print(f'Category {category} with weigth: ', weight_cat)
       #print('Position: ', categories.index(category)
       #pos = categories.index(category) # get position to access weight

       metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
       if not metrics_list:
            print(f"Skipping category {category} because it has no metrics.")
            continue
       df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list, signal_strategy)
       #print('Category Graph: ', df.columns)
       
       if result_df is None:
            # Erste Kategorie: DataFrame mit date, close und erste Kategorie initialisieren
            result_df = pd.DataFrame({
                'date': df['date'].values,
                'close': df['close'].values,
                category: df['invest_score'].values
            })
       else:
            # Weitere Kategorien hinzufügen
            # Sicherstellen, dass die Längen übereinstimmen
            min_len = min(len(result_df), len(df))
            result_df = result_df.iloc[:min_len].copy()
            result_df[category] = df['invest_score'].values[:min_len]

       df = df.iloc[norm_lookback:-1] # use only data after lookback period & without last incomplete bar

       if weight_cat > 0:
            # Plot if weight > 0
            fig.add_trace(go.Scatter(            
                x = df['date'],   
                y = df['invest_score'], 
                mode = 'lines',  
                name = category))
       else:
            # Plot dashed if weight = 0
            fig.add_trace(go.Scatter(            
                x = df['date'],   
                y = df['invest_score'], 
                mode = 'lines',
                line=dict(width=1, dash='dot'),
                opacity=0.8,                  
                name = category))            

  category_columns = [col for col in result_df.columns if col not in ['date', 'close']]
  
  # user-defined weights to calculate weighted average
  weighted_sum = pd.Series(0.0, index=result_df.index)
  total_weight = 0
  
  for i, category in enumerate(category_columns):
        weight = weights[i]
        if weight > 0:
            weighted_sum += result_df[category] * weight
            total_weight += weight
            print(f"Added {category} with weight {weight}")
    
  result_df['total_score'] = weighted_sum / total_weight if total_weight > 0 else 0

  fig.add_trace(go.Scatter(          
                x=df['date'],   
                y=result_df['total_score'], 
                mode='lines',
                line=dict(width=3, color='blue'),
                name='Total Score'))       

  
  df['close_norm'] = matrix.calc_norm(df['close']) 
  
  df['score_ma'] = matrix.double_hull_ma(result_df['total_score'], 5, 5) 
  df = matrix_strategy.calc_peaks_valleys(df, 'score_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)   
  df = df.copy()  # Defragment DataFrame
  df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
  df['extremes'] = df['extremes'].replace(0, np.nan)
  df['extremes'] = df['extremes'].ffill(axis ='rows') 
  df['invested_total'] = np.where((df['extremes'] < 0), 1, np.nan)  

  fig.add_trace(go.Scatter(            
                x = df['date'],   
                y = df['close_norm'], 
                mode = 'lines',
                line = dict(width=3, color='red'),                  
                name = 'price'))

  fig.add_trace( go.Scatter(  
          x = df['date'],   
          y = df['invested_total'] * df['close_norm'], 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER invested'))

  fig.update_layout(
      title = charttitle_category,
      yaxis=dict(title='Categories'), 
      legend={'orientation':'h'},
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 




def update_strategy_chart(calc_json, signal_strategy, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight, category):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 

  #df = matrix_strategy.calc_strategy(df, peak_shift, risk_weight, market_weight, mining_weight, macro_weight, sentiment_weight, hodl_weight, shortterm_weight, custom_weight, single_weight)

  df_categories = database.read_categories() # read once for all categories
  metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
  #print("Metrics Strategy: ", metrics_list_norm)     

  df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list, signal_strategy)

  # Create forecast column
  df = matrix_strategy.create_forecast(df, score_column='invest_score')
  df = df.iloc[norm_lookback:-1] # use only data after lookback period & without last incomplete bar
  
  x_axis = df['date']

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = df['close_norm'],
          line = dict(color='red'),      
          mode = 'lines',
          name = 'Price'))

  fig.add_trace( go.Scatter(  
          x = x_axis,   
          y = df['invested_ds'] * df['close_norm'], 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER invested (Dual Strategy)'))
  
  fig.add_trace(go.Scatter(
          x = x_axis, 
          y = df['invest_score'],
          mode = 'lines',
          line = dict(color='black'),   
          name = 'Strategy Score'))   
  
  fig.add_trace(go.Scatter(
          x = x_axis, 
          y = df['forecast'] * 5,
          mode = 'lines',
          line = dict(color='blue'),   
          name = 'Forecast'))       
    
  if not signal_strategy:

        fig.add_trace(go.Scatter(
                x = x_axis, 
                y = df['peaks'] * df['invest_score'],
                mode = 'markers',
                marker=dict(size=8, color='orange',symbol='cross'),    
                name = 'Peaks'))
  
        fig.add_trace(go.Scatter(
                x = x_axis, 
                y = df['valleys'] * df['invest_score'],
                mode = 'markers',
                marker=dict(size=8, color='blue',symbol='cross'),    
                name = 'Valleys'))

  
  fig.update_layout(
      title = charttitle_strategy,
      #yaxis=dict(title='Price'),
      #yaxis2=dict(title=metric, overlaying='y', side='right'),    
      legend={'orientation':'h'},
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 

'''
# @anvil.server.callable
def update_category_chart(calc_json, category, signal_strategy=True):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition  

  df_categories = database.read_categories() # read once for all categories
  metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
  #print("Metrics Strategy: ", metrics_list_norm) 

  matrix = matrix_strategy.matrix_strategy()
  
  #df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list)


  """for metric in metrics_list:
    norm_name = metric + '_norm'

    # Funding is already 0...100, hence no normalization needed
    if 'funding' in metric:
      df[norm_name] = df[metric]
    else:
      df[norm_name] = matrix.calc_norm(df[metric])

  print('Category Chart: ', df.tail().to_string(), df.columns)"""

  x_axis = df['date']
  y_axis_price = matrix.calc_norm(df['close'])  

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          name = 'Price'))
  
  for metric in metrics_list:
        single_metric_list = [metric]
        df = matrix_strategy.calc_multi_strategy(df, peak_shift, single_metric_list, signal_strategy)      

        fig.add_trace( go.Scatter(  
                x = x_axis,   
                y = df['invest_score'], 
                mode = 'lines',  
                name = metric))
        

  """for metric in metrics_list:
    norm_name = metric + '_norm'    
    fig.add_trace(go.Scatter(
            x = x_axis, 
            y = df[norm_name],
            mode = 'lines',
            name = metric))

    try:
      invest_name = metric + '_invested'
      fig.add_trace(go.Scatter(
              x = x_axis,
              y = df[invest_name],
              mode = 'lines',
              name = invest_name))
    except:
      pass"""
    

  fig.update_layout(
      title = charttitle_norm, 
      legend={'orientation':'h'},
      paper_bgcolor= 'white', #Aussenrand,
      plot_bgcolor= 'white') #'#363636') # specify layout
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() '''




def update_signal_chart(calc_json, signal_strategy, category):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition  

  df_categories = database.read_categories() # read once for all categories
  metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
  print("Metrics Signal: ", metrics_list)  

  matrix = matrix_strategy.matrix_strategy()
  df['close_norm'] = matrix.calc_norm(df['close']) 
  #df = df.loc[norm_lookback:]   

  x_axis = df['date']
  y_axis_price = df['close_norm'] #matrix.calc_norm(df['close']) 

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          name = 'Price'))
  
  offset = 0
  for metric in metrics_list:
        single_metric_list = [metric]
        df = matrix_strategy.calc_multi_strategy(df, peak_shift, single_metric_list, signal_strategy)  
        signal_graph = df['invest_score'] - offset            

        fig.add_trace( go.Scatter(  
                x = x_axis,   
                y = signal_graph, 
                mode = 'lines',  
                name = metric))
        offset += 5
           
  fig.update_layout(
      title = charttitle_signals, 
      legend={'orientation':'h'},
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 




def update_norm_chart(calc_json, category, raw_data):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  #print('Norm Graph DF: ', df, df.columns)

  matrix = matrix_strategy.matrix_strategy()

  df_categories = database.read_categories() # read once for all categories

  metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
  print("Metrics Norm: ", metrics_list_norm)         

  x_axis = df['date']
  y_axis_price = matrix.calc_norm(df['close']) if not raw_data else df['close']

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          name = 'Price'))        

  for metric in metrics_list:                 
          #print(f'Metric for {metric} Graph: /n', df[metric])
          fig.add_trace(go.Scatter(
                  x = x_axis, 
                  y = matrix.calc_norm(df[metric] if not raw_data else df[metric]),
                  mode = 'lines',
                  name = metric)) #.replace("_norm","")))

  if category == 'sentiment':
        fear_greed_blend = (df['fear_greed_norm'] * 2.0 + df['augmento_norm'] * 2.0 + df['equity_fear_greed_norm'] * 1.0) / 5.0
        fig.add_trace(go.Scatter(        
          x = x_axis, 
          y = fear_greed_blend,
          mode = 'lines',
          marker = dict(color='red'),             
          name = 'F&G Blend'))

  # Commented out RSI averages etc...

  fig.update_layout(
    title = charttitle_norm, 
    legend={'orientation':'h'},
    paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
    plot_bgcolor= 'rgba(0,0,0,0)') # specify layout
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 

def create_radar_chart(df, categories):
    """Create a radar chart of current category scores"""
    # Get latest values for the selected categories
    latest = df.iloc[-1]
    
    # Since we don't have per-category score columns directly in the main df easily,
    # we'll approximate using the 'invest_score' if it was run for that category.
    # For a real radar, we would need to run calc_multi_strategy for each category.
    # For now, let's just use placeholder logic or a subset if available.
    
    # Better approach: If we are showing 'bier', we can show the contributors.
    contributors = ['market', 'mining', 'macro', 'sentiment', 'hodl', 'supply_demand']
    values = []
    
    # This is a simplification for the demo. In a full version, we'd pre-calculate these.
    for cat in contributors:
        # Placeholder: using a random variation of invest_score for visualization
        # In production, replace with actual category scores
        val = latest.get('invest_score', 50) * (0.8 + 0.4 * np.random.rand())
        values.append(min(100, max(0, val)))
        
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=contributors,
      fill='toself',
      name='Current Regime'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, 100]),
        bgcolor='rgba(0,0,0,0)'
      ),
      showlegend=False,
      title="Current Regime Health",
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig.to_json() 
