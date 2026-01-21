
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
  
  # Handle all_categories or custom: use the invest_score that was already calculated
  if category == 'all_categories' or (category == 'custom' and 'invest_score' in df.columns):
      # The invest_score is already calculated in load_and_process_data
      # We just need to use it directly
      y_axis_invested = df['invest_score']
  else:
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
      xaxis=dict(range=[x_axis.min(), x_axis.max()]),  # Stretch to full data range
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

  # Apply the data slice ONCE after all categories are processed
  df_plot = df.iloc[norm_lookback:] # use only data after lookback period
  result_df_plot = result_df.iloc[norm_lookback:]

  for category in categories:
       weight_cat = df_weights[category].iloc[0]
       if category in result_df_plot.columns:
           if weight_cat > 0:
                # Plot if weight > 0
                fig.add_trace(go.Scatter(            
                    x = result_df_plot['date'],   
                    y = result_df_plot[category], 
                    mode = 'lines',  
                    name = category))
           else:
                # Plot dashed if weight = 0
                fig.add_trace(go.Scatter(            
                    x = result_df_plot['date'],   
                    y = result_df_plot[category], 
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
  result_df_plot['total_score'] = result_df['total_score'].iloc[norm_lookback:].values

  fig.add_trace(go.Scatter(          
                x=result_df_plot['date'],   
                y=result_df_plot['total_score'], 
                mode='lines',
                line=dict(width=3, color='blue'),
                name='Total Score'))       

  
  df_plot['close_norm'] = matrix.calc_norm(df_plot['close']) 
  
  df_plot['score_ma'] = matrix.double_hull_ma(result_df_plot['total_score'], 5, 5) 
  df_plot = matrix_strategy.calc_peaks_valleys(df_plot, 'score_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)   
  df_plot = df_plot.copy()  # Defragment DataFrame
  df_plot['extremes'] = df_plot['peaks'].shift(peak_shift).fillna(0) - df_plot['valleys'].shift(peak_shift).fillna(0)
  df_plot['extremes'] = df_plot['extremes'].replace(0, np.nan)
  df_plot['extremes'] = df_plot['extremes'].ffill(axis ='rows') 
  df_plot['invested_total'] = np.where((df_plot['extremes'] < 0), 1, np.nan)  

  fig.add_trace(go.Scatter(            
                x = df_plot['date'],   
                y = df_plot['close_norm'], 
                mode = 'lines',
                line = dict(width=3, color='red'),                  
                name = 'price'))

  fig.add_trace( go.Scatter(  
          x = df_plot['date'],   
          y = df_plot['invested_total'] * df_plot['close_norm'], 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER invested'))

  fig.update_layout(
      title = charttitle_category,
      xaxis=dict(range=[df_plot['date'].min(), df_plot['date'].max()]),  # Stretch to full data range
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
  
  # Handle all_categories: use the invest_score that was already calculated
  if category != 'all_categories':
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




def update_signal_chart(calc_json, signal_strategy, category, custom_metrics=None, show_raw_data=False):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition  

  df_categories = database.read_categories() # read once for all categories
  
  # Handle all_categories: skip individual metric plotting
  if category == 'all_categories':
      print("Metrics Signal: all_categories (weighted combination)")
      metrics_list = []  # Empty list, will just show the combined score
  elif category == 'custom' and custom_metrics:
      # Use the selected custom metrics from UI
      print("Metrics Signal (custom selected): ", custom_metrics)
      metrics_list = custom_metrics
  else:
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
  
  for metric in metrics_list:
        single_metric_list = [metric]
        df_metric = matrix_strategy.calc_multi_strategy(df.copy(), peak_shift, single_metric_list, signal_strategy)  
        
        # Use raw metric values if show_raw_data is enabled, otherwise use invest_score
        if show_raw_data:
            # Show the original metric values
            signal_graph = df_metric[metric] if metric in df_metric.columns else df_metric['invest_score']
        else:
            # Show the normalized invest_score (0-100)
            signal_graph = df_metric['invest_score']

        fig.add_trace( go.Scatter(  
                x = x_axis,   
                y = signal_graph, 
                mode = 'lines',  
                name = metric))
           
  fig.update_layout(
      title = charttitle_signals, 
      legend={'orientation':'h'},
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 




def update_norm_chart(calc_json, category, raw_data, signal_strategy, custom_metrics=None):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6
  #print('Norm Graph DF: ', df, df.columns)

  matrix = matrix_strategy.matrix_strategy()

  df_categories = database.read_categories() # read once for all categories

  # Handle all_categories: skip individual metric plotting
  if category == 'all_categories':
      print("Metrics Norm: all_categories (weighted combination)")
      metrics_list = []  # Empty list
  elif category == 'custom' and custom_metrics:
      # Use the selected custom metrics from UI
      print("Metrics Norm (custom selected): ", custom_metrics)
      metrics_list = custom_metrics
  else:
      metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories) 
      print("Metrics Norm: ", metrics_list_norm)         

  x_axis = df['date']
  y_axis_price = matrix.calc_norm(df['close']) if not raw_data else df['close']

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          name = 'Price'))        

  # Plot individual metric signals (similar to signal chart)
  for metric in metrics_list:
        single_metric_list = [metric]
        df_metric = matrix_strategy.calc_multi_strategy(df.copy(), peak_shift, single_metric_list, signal_strategy)  
        
        # Use raw metric values if show_raw_data is enabled, otherwise use invest_score
        if raw_data:
            # Show the original metric values
            metric_graph = df_metric[metric] if metric in df_metric.columns else df_metric['invest_score']
        else:
            # Show the normalized invest_score (0-100)
            metric_graph = df_metric['invest_score']
        
        fig.add_trace(go.Scatter(
                x = x_axis, 
                y = metric_graph,
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

def create_radar_chart(calc_json, categories, signal_strategy):
    """Create a radar chart of current category scores"""
    # Parse the JSON data
    df = pd.read_json(StringIO(calc_json), orient='records')
    peak_shift = 6
    
    # Read category weights from database
    df_weights = database.read_category_weight_table()
    weights = df_weights.iloc[0]  # Get first row
    df_categories = database.read_categories()
    
    # Get categories with weight > 0
    active_categories = []
    values = []
    
    for category in df_weights.columns:
        weight = weights[category]
        if weight > 0:
            # Calculate the score for this category
            metrics_list, metrics_list_norm = database.load_category_list(category, metric='', df=df_categories)
            if metrics_list:
                # Calculate strategy for this category
                df_cat = matrix_strategy.calc_multi_strategy(df.copy(), peak_shift, metrics_list, signal_strategy)
                # Get the latest score
                score = df_cat['invest_score'].iloc[-1]
                
                active_categories.append(category.replace('_', ' ').title())
                values.append(score)
    
    # Create radar chart
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=active_categories,
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
