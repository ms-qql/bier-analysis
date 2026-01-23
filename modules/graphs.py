
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
charttitle_category = 'Categories Score'
charttitle_norm = 'Norm / Raw Data'

# ------ Support Functions ---------------------------------------------------------------------------------------

norm_lookback = 0 #4 * 360 # use 4 year lookback for normalization (equals to one BTC cycle)

# ------ Show Graphs ---------------------------------------------------------------------------------------



def update_price_chart(calc_json, signal_strategy, category, use_alt_signal=False, alt_signal_deviation=10, allow_short=False):  
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 
     

  df = df.copy()
  df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
  df['extremes'] = df['extremes'].replace(0, np.nan)
  df['extremes'] = df['extremes'].ffill(axis = 0)
  
  if allow_short:
      df['invested'] = np.where(df['extremes'] < 0, 1, -1)
  else:
      df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)

  print('Price Graph: ', df.tail(3), df.info())

  df = df.iloc[norm_lookback:-1] # use only data after lookback period & without last incomplete bar
  
  x_axis = df['date']
  y_axis_price = df['close']
  y_axis_invested = df['invest_score']    
  y_axis_invest_graph = df['invested'] * df['close']

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          line = dict(color='red'),   
          name = 'Price'))

  # Split Long/Short markers
  long_mask = df['invested'] == 1
  short_mask = df['invested'] == -1
  
  fig.add_trace( go.Scatter(  
          x = x_axis[long_mask],   
          y = y_axis_price[long_mask], 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER Long'))

  if allow_short:
       fig.add_trace( go.Scatter(  
          x = x_axis[short_mask],   
          y = y_axis_price[short_mask], 
          mode = 'markers',
          marker = dict(color='darkred'),    
          name = 'BIER Short'))
  
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
      yaxis=dict(title='Price', type='log'),
      yaxis2=dict(title='Score', overlaying='y', side='right', type='linear'),    
      legend={'orientation':'h'},
      margin=dict(l=0, r=0),
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 



def update_norm_chart(calc_json, signal_strategy, category, metrics_list, use_alt_signal=False, alt_signal_deviation=10, allow_short=False):  
  matrix = matrix_strategy.matrix_strategy()    
  df = pd.read_json(StringIO(calc_json), orient='records') 
  peak_shift = 6 # Shift peak by x bars to reflect delayed peak recognition 
  
  df = df.copy()
  df['extremes'] = df['peaks'].shift(peak_shift).fillna(0) - df['valleys'].shift(peak_shift).fillna(0)
  df['extremes'] = df['extremes'].replace(0, np.nan)
  df['extremes'] = df['extremes'].ffill(axis = 0)
  
  if allow_short:
      df['invested'] = np.where(df['extremes'] < 0, 1, -1)
  else:
      df['invested'] = np.where((df['extremes'] < 0), 1, np.nan)

  print('Price Graph: ', df.tail(3), df.info())

  df = df.iloc[norm_lookback:-1] # use only data after lookback period & without last incomplete bar
  
  x_axis = df['date']
  y_axis_price = matrix.calc_norm(df['close']) #if not raw_data else df['close']

  # Plot price line with investment markers
  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          line = dict(color='red'),   
          name = 'Price'))

  # Split Long/Short markers
  long_mask = df['invested'] == 1
  short_mask = df['invested'] == -1

  fig.add_trace( go.Scatter(  
          x = x_axis[long_mask],   
          y = y_axis_price[long_mask], 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER Long'))
          
  if allow_short:
       fig.add_trace( go.Scatter(  
          x = x_axis[short_mask],   
          y = y_axis_price[short_mask], 
          mode = 'markers',
          marker = dict(color='darkred'),    
          name = 'BIER Short'))

  # Add Total Score
  fig.add_trace(go.Scatter(          
                x=x_axis,   
                y=df['invest_score'], 
                mode='lines',
                line=dict(width=3, color='blue'),
                name='Total Score'))

  # Add Peaks Markers (Sell/High checks) on Total Score
  # Peaks in alternative signal logic = Divest signal
  if 'peaks' in df.columns:
      peaks_y = df.apply(lambda row: row['invest_score'] if row['peaks'] == 1 else None, axis=1)
      fig.add_trace(go.Scatter(
          x=x_axis,
          y=peaks_y,
          mode='markers',
          marker=dict(symbol='triangle-down', size=10, color='orange'),
          name='Peak (Divest)'
      ))

  # Add Valleys Markers (Buy/Low checks) on Total Score
  # Valleys in alternative signal logic = Invest signal
  if 'valleys' in df.columns:
      valleys_y = df.apply(lambda row: row['invest_score'] if row['valleys'] == 1 else None, axis=1)
      fig.add_trace(go.Scatter(
          x=x_axis,
          y=valleys_y,
          mode='markers',
          marker=dict(symbol='triangle-up', size=10, color='lime'),
          name='Valley (Invest)'
      ))

  # Add individual metrics
  for metric in metrics_list:                 
        #print(f'Metric for {metric} Graph: /n', df[metric])
        fig.add_trace(go.Scatter(
                x = x_axis, 
                y = matrix.calc_norm(df[metric]),
                mode = 'lines',
                name = metric))


  fig.update_layout(
    title = charttitle_norm, 
    legend={'orientation':'h'},
    margin=dict(l=0, r=0),
    paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
    plot_bgcolor= 'rgba(0,0,0,0)') # specify layout
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 





def update_category_chart(calc_json, signal_strategy, category, use_alt_signal=False, alt_signal_deviation=10, allow_short=False):  
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
       df = matrix_strategy.calc_multi_strategy(df, peak_shift, metrics_list, signal_strategy, use_alt_signal, alt_signal_deviation)
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
  
  if use_alt_signal:
      # Use alternative signal logic on total_score
      temp_df = pd.DataFrame({'invest_score': result_df['total_score']})
      temp_df = matrix_strategy.calc_alternative_signal(temp_df, 'invest_score', deviation=alt_signal_deviation)
      total_invested_full = temp_df['invested'].values
      df_plot['invested_total'] = total_invested_full[norm_lookback:][:len(df_plot)]
  else:
      df_plot['score_ma'] = matrix.double_hull_ma(result_df_plot['total_score'], 5, 5) 
      df_plot = matrix_strategy.calc_peaks_valleys(df_plot, 'score_ma', peak_min = 50, vert_dist = 0, peak_dist = 2, peak_width = 0, peak_prominence = 10, filt_double_extremes = False)   
      df_plot = df_plot.copy()  # Defragment DataFrame
      df_plot['extremes'] = df_plot['peaks'].shift(peak_shift).fillna(0) - df_plot['valleys'].shift(peak_shift).fillna(0)
      df_plot['extremes'] = df_plot['extremes'].replace(0, np.nan)
      df_plot['extremes'] = df_plot['extremes'].ffill(axis ='rows') 
      
      if allow_short:
          df_plot['invested_total'] = np.where(df_plot['extremes'] < 0, 1, -1)
      else:
          df_plot['invested_total'] = np.where((df_plot['extremes'] < 0), 1, np.nan)  

  fig.add_trace(go.Scatter(            
                x = df_plot['date'],   
                y = df_plot['close_norm'], 
                mode = 'lines',
                line = dict(width=3, color='red'),                  
                name = 'price'))

  # Split Long/Short markers
  long_mask = df_plot['invested_total'] == 1
  short_mask = df_plot['invested_total'] == -1

  fig.add_trace( go.Scatter(  
          x = df_plot['date'][long_mask],   
          y = df_plot['close_norm'][long_mask], 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER Long'))
          
  if allow_short:
       fig.add_trace( go.Scatter(  
          x = df_plot['date'][short_mask],   
          y = df_plot['close_norm'][short_mask], 
          mode = 'markers',
          marker = dict(color='darkred'),    
          name = 'BIER Short'))

  fig.update_layout(
      title = charttitle_category,
      xaxis=dict(range=[df_plot['date'].min(), df_plot['date'].max()]),  # Stretch to full data range
      yaxis=dict(title='Categories'),
      margin=dict(l=0, r=0), 
      legend={'orientation':'h'},
      paper_bgcolor= 'rgba(0,0,0,0)', # Transparent
      plot_bgcolor= 'rgba(0,0,0,0)') # Transparent
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='silver',zeroline=True, zerolinewidth=2, zerolinecolor='silver')

  return fig.to_json() 




'''def update_norm_chart(calc_json, category, raw_data, signal_strategy, custom_metrics=None, use_alt_signal=False, alt_signal_deviation=10):  
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
  
  # Calculate Total Score based on the metrics in the chart
  # Note: logic behaves similar to update_category_chart but scoped to the selected metrics
  df_total = matrix_strategy.calc_multi_strategy(df.copy(), peak_shift, metrics_list, signal_strategy, use_alt_signal)
  total_score = df_total['invest_score']

  # Use centralized signal logic from calc_multi_strategy (df_total)
  # Map 'invested' to 'invested_total' for compatibility if needed, or use directly
  if 'invested' in df_total.columns:
      df_total['invested_total'] = df_total['invested']
  else:
      # Fallback should not be hit if calc_multi_strategy works as expected
      df_total['invested_total'] = np.nan

  # Determine Price Y-axis
  if raw_data:
      y_axis_price = df['close']
      price_name = 'Price' 
  else:
      y_axis_price = matrix.calc_norm(df['close'])
      price_name = 'price'

  fig = go.Figure(go.Scatter(
          x = x_axis, 
          y = y_axis_price,
          mode = 'lines',
          line = dict(width=3, color='red'),
          name = price_name)) 

  # Add Total Score
  fig.add_trace(go.Scatter(          
                x=x_axis,   
                y=total_score, 
                mode='lines',
                line=dict(width=3, color='blue'),
                name='Total Score'))

  # Add BIER Invested (dots on Price line)
  fig.add_trace( go.Scatter(  
          x = x_axis,   
          y = df_total['invested_total'] * y_axis_price, 
          mode = 'markers',
          marker = dict(color='green'),    
          name = 'BIER invested'))

  # Add Peaks Markers (Sell/High checks) on Total Score
  # Peaks in alternative signal logic = Divest signal
  if 'peaks' in df_total.columns:
      peaks_y = df_total.apply(lambda row: row['invest_score'] if row['peaks'] == 1 else None, axis=1)
      fig.add_trace(go.Scatter(
          x=x_axis,
          y=peaks_y,
          mode='markers',
          marker=dict(symbol='triangle-down', size=10, color='orange'),
          name='Peak (Divest)'
      ))

  # Add Valleys Markers (Buy/Low checks) on Total Score
  # Valleys in alternative signal logic = Invest signal
  if 'valleys' in df_total.columns:
      valleys_y = df_total.apply(lambda row: row['invest_score'] if row['valleys'] == 1 else None, axis=1)
      fig.add_trace(go.Scatter(
          x=x_axis,
          y=valleys_y,
          mode='markers',
          marker=dict(symbol='triangle-up', size=10, color='lime'),
          name='Valley (Invest)'
      ))

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

  return fig.to_json() '''

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
