import numpy as np
import pandas as pd



# This module follows the CRUD (Create, Read, Update, Delete) structure

# ----------------------------- Create & Load tables ----------------------------------------------------


# ----------------------------- Read tables -------------------------------------------------------------




def read_categories():
    """
    Load metric data from the Postgres table 'bier_categories' and return as a DataFrame.
    Returns: pandas.DataFrame: DataFrame with categories data.
    """
    from . import database
    dict_categories = database.read_bier_categories()
    df_categories = pd.DataFrame.from_dict(dict_categories)    
    if 'id' in df_categories.columns:
        df_categories = df_categories.drop(columns=['id'])
    return df_categories  


def load_category_list(category: str, metric='', df=None) -> list:
    """
    Load the category list from the Anvil datatable and return the metrics for the specified category.
    
    Args:
        category (str): The category to filter by (e.g., 'capriole', 'bmp', etc.)
        
    Returns:
        list: List of metric names for the specified category
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

    #print(f"Metrics in category '{category}': {category_list}")      
    #print(f"Norm Metrics in category '{category}': {category_norm_list}")

    return category_list, category_norm_list

def read_category_weight_table():
    from . import database
    dict_weights = database.read_bier_category_weight()
    df_weights = pd.DataFrame.from_dict(dict_weights)    
    if 'id' in df_weights.columns:
        df_weights = df_weights.drop(columns=['id'])
    return df_weights 

