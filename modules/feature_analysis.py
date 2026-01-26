import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

def calculate_correlation(df, target_col='close', threshold=0.95):
    """
    Calculates the Pearson correlation matrix for the dataframe.
    Drops non-numeric columns.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calculate Correlation Matrix
    corr_matrix = df_numeric.corr(method='pearson')
    
    return corr_matrix

def identify_high_correlation_features(corr_matrix, threshold=0.95):
    """
    Identifies features with absolute correlation greater than threshold.
    Returns:
    - pairs: List of tuples (feature1, feature2, correlation)
    - to_drop: List of features suggested to drop (simplified approach: drop the second one encountered)
    """
    pairs = []
    to_drop = set()
    
    # Get upper triangle of correlation matrix (excluding diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index of feature columns with correlation greater than threshold
    to_drop_cols = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]
    
    for col in to_drop_cols:
        # Find which row (feature) it correlates with
        correlated_rows = upper_tri.index[upper_tri[col].abs() > threshold].tolist()
        for row in correlated_rows:
            correlation_val = upper_tri.loc[row, col]
            pairs.append((row, col, correlation_val))
            # Naive strategy: drop 'col' (the one in the columns list, which comes later in order)
            to_drop.add(col)
            
    return pairs, list(to_drop)

def get_feature_clusters(corr_matrix):
    """
    Performs hierarchical clustering to order features for better heatmap visualization.
    Returns list of feature names in clustered order.
    """
    # Handle NaN values in correlation matrix (replace with 0 for clustering)
    corr_clean = corr_matrix.fillna(0)
    
    # Convert correlation to distance matrix
    dist_matrix = 1 - np.abs(corr_clean)
    
    # Ensure symmetric and zero diagonal
    np.fill_diagonal(dist_matrix.values, 0)
    dist_condensed = squareform(dist_matrix)
    
    try:
        # Hierarchical clustering
        linkage_matrix = hierarchy.linkage(dist_condensed, method='ward')
        dendrogram = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        leaves_order = dendrogram['leaves']
        
        # Reorder columns
        clustered_features = [corr_matrix.columns[i] for i in leaves_order]
        return clustered_features
    except Exception as e:
        print(f"Clustering failed: {e}. Returning original order.")
        return corr_matrix.columns.tolist()

def create_heatmap(corr_matrix, title="Feature Correlation Matrix"):
    """
    Creates an interactive Plotly Heatmap from the correlation matrix.
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Pearson Corr')
    ))
    
    # Update layout to be square and handle many labels
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-90, side='bottom'),
        yaxis=dict(autorange='reversed'), # Match matrix orientation
        width=1000,
        height=1000,
        #margin=dict(l=200, r=100, t=100, b=200) # Give space for labels
    )
    
    return fig
