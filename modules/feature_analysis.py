import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import mutual_info_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

def calculate_mi_ranking(df, target_col='close', lookahead=7, bins=50):
    """
    Calculates Mutual Information (MI) between features and future returns.
    Uses Equal-Frequency Binning and Miller-Madow bias correction.
    
    Args:
        df: Input DataFrame containing features and target.
        target_col: Name of the target column (usually 'close' price).
        lookahead: Number of days to look ahead for target returns.
        bins: Number of bins for discretization.
        
    Returns:
        pd.DataFrame: Sorted dataframe with 'Feature' and 'MI_Score'.
    """
    df_mi = df.copy()
    
    # 1. Create Target: Future Returns
    # We want to predict returns lookahead days into the future
    df_mi['target_returns'] = df_mi[target_col].pct_change(lookahead).shift(-lookahead)
    
    # Drop rows where EXPECTED TARGET is NaN (e.g. at the end of the series)
    df_mi = df_mi.dropna(subset=['target_returns'])
    
    if df_mi.empty:
        return pd.DataFrame(columns=['Feature', 'MI_Score', 'Raw_MI'])

    # 2. Setup
    features = [c for c in df_mi.columns if c not in ['date', target_col, 'target_returns']]
    mi_scores = []
    
    # Pre-calculate Global Discretized Target (for common index, but we might need to re-align per feature)
    # Actually, simpler to do it inside the loop to match indices perfectly.
    
    N_global = len(df_mi)
    
    # 3. Calculate MI for each feature
    for feature in features:
        try:
            # Subset valid data for this specific pair
            df_pair = df_mi[[feature, 'target_returns']].dropna()
            
            if len(df_pair) < bins:
                 # print(f"DEBUG: Skipping {feature} - not enough data points ({len(df_pair)})")
                 continue

            # Discretize Feature (Equal-Frequency)
            x_discrete = pd.qcut(df_pair[feature], q=bins, labels=False, duplicates='drop')
            
            # Discretize Target (on the same subset)
            try:
                y_discrete = pd.qcut(df_pair['target_returns'], q=bins, labels=False, duplicates='drop')
            except ValueError:
                y_discrete = pd.cut(df_pair['target_returns'], bins=bins, labels=False)
            
            # Calculate Raw MI
            mi = mutual_info_score(x_discrete, y_discrete)
            
            # 4. Miller-Madow Bias Correction
            # MM_correction = (R-1)(C-1) / (2N)
            N = len(df_pair)
            R = len(np.unique(x_discrete))
            C = len(np.unique(y_discrete))
            
            correction = ((R - 1) * (C - 1)) / (2 * N)
            mi_adjusted = mi - correction
            
            # Clamp to 0
            mi_final = max(0, mi_adjusted)
            
            mi_scores.append({
                'Feature': feature,
                'MI_Score': mi_final,
                'Raw_MI': mi,
                'Samples': N # Useful debug info
            })
            
        except Exception as e:
            print(f"Error calculating MI for {feature}: {e}")
            mi_scores.append({
                'Feature': feature,
                'MI_Score': 0,
                'Raw_MI': 0,
                'Samples': 0
            })

    # Create Result DataFrame
    mi_df = pd.DataFrame(mi_scores)
    mi_df = mi_df.sort_values('MI_Score', ascending=False).reset_index(drop=True)
    
    return mi_df

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
    return fig

def calculate_vif(df):
    """
    Calculates Variance Inflation Factor (VIF) for each feature.
    """
    # Drop NaNs and ensure numeric
    df_vif = df.select_dtypes(include=[np.number]).dropna()
    
    # Add constant? VIF usually requires constant if not centered, but 
    # typically we look at feature-vs-feature linear dependency.
    # VIF requires X.
    # VIF requires X.
    if df_vif.empty:
         return pd.DataFrame(columns=['Feature', 'VIF'])
         
    if df_vif.shape[0] < 10:
         # Not enough samples for reliable VIF
         return pd.DataFrame(columns=['Feature', 'VIF'])

    # Check for NaNs or Inf just in case
    if np.any(np.isinf(df_vif.values)) or np.any(np.isnan(df_vif.values)):
         # Clean invalid values if necessary, or let statsmodels fail/handle
         pass

    thresh_scores = []
    
    # Iterate to calculate VIF for each feature
    # Note: This can be slow for many features.
    for i, feature in enumerate(df_vif.columns):
        try:
            # VIF = 1 / (1 - R^2)
            vif = variance_inflation_factor(df_vif.values, i)
        except Exception:
            vif = np.inf
            
        thresh_scores.append({
            'Feature': feature,
            'VIF': vif
        })
        
    return pd.DataFrame(thresh_scores).sort_values('VIF', ascending=False)

def analyze_vif_clusters(df, threshold=10):
    """
    Stage 3: VIF Analysis & PCA Selection.
    1. Calculate VIF.
    2. Identify High-VIF features (> threshold).
    3. Cluster High-VIF features (using correlation distance).
    4. Run PCA on each cluster -> Pick feature with max loading on PC1.
    
    Returns:
        vif_df: DataFrame of all VIF scores.
        recommendations: List of dicts describing clusters and actions.
    """
    # 1. Calculate Full VIF
    vif_df = calculate_vif(df)
    
    # 2. Identify Candidates
    high_vif_features = vif_df[vif_df['VIF'] > threshold]['Feature'].tolist()
    
    recommendations = []
    
    if not high_vif_features:
        return vif_df, recommendations

    # 3. Cluster High-VIF Features
    # We need the correlation matrix of ONLY the high-vif features
    df_high = df[high_vif_features]
    corr_high = df_high.corr().fillna(0)
    
    # Distance matrix
    dist_matrix = 1 - np.abs(corr_high)
    np.fill_diagonal(dist_matrix.values, 0)
    
    try:
        # Hierarchical Clustering
        # We need to define a cut-off to form clusters. 
        # A correlation > 0.7 (distance < 0.3) is usually a good "cluster" proxy for VIF groups, 
        # or we just cut the tree to find natural groups.
        # Let's use distance threshold t=0.5 (corr > 0.5) to be safe, or stricter.
        # Given VIF > 10 usually implies R^2 > 0.9, we expect high correlations.
        
        dist_condensed = squareform(dist_matrix)
        linkage_matrix = hierarchy.linkage(dist_condensed, method='ward')
        
        # Form flat clusters
        # t=1.0 means keeping things quite related
        cluster_labels = hierarchy.fcluster(linkage_matrix, t=0.5, criterion='distance')
        
        # Group features by cluster label
        clusters = {}
        for feat, label in zip(high_vif_features, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feat)
            
        # 4. PCA Selection per Cluster
        for label, feats in clusters.items():
            if len(feats) < 2:
                continue # Singletons don't need reduction (though they have High VIF? Maybe with outside features?)
                
            # Run PCA
            # Standardize first
            X = df[feats].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=1)
            pca.fit(X_scaled)
            
            # Loadings: pca.components_[0]
            loadings = np.abs(pca.components_[0])
            
            # Find Winner: Max loading
            winner_idx = np.argmax(loadings)
            winner = feats[winner_idx]
            
            losers = [f for f in feats if f != winner]
            
            recommendations.append({
                'Cluster_ID': int(label),
                'Features': feats,
                'Representative_Feature': winner,
                'Suggested_Drops': losers,
                'Explained_Variance_PC1': float(pca.explained_variance_ratio_[0])
            })
            
    except Exception as e:
        print(f"Error in VIF Clustering: {e}")
        # Fallback: Just return raw VIF
        
    return vif_df, recommendations

def analyze_boruta_stability(df, target_col='close', lookahead=7):
    """
    Stage 4: Regime Robustness (Boruta Walk-Forward).
    """
    df_boruta = df.copy()
    
    # 1. Create Target
    
    # 1. Create Target (can introduce NaNs at end)
    df_boruta['target_returns'] = df_boruta[target_col].pct_change(lookahead).shift(-lookahead)
    
    # Do NOT globally dropna. We want to keep early data for some features.
    # But we must drop rows where TARGET is missing (e.g. at the very end).
    df_boruta = df_boruta.dropna(subset=['target_returns'])
    
    # Ensure index is datetime
    if 'date' in df_boruta.columns:
        df_boruta['date'] = pd.to_datetime(df_boruta['date'])
        df_boruta = df_boruta.set_index('date')
    elif not isinstance(df_boruta.index, pd.DatetimeIndex):
         df_boruta.index = pd.to_datetime(df_boruta.index)

    # Features (Numeric Only)
    feature_cols_all = [c for c in df_boruta.columns if c not in [target_col, 'target_returns', 'date']]
    
    # Align full X and y on index
    # (Since we only dropped target_returns NaN, they should be aligned, but let's be safe)
    y_full = df_boruta['target_returns']
    X_full = df_boruta[feature_cols_all] # Keep all numeric cols, some may have NaNs
    
    start_date = X_full.index.min()
    end_date = X_full.index.max()
    print(f"DEBUG: Start Date: {start_date}, End Date: {end_date}")
    
    current_date = start_date
    windows_results = []
    total_windows_attempted = 0
    
    # Stats Tracking
    feature_stats = {f: {'available': 0, 'confirmed': 0} for f in feature_cols_all}
    
    # 6-month windows loop
    while current_date < end_date:
        window_end = current_date + pd.DateOffset(months=6)
        train_end = current_date + pd.DateOffset(months=4)
        
        if train_end > end_date:
            break
            
        # Get Window Slice
        mask_window = (X_full.index >= current_date) & (X_full.index < train_end)
        X_window_raw = X_full.loc[mask_window]
        y_window = y_full.loc[mask_window]
        
        # Determine Valid Features for THIS window
        # Allow features that have full coverage (no NaNs) in this specific window
        # or maybe allow small % missing? Let's say we require 100% coverage for the window to include the feature.
        # This allows "New Features" to be excluded from Old Windows, but included in New Windows.
        features_in_window = X_window_raw.columns[X_window_raw.notna().all()].tolist()
        
        if len(features_in_window) < 1:
             current_date = window_end
             continue
             
        X_train = X_window_raw[features_in_window]
        
        # Check sample size
        if len(X_train) > 10: 
            total_windows_attempted += 1
            print(f"DEBUG: Window {total_windows_attempted} ({current_date.date()} to {train_end.date()}): {len(X_train)} samples, {len(features_in_window)} features")
            
            # Boruta Setup
            rf = RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=500) 
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100, alpha=0.01)
            
            try:
                feat_selector.fit(X_train.values, y_window.values)
                
                confirmed = X_train.columns[feat_selector.support_].tolist()
                tentative = X_train.columns[feat_selector.support_weak_].tolist()
                
                # Update Stats
                for f in features_in_window:
                    feature_stats[f]['available'] += 1
                    
                for f in confirmed:
                    feature_stats[f]['confirmed'] += 1
                for f in tentative:
                    feature_stats[f]['confirmed'] += 1 # Treating tentative as confirmed
                    
            except Exception as e:
                print(f"Boruta failed for window {current_date}: {e}")
        
        current_date = window_end

    # Aggregate
    stability_data = []
    
    # Only report on features that were available at least once
    # Or report all, with 0 availability?
    for feat, stats in feature_stats.items():
        avail = stats['available']
        confirmed = stats['confirmed']
        
        if avail > 0:
            pct = confirmed / avail
            
            if pct >= 0.7:
                 status = "Robust"
            elif pct >= 0.3:
                 status = "Regime-Sensitive"
            else:
                 status = "Inconsistent"
                 
            stability_data.append({
                'Feature': feat,
                'Stability_Score': pct,
                'Status': status,
                'Windows_Count': avail,
                'Confirmed_Count': confirmed
            })
            
    if not stability_data:
        return pd.DataFrame(columns=['Feature', 'Stability_Score', 'Status', 'Windows_Count']), 0
            
    return pd.DataFrame(stability_data).sort_values('Stability_Score', ascending=False), total_windows_attempted
