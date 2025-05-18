# src/feature_engineering/feature_extraction.py

import os
import pandas as pd
import numpy as np
from scipy import stats

def extract_timeseries_features(series, product_id):
    """Extract comprehensive features from time series"""
    # Basic statistics
    features = {
        'product_id': product_id,
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min()
    }

    # Process states (based on bimodal distribution)
    threshold = 0  # Separates high and low states
    high_state = series[series > threshold]
    low_state = series[series <= threshold]

    # State-based features
    features.update({
        'high_state_mean': high_state.mean() if len(high_state) > 0 else 0,
        'low_state_mean': low_state.mean() if len(low_state) > 0 else 0,
        'high_state_ratio': len(high_state) / len(series),
        'high_state_std': high_state.std() if len(high_state) > 0 else 0,
        'low_state_std': low_state.std() if len(low_state) > 0 else 0
    })

    # Transition features
    diff = series.diff()
    large_drops = diff[diff < -50]  # Significant drops

    if not large_drops.empty:
        largest_drop_idx = large_drops.idxmin()
        features['transition_point'] = largest_drop_idx / len(series)  # Normalized time
        features['transition_magnitude'] = abs(diff.loc[largest_drop_idx])
    else:
        features['transition_point'] = 1.0  # No transition found
        features['transition_magnitude'] = 0

    # Trend analysis
    x = np.arange(len(series))
    slope, intercept, _, _, _ = stats.linregress(x, series.values)
    features['trend_slope'] = slope

    return features

def process_all_timeseries(timeseries_dir, product_ids, target_column):
    """Process all time series and extract features"""
    from src.data_processing.preprocessing import clean_timeseries_data
    
    all_features = []
    successful = 0
    failed = []

    print(f"Processing {len(product_ids)} products...")

    for i, product_id in enumerate(product_ids):
        # Progress indicator
        if i % 50 == 0:
            print(f"Progress: {i}/{len(product_ids)}")

        try:
            # Load file
            filepath = os.path.join(timeseries_dir, f"{product_id}.txt")
            df = pd.read_csv(filepath)

            # Clean data
            df = clean_timeseries_data(df)

            # Extract features if target column exists
            if target_column in df.columns:
                features = extract_timeseries_features(df[target_column], product_id)
                all_features.append(features)
                successful += 1
            else:
                failed.append(product_id)
        except Exception as e:
            failed.append(product_id)
            print(f"Error processing {product_id}: {e}")

    print(f"\nProcessed {successful} products successfully")
    print(f"Failed to process {len(failed)} products")

    # Convert to DataFrame
    return pd.DataFrame(all_features)

def combine_with_product_info(ts_features, product_info):
    """Combine time series features with product information"""
    # Ensure IDs are strings for proper merging
    ts_features['product_id'] = ts_features['product_id'].astype(str)
    product_info['IDENTIFIANT'] = product_info['IDENTIFIANT'].astype(str)

    # Merge dataframes
    combined = pd.merge(
        ts_features,
        product_info,
        left_on='product_id',
        right_on='IDENTIFIANT',
        how='inner'
    )

    # Drop duplicate ID column
    if 'IDENTIFIANT' in combined.columns:
        combined = combined.drop('IDENTIFIANT', axis=1)

    print(f"Combined data shape: {combined.shape}")
    print(f"Products with complete data: {len(combined)}")

    return combined

def save_combined_data(df, output_dir='./results'):
    """Save the combined data for clustering"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'combined_data.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved combined data to {filepath}")
    return filepath