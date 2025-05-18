# src/data_processing/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_product_info(df):
    """Remove unnecessary columns from product info dataframe"""
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()

    # Remove Unnamed: 0 column if present
    if 'Unnamed: 0' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop('Unnamed: 0', axis=1)
        print("Removed 'Unnamed: 0' column")

    # Ensure ID column is string
    df_cleaned['IDENTIFIANT'] = df_cleaned['IDENTIFIANT'].astype(str)

    print(f"Cleaned product info shape: {df_cleaned.shape}")
    return df_cleaned

def clean_timeseries_data(df):
    """Clean time series data by removing unnecessary columns"""
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Remove Unnamed columns if present
    unnamed_cols = [col for col in cleaned_df.columns if 'Unnamed' in col]
    if unnamed_cols:
        cleaned_df = cleaned_df.drop(unnamed_cols, axis=1)
        print(f"Removed {len(unnamed_cols)} unnamed column(s)")

    return cleaned_df

def fix_time_column(df):
    """Convert time column to datetime format"""
    # Create a copy to avoid modifying the original
    df_with_time = df.copy()

    if 'time' in df_with_time.columns:
        # Sample time value for diagnosis
        sample_time = df_with_time['time'].iloc[0]
        print(f"Sample time value: {sample_time}")

        try:
            # Try European format (day first)
            df_with_time['time'] = pd.to_datetime(df_with_time['time'], dayfirst=True)
            print("Converted time column to datetime (dayfirst=True)")
        except Exception as e:
            print(f"Error converting time column: {e}")

    return df_with_time

def find_target_column(df, target_name='Pinceur Sup Mesure de courant'):
    """Find target column in time series data and its stats"""
    if target_name in df.columns:
        print(f"Found target column: '{target_name}'")
        print(f"\nBasic statistics for target column:")
        print(df[target_name].describe())
        return target_name
    else:
        print(f"Target column '{target_name}' not found")
        return None

def handle_missing_values(df):
    """Identify and handle missing values"""
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]

    if len(missing_cols) > 0:
        print("\nColumns with missing values:")
        print(missing_cols)

        # Fill missing values with column means
        for col in missing_cols.index:
            df[col] = df[col].fillna(df[col].mean())
            print(f"Filled missing values in {col} with mean")
    else:
        print("\nNo missing values found in the dataset.")

    return df

def select_and_scale_features(df):
    """Select relevant features and scale them for clustering"""
    # Time series features
    ts_features = [
        'mean', 'std', 'min', 'max', 'range',
        'high_state_mean', 'low_state_mean', 'high_state_ratio',
        'high_state_std', 'low_state_std', 'transition_point',
        'transition_magnitude', 'trend_slope'
    ]

    # Physical properties
    physical_features = ['EPAI_COILC', 'LARGCOILVISC']

    # Get all chemical elements (all columns except ID, time series features, and physical properties)
    exclude_cols = ['product_id'] + ts_features + physical_features
    chemical_features = [col for col in df.columns if col not in exclude_cols]

    # Combine all features
    selected_features = ts_features + physical_features + chemical_features

    # Create feature matrix
    X = df[selected_features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to DataFrame for better interpretability
    X_scaled_df = pd.DataFrame(
        data=X_scaled,
        columns=selected_features,
        index=X.index
    )

    print(f"Selected {len(selected_features)} features for clustering")
    print(f"Feature matrix shape: {X_scaled.shape}")

    return X, X_scaled, X_scaled_df, selected_features, scaler

def analyze_product_info(df):
    """Analyze product information data"""
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)

    # Get unique products
    unique_products = df['IDENTIFIANT'].nunique()
    product_ids = df['IDENTIFIANT'].tolist()

    print(f"\nTotal unique products: {unique_products}")
    print(f"Sample product IDs: {product_ids[:5]}")

    # Basic statistics for dimensions
    print("\nStatistics for product dimensions:")
    print(df[['EPAI_COILC', 'LARGCOILVISC']].describe())

    return {
        'missing_values': missing_values,
        'unique_products': unique_products,
        'product_ids': product_ids
    }