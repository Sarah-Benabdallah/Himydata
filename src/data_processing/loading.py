# src/data_processing/loading.py

import os
import pandas as pd
from glob import glob
from typing import Dict, List, Tuple, Optional

def setup_data_paths(base_dir='./data'):
    """Set up and validate data paths"""
    paths = {
        'data_dir': base_dir,
        'timeseries_dir': os.path.join(base_dir, 'timeseries'),
        'product_info_file': os.path.join(base_dir, 'info_produits.txt')
    }

    # Check if paths exist
    for name, path in paths.items():
        exists = os.path.exists(path)
        print(f"{name} exists: {exists} - {path}")

    return paths

def load_product_info(filepath):
    """Load product information data"""
    df = pd.read_csv(filepath)
    print(f"Product info shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    return df

def load_timeseries_file(filepath):
    """Load a single time series file"""
    df = pd.read_csv(filepath)
    print(f"Loaded: {os.path.basename(filepath)}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:")
    print(df.dtypes)
    return df

def count_timeseries_files(timeseries_dir):
    """Count time series files in directory"""
    timeseries_files = glob(os.path.join(timeseries_dir, '*.txt'))
    print(f"\nFound {len(timeseries_files)} time series files")
    print("\nSample filenames:")
    for file in timeseries_files[:5]:
        print(f"  - {os.path.basename(file)}")
    return timeseries_files

def load_combined_data(filepath='./results/combined_data.csv'):
    """Load the combined dataset prepared for clustering"""
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape}")
    print(f"Number of products: {len(df)}")
    return df

