# src/visualization/exploratory_visualization.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_product_info(df):
    """Create visualizations for product information"""
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Thickness distribution
    axes[0, 0].hist(df['EPAI_COILC'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Product Thickness (EPAI_COILC) Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Thickness (mm)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)

    # Width distribution
    axes[0, 1].hist(df['LARGCOILVISC'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Product Width (LARGCOILVISC) Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)

    # Thickness vs Width scatter plot
    axes[1, 0].scatter(df['EPAI_COILC'], df['LARGCOILVISC'], alpha=0.6, edgecolor='black')
    axes[1, 0].set_title('Thickness vs Width', fontsize=14)
    axes[1, 0].set_xlabel('Thickness (mm)')
    axes[1, 0].set_ylabel('Width')
    axes[1, 0].grid(True, alpha=0.3)

    # Chemical composition boxplot
    chemical_cols = ['Carbone', 'Chrome', 'Manganese', 'Silicium']
    df[chemical_cols].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('Key Chemical Composition', fontsize=14)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/product_info_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_timeseries(df, column, title_prefix=""):
    """Plot time series data"""
    if column not in df.columns:
        print(f"Column '{column}' not found")
        return

    os.makedirs('results', exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot full time series
    ax1.plot(df.index, df[column], linewidth=1, color='navy')
    ax1.set_title(f'{title_prefix} Full Time Series - {column}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Plot first 150 points for detail
    ax2.plot(df.index[:150], df[column][:150], linewidth=1.5, color='darkblue')
    ax2.set_title(f'{title_prefix} First 150 Points - {column}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/timeseries_{title_prefix.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_distribution(df, column, bins=50):
    """Plot distribution of a column"""
    if column not in df.columns:
        print(f"Column '{column}' not found")
        return

    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.hist(df[column], bins=bins, color='darkblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/distribution_{column.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(X):
    """Plot correlation matrix of features"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(16, 14))
    correlation_matrix = X.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nIdentifying highly correlated features (|r| > 0.9):")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col1, col2, corr_matrix.loc[col1, col2])
                  for col1 in corr_matrix.columns
                  for col2 in corr_matrix.columns
                  if corr_matrix.loc[col1, col2] > 0.9 and col1 != col2]

    if high_corr:
        print("Highly correlated features:")
        for col1, col2, val in high_corr:
            print(f"{col1} & {col2}: {val:.3f}")
    else:
        print("No feature pairs with correlation > 0.9 found.")

def compare_multiple_timeseries(timeseries_dir, product_ids, target_column='Pinceur Sup Mesure de courant', max_products=4):
    """Compare multiple time series products and visualize their differences"""
    from src.data_processing.loading import load_timeseries_file
    from src.data_processing.preprocessing import clean_timeseries_data, fix_time_column
    
    # Limit number of products to compare
    if len(product_ids) > max_products:
        print(f"Limiting comparison to {max_products} products")
        product_ids = product_ids[:max_products]

    # Storage for processed data
    all_data = {}
    all_features = {}

    # Process each product
    for product_id in product_ids:
        try:
            # Construct filepath
            filepath = os.path.join(timeseries_dir, f"{product_id}.txt")

            # Load and process the data
            print(f"\nProcessing: {product_id}")
            ts_data = load_timeseries_file(filepath)

            # Clean data
            ts_data = clean_timeseries_data(ts_data)
            ts_data = fix_time_column(ts_data)

            # Find target column
            if target_column in ts_data.columns:
                # Store data
                all_data[product_id] = ts_data

                # Extract basic features
                features = {
                    'mean': ts_data[target_column].mean(),
                    'std': ts_data[target_column].std(),
                    'min': ts_data[target_column].min(),
                    'max': ts_data[target_column].max(),
                    'range': ts_data[target_column].max() - ts_data[target_column].min(),
                    'skew': ts_data[target_column].skew(),
                    'kurtosis': ts_data[target_column].kurtosis()
                }
                all_features[product_id] = features
                print(f"Features extracted for {product_id}")
            else:
                print(f"Target column not found in {product_id}")
        except Exception as e:
            print(f"Error processing {product_id}: {e}")

    # Make sure we have data to compare
    if not all_data:
        print("No valid data found for comparison")
        return None

    # COMPARISON VISUALIZATIONS
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(15, 10))

    # 1. Compare time series patterns
    plt.subplot(2, 2, 1)

    # Determine common length for comparison
    min_length = min(len(df[target_column]) for df in all_data.values())
    sample_length = min(300, min_length)  # Limit to 300 points for visibility

    for product_id, df in all_data.items():
        plt.plot(df[target_column].values[:sample_length],
                label=product_id, alpha=0.7, linewidth=1.5)

    plt.title('Time Series Patterns Comparison', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Compare distributions
    plt.subplot(2, 2, 2)
    for product_id, df in all_data.items():
        plt.hist(df[target_column].values, bins=30, alpha=0.4, label=product_id)

    plt.title('Value Distribution Comparison', fontsize=14)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Compare boxplots
    plt.subplot(2, 2, 3)
    boxplot_data = [df[target_column].values for df in all_data.values()]
    plt.boxplot(boxplot_data)
    plt.xticks(range(1, len(all_data) + 1), all_data.keys())
    plt.title('Statistical Distribution Comparison', fontsize=14)
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)

    # 4. Compare feature values
    plt.subplot(2, 2, 4)
    feature_names = ['mean', 'std', 'range']
    x = np.arange(len(feature_names))
    width = 0.8 / len(all_data)

    for i, (product_id, features) in enumerate(all_features.items()):
        values = [features.get(feat, 0) for feat in feature_names]
        plt.bar(x + i*width, values, width, label=product_id)

    plt.title('Feature Comparison', fontsize=14)
    plt.xticks(x + width * (len(all_data) - 1) / 2, feature_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/timeseries_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create detailed feature comparison table
    feature_df = pd.DataFrame(all_features).T
    print("\nFeature Comparison Table:")
    print(feature_df.round(2))

    # Process pattern analysis
    print("\nProcess Pattern Analysis:")
    patterns = {}

    for product_id, df in all_data.items():
        series = df[target_column]

        # Define high and low states based on distribution
        threshold = 0  # Threshold between positive and negative values
        high_state = series[series > threshold]
        low_state = series[series <= threshold]

        # Calculate process patterns
        patterns[product_id] = {
            'high_state_ratio': len(high_state) / len(series),
            'high_state_mean': high_state.mean() if len(high_state) > 0 else 0,
            'low_state_mean': low_state.mean() if len(low_state) > 0 else 0
        }

    patterns_df = pd.DataFrame(patterns).T
    print(patterns_df.round(3))

    return {
        'data': all_data,
        'features': all_features,
        'patterns': patterns
    }

def analyze_process_patterns(comparison_results, target_column='Pinceur Sup Mesure de courant'):
    """Analyze process patterns in more detail"""
    if not comparison_results or 'data' not in comparison_results:
        print("No comparison results available")
        return

    os.makedirs('results', exist_ok=True)
    
    # Normalize time series for shape comparison
    plt.figure(figsize=(12, 6))

    min_length = min(len(df[target_column]) for df in comparison_results['data'].values())
    sample_length = min(300, min_length)

    for product_id, df in comparison_results['data'].items():
        # Normalize series
        series = df[target_column]
        normalized = (series - series.mean()) / series.std()
        plt.plot(normalized.values[:sample_length], label=product_id)

    plt.title('Normalized Time Series Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/normalized_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Process state transitions
    print("State Transition Analysis:")
    transitions = {}

    for product_id, df in comparison_results['data'].items():
        series = df[target_column]

        # Find major transitions
        diff = series.diff()
        large_drops = diff[diff < -50]  # Significant drops

        if not large_drops.empty:
            largest_drop_idx = large_drops.idxmin()
            largest_drop_time = largest_drop_idx / len(series)  # Normalized time
            largest_drop_magnitude = abs(diff.loc[largest_drop_idx])
        else:
            largest_drop_idx = None
            largest_drop_time = None
            largest_drop_magnitude = None

        transitions[product_id] = {
            'transition_point': largest_drop_time,
            'transition_magnitude': largest_drop_magnitude
        }

    transitions_df = pd.DataFrame(transitions).T
    print(transitions_df)

    # Annotated time series with transition points
    plt.figure(figsize=(15, 8))

    for product_id, df in comparison_results['data'].items():
        series = df[target_column]
        plt.plot(series.values[:sample_length], label=product_id, alpha=0.7)

        # Mark transition point
        transition_point = transitions[product_id]['transition_point']
        if transition_point is not None:
            transition_idx = int(transition_point * len(series))
            if transition_idx < sample_length:
                plt.axvline(x=transition_idx, linestyle='--', alpha=0.5,
                           color=plt.gca().lines[-1].get_color())

    plt.title('Time Series with Transition Points', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/time_series_transitions.png', dpi=300, bbox_inches='tight')
    plt.show()