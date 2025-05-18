# scripts/run_clustering.py

import argparse

from src.clustering.compare_clustering import compare_clustering_methods
from src.data_processing.loading import load_combined_data
from src.data_processing.preprocessing import handle_missing_values, select_and_scale_features
from src.visualization.exploratory_visualization import plot_correlation_matrix
from src.visualization.cluster_visualization import apply_pca, plot_pca_results, plot_optimal_clusters, \
    visualize_kmeans, visualize_gmm, visualize_dbscan
from src.clustering.kmeans import determine_optimal_clusters, apply_kmeans
from src.clustering.gmm import select_best_covariance_type, apply_gmm
from src.clustering.dbscan import estimate_dbscan_parameters, apply_dbscan


def display_menu():
    """Display the interactive clustering menu"""
    print("\n" + "=" * 50)
    print(" PRODUCT CLUSTERING MENU")
    print("=" * 50)
    print("1. Load and prepare data")
    print("2. Visualize data (PCA & correlation)")
    print("3. Determine optimal number of clusters")
    print("4. Run K-means clustering")
    print("5. Run Gaussian Mixture Model (GMM) clustering")
    print("6. Run DBSCAN clustering")
    print("7. Compare all clustering methods")
    print("8. Run complete clustering pipeline")
    print("0. Exit")
    print("-" * 50)

    choice = input("Enter your choice (0-8): ")
    return choice


def load_and_prepare_data():
    """Load and prepare data for clustering"""
    print("\nLoading and preparing data...")

    # Ask for data source
    data_path = input("Enter path to combined data (default: results/combined_data.csv): ").strip()
    if not data_path:
        data_path = 'results/combined_data.csv'

    # Load data
    try:
        combined_data = load_combined_data(data_path)
        print(f"Loaded {len(combined_data)} records from {data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

    # Handle missing values
    data = handle_missing_values(combined_data)

    # Select and scale features
    X, X_scaled, X_scaled_df, selected_features, scaler = select_and_scale_features(data)

    print(f"Data prepared with {X.shape[1]} features")

    return X, X_scaled, X_scaled_df, selected_features, scaler


def visualize_data(X, X_scaled):
    """Visualize data using PCA and correlation matrix"""
    print("\nVisualizing data...")

    # Plot correlation matrix
    plot_correlation_matrix(X)

    # Apply PCA for dimensionality reduction and visualization
    pca, X_pca = apply_pca(X_scaled)
    plot_pca_results(X_pca, pca)

    return pca, X_pca


def find_optimal_clusters(X_scaled):
    """Determine the optimal number of clusters"""
    print("\nFinding optimal number of clusters...")

    # Ask for maximum clusters to consider
    max_clusters_input = input("Maximum number of clusters to consider (default: 10): ").strip()

    try:
        max_clusters = int(max_clusters_input) if max_clusters_input else 10
        max_clusters = max(2, min(max_clusters, 20))  # Between 2 and 20
    except ValueError:
        print("Invalid input. Using default value of 10.")
        max_clusters = 10

    # Determine optimal number of clusters
    optimal_k, metrics = determine_optimal_clusters(X_scaled, max_clusters)

    # Plot optimal clusters
    plot_optimal_clusters(metrics)

    # Allow user to override optimal k
    override = input(f"\nUse recommended k={optimal_k}? (y/n, default: y): ").strip().lower()

    if override == 'n':
        custom_k = input("Enter preferred number of clusters: ").strip()
        try:
            k = int(custom_k)
            if 2 <= k <= max_clusters:
                print(f"Using custom value k={k}")
                optimal_k = k
            else:
                print(f"Value must be between 2 and {max_clusters}. Using recommended k={optimal_k}")
        except ValueError:
            print(f"Invalid input. Using recommended k={optimal_k}")

    return optimal_k, metrics


def run_kmeans_clustering(X_scaled, selected_features, scaler, k=None):
    """Run K-means clustering"""
    print("\nRunning K-means clustering...")

    # Get number of clusters if not provided
    if k is None:
        clusters_input = input("Number of clusters (default: 2): ").strip()

        try:
            k = int(clusters_input) if clusters_input else 2
            k = max(2, k)  # At least 2 clusters
        except ValueError:
            print("Invalid input. Using default value of 2 clusters.")
            k = 2

    # Apply K-means clustering
    kmeans_result = apply_kmeans(X_scaled, k)

    # Visualize K-means results
    visualize_kmeans(X_scaled, kmeans_result, selected_features, scaler)

    return kmeans_result


def run_gmm_clustering(X_scaled, selected_features, k=None):
    """Run Gaussian Mixture Model clustering"""
    print("\nRunning Gaussian Mixture Model clustering...")

    # Get number of clusters if not provided
    if k is None:
        clusters_input = input("Number of components (default: 2): ").strip()

        try:
            k = int(clusters_input) if clusters_input else 2
            k = max(2, k)  # At least 2 components
        except ValueError:
            print("Invalid input. Using default value of 2 components.")
            k = 2

    # Ask if user wants to select covariance type manually
    cov_choice = input("Select covariance type manually? (y/n, default: n): ").strip().lower()

    if cov_choice == 'y':
        print("\nCovariance type options:")
        print("1. Full - Each component has its own general covariance matrix")
        print("2. Tied - All components share the same general covariance matrix")
        print("3. Diag - Each component has its own diagonal covariance matrix")
        print("4. Spherical - Each component has its own single variance")

        cov_option = input("Select option (1-4): ").strip()

        cov_types = {
            '1': 'full',
            '2': 'tied',
            '3': 'diag',
            '4': 'spherical'
        }

        if cov_option in cov_types:
            covariance_type = cov_types[cov_option]
            print(f"Using covariance type: {covariance_type}")
        else:
            print("Invalid selection. Finding optimal covariance type automatically.")
            covariance_type = select_best_covariance_type(X_scaled, k)
    else:
        # Find optimal covariance type
        covariance_type = select_best_covariance_type(X_scaled, k)

    # Apply GMM clustering
    gmm_result = apply_gmm(X_scaled, k, covariance_type=covariance_type)

    # Visualize GMM results
    visualize_gmm(X_scaled, gmm_result, selected_features)

    return gmm_result


def run_dbscan_clustering(X_scaled, selected_features):
    """Run DBSCAN clustering"""
    print("\nRunning DBSCAN clustering...")

    # Ask if user wants to set parameters manually
    manual_params = input("Set DBSCAN parameters manually? (y/n, default: n): ").strip().lower()

    if manual_params == 'y':
        # Manual parameter setting
        epsilon_input = input("Enter epsilon value (neighborhood distance): ").strip()
        min_samples_input = input("Enter min_samples value (default: 5): ").strip()

        try:
            epsilon = float(epsilon_input)
            min_samples = int(min_samples_input) if min_samples_input else 5
            min_samples = max(1, min_samples)

            print(f"Using manual parameters: epsilon={epsilon:.4f}, min_samples={min_samples}")
        except ValueError:
            print("Invalid input. Estimating parameters automatically.")
            epsilon, min_samples = estimate_dbscan_parameters(X_scaled)
    else:
        # Estimate parameters
        epsilon, min_samples = estimate_dbscan_parameters(X_scaled)

    # Apply DBSCAN clustering
    dbscan_result = apply_dbscan(X_scaled, epsilon, min_samples)

    # Visualize DBSCAN results
    visualize_dbscan(X_scaled, dbscan_result, selected_features)

    return dbscan_result


def compare_methods(clustering_results):
    """Compare different clustering methods"""
    print("\nComparing clustering methods...")

    # Run comparison
    best_method = compare_clustering_methods(clustering_results)

    return best_method


def run_clustering_interactive():
    """Run the clustering analysis pipeline interactively"""
    print("Starting interactive clustering analysis...")

    # Initialize variables for sharing between functions
    X = None
    X_scaled = None
    X_scaled_df = None
    selected_features = None
    scaler = None
    optimal_k = None
    metrics = None
    pca = None
    X_pca = None
    kmeans_result = None
    gmm_result = None
    dbscan_result = None
    clustering_results = None

    while True:
        choice = display_menu()

        if choice == '0':
            print("\nExiting clustering analysis. Goodbye!")
            break

        elif choice == '1':
            # Load and prepare data
            X, X_scaled, X_scaled_df, selected_features, scaler = load_and_prepare_data()

        elif choice == '2':
            # Visualize data
            if X is None or X_scaled is None:
                print("\nData not loaded yet. Please load data first (option 1).")
            else:
                pca, X_pca = visualize_data(X, X_scaled)

        elif choice == '3':
            # Determine optimal clusters
            if X_scaled is None:
                print("\nData not loaded yet. Please load data first (option 1).")
            else:
                optimal_k, metrics = find_optimal_clusters(X_scaled)

        elif choice == '4':
            # Run K-means clustering
            if X_scaled is None:
                print("\nData not loaded yet. Please load data first (option 1).")
            else:
                # Use optimal_k if available, otherwise let user specify
                k = optimal_k if optimal_k is not None else None
                kmeans_result = run_kmeans_clustering(X_scaled, selected_features, scaler, k)

        elif choice == '5':
            # Run GMM clustering
            if X_scaled is None:
                print("\nData not loaded yet. Please load data first (option 1).")
            else:
                # Use optimal_k if available, otherwise let user specify
                k = optimal_k if optimal_k is not None else None
                gmm_result = run_gmm_clustering(X_scaled, selected_features, k)

        elif choice == '6':
            # Run DBSCAN clustering
            if X_scaled is None:
                print("\nData not loaded yet. Please load data first (option 1).")
            else:
                dbscan_result = run_dbscan_clustering(X_scaled, selected_features)

        elif choice == '7':
            # Compare all clustering methods
            if kmeans_result is None and gmm_result is None and dbscan_result is None:
                print("\nNo clustering methods have been run yet. Please run at least one clustering method first.")
            else:
                # Collect available results
                clustering_results = {}
                if kmeans_result is not None:
                    clustering_results['KMeans'] = kmeans_result
                if gmm_result is not None:
                    clustering_results['GMM'] = gmm_result
                if dbscan_result is not None:
                    clustering_results['DBSCAN'] = dbscan_result

                # Compare methods
                best_method = compare_methods(clustering_results)

        elif choice == '8':
            # Run complete pipeline
            print("\nRunning complete clustering pipeline...")
            clustering_results, best_method = run_clustering()

            # Extract results for future use
            X = clustering_results.get('X', X)
            X_scaled = clustering_results.get('X_scaled', X_scaled)
            selected_features = clustering_results.get('selected_features', selected_features)
            scaler = clustering_results.get('scaler', scaler)
            kmeans_result = clustering_results.get('KMeans', kmeans_result)
            gmm_result = clustering_results.get('GMM', gmm_result)
            dbscan_result = clustering_results.get('DBSCAN', dbscan_result)

        else:
            print("\nInvalid choice. Please select a number between 0 and 8.")

        # Pause before returning to menu
        input("\nPress Enter to continue...")

    return clustering_results


def run_clustering(combined_data=None):
    """Run the clustering analysis pipeline"""
    print("Starting clustering analysis...")

    # Load data if not provided
    if combined_data is None:
        combined_data = load_combined_data()

    # Handle missing values
    data = handle_missing_values(combined_data)

    # Select and scale features
    X, X_scaled, X_scaled_df, selected_features, scaler = select_and_scale_features(data)

    # Plot correlation matrix
    plot_correlation_matrix(X)

    # Apply PCA for dimensionality reduction and visualization
    pca, X_pca = apply_pca(X_scaled)
    plot_pca_results(X_pca, pca)

    # Determine optimal number of clusters
    optimal_k, metrics = determine_optimal_clusters(X_scaled)
    plot_optimal_clusters(metrics)

    # Apply K-means clustering
    kmeans_result = apply_kmeans(X_scaled, optimal_k)
    visualize_kmeans(X_scaled, kmeans_result, selected_features, scaler)

    # Apply GMM clustering
    best_cov_type = select_best_covariance_type(X_scaled, optimal_k)
    gmm_result = apply_gmm(X_scaled, optimal_k, covariance_type=best_cov_type)
    visualize_gmm(X_scaled, gmm_result, selected_features)

    # Apply DBSCAN clustering
    epsilon, min_samples = estimate_dbscan_parameters(X_scaled)
    dbscan_result = apply_dbscan(X_scaled, epsilon, min_samples)
    visualize_dbscan(X_scaled, dbscan_result, selected_features)

    # Compare clustering methods
    clustering_results = {
        'KMeans': kmeans_result,
        'GMM': gmm_result,
        'DBSCAN': dbscan_result
    }
    best_method = compare_clustering_methods(clustering_results)

    # Add data and preprocessing outputs to results
    clustering_results['X'] = X
    clustering_results['X_scaled'] = X_scaled
    clustering_results['selected_features'] = selected_features
    clustering_results['scaler'] = scaler

    print(f"Clustering analysis complete. Best method: {best_method}")
    return clustering_results, best_method


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Product Clustering Analysis')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode with menu')
    parser.add_argument('--data', type=str,
                        help='Path to combined data file')

    args = parser.parse_args()

    if args.interactive:
        # Run interactive clustering
        run_clustering_interactive()
    else:
        # Run full pipeline
        if args.data:
            combined_data = load_combined_data(args.data)
            run_clustering(combined_data)
        else:
            run_clustering()