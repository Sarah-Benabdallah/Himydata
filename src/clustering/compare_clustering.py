def compare_clustering_methods(clustering_results):
    """Compare different clustering methods with basic metrics"""
    # Extract methods and scores
    methods = list(clustering_results.keys())
    silhouette_scores = [result['silhouette'] for method, result in clustering_results.items()]

    # Display comparison table
    comparison_data = []
    for method, result in clustering_results.items():
        method_info = {
            'Method': method,
            'Silhouette Score': result['silhouette'],
            'Number of Clusters': result.get('n_clusters', 'N/A')
        }

        # Add DBSCAN-specific info
        if method == 'DBSCAN':
            method_info['Epsilon'] = result.get('epsilon', 'N/A')
            method_info['Min Samples'] = result.get('min_samples', 'N/A')
            method_info['Noise Points'] = result.get('n_noise', 'N/A')

        # Add GMM-specific info
        if method == 'GMM':
            method_info['Covariance Type'] = result.get('covariance_type', 'N/A')

        comparison_data.append(method_info)

    # Display comparison
    print("\nClustering Methods Comparison:")
    for method_info in comparison_data:
        for key, value in method_info.items():
            print(f"{key}: {value}")
        print("-" * 50)

    # Find best method by silhouette score
    valid_methods = {k: v for k, v in clustering_results.items() if v['silhouette'] > 0}
    if valid_methods:
        best_method = max(valid_methods.items(), key=lambda x: x[1]['silhouette'])[0]
        print(f"\nBest clustering method based on silhouette score: {best_method} "
              f"(Score: {clustering_results[best_method]['silhouette']:.4f})")
    else:
        best_method = None
        print("\nNo valid clustering method found based on silhouette score.")

    # Print cluster sizes for each method
    print("\nCluster sizes by method:")
    for method, result in clustering_results.items():
        labels = result['labels']

        # Handle DBSCAN specially due to noise points
        if method == 'DBSCAN':
            from collections import Counter
            value_counts = Counter(labels)
            # Check if noise points exist
            if -1 in value_counts:
                noise_count = value_counts[-1]
                clusters_count = {k: v for k, v in value_counts.items() if k != -1}
                cluster_sizes = ', '.join([f"Cluster {i}: {count}" for i, count in clusters_count.items()])
                print(f"{method}: {cluster_sizes}, Noise: {noise_count}")
            else:
                cluster_sizes = ', '.join([f"Cluster {i}: {count}" for i, count in value_counts.items()])
                print(f"{method}: {cluster_sizes}")
        else:
            from collections import Counter
            value_counts = Counter(labels)
            cluster_sizes = ', '.join([f"Cluster {i}: {count}" for i, count in sorted(value_counts.items())])
            print(f"{method}: {cluster_sizes}")

    return best_method