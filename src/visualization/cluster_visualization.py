# src/visualization/cluster_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

def apply_pca(X_scaled, n_components=2):
    """Apply PCA for dimensionality reduction and visualization"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    return pca, X_pca

def plot_pca_results(X_pca, pca):
    """Plot PCA results"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)
    plt.title('PCA ofF Product Data', fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('results/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_optimal_clusters(metrics):
    """Plot metrics for determining optimal number of clusters"""
    os.makedirs('results', exist_ok=True)
    
    k_range = metrics['k_range']
    
    # Create a comprehensive plot of all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Elbow method (inertia)
    axes[0, 0].plot(k_range, metrics['inertia'], 'o-', color='blue')
    axes[0, 0].set_xlabel('Number of clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method for Optimal k')
    axes[0, 0].grid(True, alpha=0.3)

    # Silhouette score
    axes[0, 1].plot(k_range, metrics['silhouette_scores'], 'o-', color='green')
    axes[0, 1].set_xlabel('Number of clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Analysis (higher is better)')
    axes[0, 1].grid(True, alpha=0.3)

    # Davies-Bouldin index
    axes[1, 0].plot(k_range, metrics['davies_bouldin_scores'], 'o-', color='red')
    axes[1, 0].set_xlabel('Number of clusters (k)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Index (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)

    # Calinski-Harabasz index
    axes[1, 1].plot(k_range, metrics['calinski_harabasz_scores'], 'o-', color='purple')
    axes[1, 1].set_xlabel('Number of clusters (k)')
    axes[1, 1].set_ylabel('Calinski-Harabasz Index')
    axes[1, 1].set_title('Calinski-Harabasz Index (higher is better)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/optimal_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_kmeans(X_scaled, kmeans_result, selected_features, scaler=None, title_prefix="K-means"):
    """Visualize K-means clustering results"""
    os.makedirs('results/kmeans', exist_ok=True)
    
    labels = kmeans_result['labels']

    # 1. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2']
    )
    pca_df['cluster'] = labels

    # Plot PCA
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['cluster'],
        cmap='viridis',
        s=80,
        alpha=0.7,
        edgecolor='w'
    )

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         title="Clusters",
                         loc="upper right")
    plt.gca().add_artist(legend1)

    # Add titles and labels
    plt.title(f'{title_prefix} Clustering (Silhouette: {kmeans_result["silhouette"]:.4f})', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/kmeans/kmeans_pca_{kmeans_result["n_clusters"]}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Cluster Centers Analysis
    centers = kmeans_result['model'].cluster_centers_

    # Create a dataframe of cluster centers
    centers_df = pd.DataFrame(centers, columns=selected_features)

    # Transform centers to original scale if a scaler is provided
    if scaler is not None:
        # Use the inverse_transform method of the scaler
        centers_original = scaler.inverse_transform(centers)
        centers_df_original = pd.DataFrame(centers_original, columns=selected_features)

        # Display both scaled and original centers
        print("\nCluster Centers (Original Scale):")
        print(centers_df_original.round(2))

        # Create and show heatmap for original scale centers
        plt.figure(figsize=(16, 10))
        # Normalize for visualization purposes
        centers_original_scaled = (centers_df_original - centers_df_original.min()) / (centers_df_original.max() - centers_df_original.min())
        sns.heatmap(
            centers_original_scaled.T,
            annot=False,
            cmap="YlGnBu",
            linewidths=0.5
        )
        plt.title(f'{title_prefix} Cluster Centers (Original Scale, Normalized for Visualization)', fontsize=14)
        plt.ylabel('Features')
        plt.xlabel('Cluster')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'results/kmeans/kmeans_centers_original_{kmeans_result["n_clusters"]}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Scale the centers for better visualization (standardized centers)
    print("\nCluster Centers (Standardized Scale):")
    print(centers_df.round(2))

    centers_scaled = (centers_df - centers_df.min()) / (centers_df.max() - centers_df.min())

    # Heatmap of cluster centers (standardized)
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        centers_scaled.T,
        annot=False,
        cmap="YlGnBu",
        linewidths=0.5
    )
    plt.title(f'{title_prefix} Cluster Centers (Standardized Scale, Normalized for Visualization)', fontsize=14)
    plt.ylabel('Features')
    plt.xlabel('Cluster')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/kmeans/kmeans_centers_standardized_{kmeans_result["n_clusters"]}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title(f'{title_prefix} Cluster Size Distribution', fontsize=14)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Products')
    plt.grid(True, alpha=0.3)

    # Add count labels
    for i, count in enumerate(cluster_counts):
        plt.text(i, count + 1, str(count), ha='center')

    plt.tight_layout()
    plt.savefig(f'results/kmeans/kmeans_distribution_{kmeans_result["n_clusters"]}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_gmm(X_scaled, gmm_result, selected_features, title_prefix="GMM"):
    """Visualize Gaussian Mixture Model clustering results"""
    os.makedirs('results/gmm', exist_ok=True)
    
    labels = gmm_result['labels']
    gmm_model = gmm_result['model']
    covariance_type = gmm_result['covariance_type']

    # 1. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2']
    )
    pca_df['cluster'] = labels

    # Plot PCA
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['cluster'],
        cmap='viridis',
        s=80,
        alpha=0.7,
        edgecolor='w'
    )

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         title="Clusters",
                         loc="upper right")
    plt.gca().add_artist(legend1)

    # Add titles and labels
    plt.title(f'{title_prefix} Clustering (Silhouette: {gmm_result["silhouette"]:.4f}, Covariance: {covariance_type})', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/gmm/gmm_pca_{gmm_result["n_clusters"]}_{covariance_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Cluster Centers Analysis
    means = gmm_model.means_

    # Create a dataframe of cluster centers
    means_df = pd.DataFrame(means, columns=selected_features)

    # Display means
    print("\nGMM Component Means:")
    print(means_df.round(2))

    # Scale the means for better visualization
    means_scaled = (means_df - means_df.min()) / (means_df.max() - means_df.min())

    # Heatmap of cluster centers
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        means_scaled.T,
        annot=False,
        cmap="YlGnBu",
        linewidths=0.5
    )
    plt.title(f'{title_prefix} Component Means (Scaled 0-1)', fontsize=14)
    plt.ylabel('Features')
    plt.xlabel('Cluster')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/gmm/gmm_means_{gmm_result["n_clusters"]}_{covariance_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title(f'{title_prefix} Cluster Size Distribution', fontsize=14)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Products')
    plt.grid(True, alpha=0.3)

    # Add count labels
    for i, count in enumerate(cluster_counts):
        plt.text(i, count + 1, str(count), ha='center')

    plt.tight_layout()
    plt.savefig(f'results/gmm/gmm_distribution_{gmm_result["n_clusters"]}_{covariance_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Probability Analysis
    # We can visualize how confident GMM is about its assignments
    probabilities = gmm_model.predict_proba(X_scaled)

    # Check the maximum probability for each sample
    max_probs = np.max(probabilities, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=20, color='skyblue', edgecolor='black')
    plt.title('GMM Assignment Confidence', fontsize=14)
    plt.xlabel('Maximum Probability')
    plt.ylabel('Number of Products')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/gmm/gmm_probabilities_{gmm_result["n_clusters"]}_{covariance_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_dbscan(X_scaled, dbscan_result, selected_features, title_prefix="DBSCAN"):
    """Visualize DBSCAN Clustering results"""
    os.makedirs('results/dbscan', exist_ok=True)
    
    labels = dbscan_result['labels']
    epsilon = dbscan_result['epsilon']
    min_samples = dbscan_result['min_samples']
    n_clusters = dbscan_result['n_clusters']
    n_noise = dbscan_result['n_noise']

    # 1. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2']
    )
    pca_df['cluster'] = labels

    # Plot PCA
    plt.figure(figsize=(12, 10))

    # First plot noise points in black with 'x' marker
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        plt.scatter(
            X_pca[noise_mask, 0],
            X_pca[noise_mask, 1],
            c='black',
            marker='x',
            s=100,
            alpha=0.5,
            label='Noise'
        )

    # Then plot actual clusters
    non_noise_mask = ~noise_mask
    if np.any(non_noise_mask):
        scatter = plt.scatter(
            X_pca[non_noise_mask, 0],
            X_pca[non_noise_mask, 1],
            c=labels[non_noise_mask],
            cmap='viridis',
            s=80,
            alpha=0.7,
            edgecolor='w'
        )

        # Add legend for clusters
        legend1 = plt.legend(*scatter.legend_elements(),
                           title="Clusters",
                           loc="upper right")
        plt.gca().add_artist(legend1)

    # Add a separate legend for noise
    if np.any(noise_mask):
        plt.legend(loc="lower left")

    # Add titles and labels
    if 'silhouette' in dbscan_result and dbscan_result['n_clusters'] > 1:
        plt.title(f'{title_prefix} Clustering (Silhouette: {dbscan_result["silhouette"]:.4f}, '
                f'Epsilon: {epsilon:.4f}, Min Samples: {min_samples})', fontsize=14)
    else:
        plt.title(f'{title_prefix} Clustering (Epsilon: {epsilon:.4f}, Min Samples: {min_samples})', fontsize=14)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/dbscan/dbscan_pca_eps{epsilon:.4f}_min{min_samples}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Only continue with the remaining visualizations if we have clusters
    if n_clusters > 1:
        # 2. Cluster Characteristics Analysis
        # For DBSCAN, we need to calculate cluster centers manually
        # Convert X_scaled to DataFrame for easier manipulation
        X_df = pd.DataFrame(X_scaled, columns=selected_features)
        X_df['cluster'] = labels

        # Filter out noise points (-1)
        X_df_no_noise = X_df[X_df['cluster'] >= 0]

        # Calculate cluster means in scaled space
        centers_df = X_df_no_noise.groupby('cluster')[selected_features].mean()

        # Display scaled centers
        print("\nCluster Centers (Standardized Scale):")
        print(centers_df.round(2))

        # Scale the centers for better visualization (in standardized space)
        centers_scaled = (centers_df - centers_df.min()) / (centers_df.max() - centers_df.min())

        # Heatmap of cluster means
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            centers_scaled.T,
            annot=False,
            cmap="YlGnBu",
            linewidths=0.5
        )
        plt.title(f'{title_prefix} Cluster Characteristics (Standardized Scale, Normalized for Visualization)', fontsize=14)
        plt.ylabel('Features')
        plt.xlabel('Cluster')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'results/dbscan/dbscan_centers_eps{epsilon:.4f}_min{min_samples}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Cluster Size Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = pd.Series(labels).value_counts().sort_index()

        # Create bar plot - handle noise separately
        ax = plt.subplot()

        # Normal clusters (non-noise)
        non_noise_counts = cluster_counts[cluster_counts.index >= 0]
        non_noise_counts.plot(kind='bar', color='skyblue', ax=ax)

        # Noise points (if any)
        if -1 in cluster_counts.index:
            noise_count = cluster_counts[-1]
            # Add as a separate bar at the end with different color
            ax.bar(['Noise'], [noise_count], color='lightgray')

        # Add count labels
        for i, count in enumerate(non_noise_counts):
            ax.text(i, count + 1, str(count), ha='center')

        if -1 in cluster_counts.index:
            ax.text(len(non_noise_counts), noise_count + 1, str(noise_count), ha='center')

        plt.title(f'{title_prefix} Cluster Size Distribution', fontsize=14)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Products')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/dbscan/dbscan_distribution_eps{epsilon:.4f}_min{min_samples}.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("DBSCAN did not find multiple clusters. Skipping cluster analysis visualizations.")