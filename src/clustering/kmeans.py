# src/clustering/kmeans.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def determine_optimal_clusters(X_scaled, max_clusters=10):
    """Determine optimal number of clusters using multiple methods"""
    # Calculate metrics for different number of clusters
    k_range = range(2, max_clusters + 1)
    inertia = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for k in k_range:
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate metrics
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))

        print(f"K={k}: inertia={kmeans.inertia_:.2f}, "
              f"silhouette score={silhouette_scores[-1]:.4f}, "
              f"Davies-Bouldin={davies_bouldin_scores[-1]:.4f}, "
              f"Calinski-Harabasz={calinski_harabasz_scores[-1]:.2f}")

    # Find optimal k based on different metrics
    optimal_k_silhouette = k_range[silhouette_scores.index(max(silhouette_scores))]
    optimal_k_davies = k_range[davies_bouldin_scores.index(min(davies_bouldin_scores))]
    optimal_k_calinski = k_range[calinski_harabasz_scores.index(max(calinski_harabasz_scores))]

    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k_silhouette}")
    print(f"Optimal number of clusters based on Davies-Bouldin index: {optimal_k_davies}")
    print(f"Optimal number of clusters based on Calinski-Harabasz index: {optimal_k_calinski}")

    # Count frequency of each k
    k_counts = {k: 0 for k in k_range}
    k_counts[optimal_k_silhouette] += 1
    k_counts[optimal_k_davies] += 1
    k_counts[optimal_k_calinski] += 1

    # Find the most frequent k
    recommended_k = max(k_counts, key=k_counts.get)

    if k_counts[recommended_k] > 1:
        print(f"\nRecommended number of clusters: {recommended_k} (supported by {k_counts[recommended_k]} metrics)")
    else:
        # If there's a tie, prefer silhouette score
        print(f"\nRecommended number of clusters: {optimal_k_silhouette} (based on silhouette score)")
        recommended_k = optimal_k_silhouette

    # Create metrics dictionary
    metrics = {
        'k_range': list(k_range),
        'inertia': inertia,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'optimal_k': recommended_k
    }

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save metrics to a file
    np.save('results/cluster_metrics.npy', metrics)

    return recommended_k, metrics

def apply_kmeans(X_scaled, n_clusters, random_state=42):
    """Apply K-means clustering"""
    print("\nApplying K-means clustering...")

    # Train the model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, labels)

    print(f"K-means with {n_clusters} clusters complete")
    print(f"Silhouette score: {silhouette:.4f}")

    # Create result dictionary
    result = {
        'model': kmeans,
        'labels': labels,
        'silhouette': silhouette,
        'n_clusters': n_clusters
    }

    # Create directory if it doesn't exist
    os.makedirs('results/kmeans', exist_ok=True)
    
    # Save the cluster labels
    np.save(f'results/kmeans/kmeans_labels_{n_clusters}.npy', labels)

    return result