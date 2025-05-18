# src/clustering/gmm.py

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def select_best_covariance_type(X_scaled, n_components):
    """Select the best covariance type for GMM"""
    cov_types = ['full', 'tied', 'diag', 'spherical']
    results = {}

    for cov_type in cov_types:
        gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
        labels = gmm.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        results[cov_type] = silhouette
        print(f"GMM with {cov_type} covariance: Silhouette score = {silhouette:.4f}")

    best_cov_type = max(results, key=results.get)
    print(f"\nBest covariance type: {best_cov_type} (Silhouette: {results[best_cov_type]:.4f})")

    return best_cov_type

def apply_gmm(X_scaled, n_clusters, random_state=42, covariance_type='full'):
    """Apply Gaussian Mixture Model clustering"""
    print("\nApplying Gaussian Mixture Model clustering...")

    # Train the model
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type=covariance_type)
    labels = gmm.fit_predict(X_scaled)

    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, labels)

    print(f"GMM with {n_clusters} clusters complete")
    print(f"Silhouette score: {silhouette:.4f}")

    # Create result dictionary
    result = {
        'model': gmm,
        'labels': labels,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'covariance_type': covariance_type
    }
    
    # Create directory if it doesn't exist
    os.makedirs('results/gmm', exist_ok=True)
    
    # Save the cluster labels
    np.save(f'results/gmm/gmm_labels_{n_clusters}_{covariance_type}.npy', labels)
    
    # Save model info
    cluster_info = {
        'means': gmm.means_,
        'covariances': gmm.covariances_,
        'weights': gmm.weights_,
    }
    np.save(f'results/gmm/gmm_info_{n_clusters}_{covariance_type}.npy', cluster_info)

    return result