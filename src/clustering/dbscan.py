# src/clustering/dbscan.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def estimate_dbscan_parameters(X_scaled, min_samples=5):
    """Estimate the optimal epsilon parameter for DBSCAN"""
    print("\nEstimating optimal epsilon parameter for DBSCAN...")

    # Compute k-distances (using k=min_samples)
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X_scaled)
    distances, indices = neigh.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1])

    # Create directory if it doesn't exist
    os.makedirs('results/dbscan', exist_ok=True)
    
    # Plot k-distances
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'{min_samples}-distance Graph for DBSCAN Epsilon Estimation', fontsize=14)
    plt.xlabel('Points sorted by distance', fontsize=12)
    plt.ylabel(f'{min_samples}-th nearest neighbor distance', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/dbscan/dbscan_kdistances_{min_samples}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Estimate epsilon from the knee point
    try:
        kneedle = KneeLocator(range(len(distances)), distances,
                             S=1.0, curve='convex', direction='increasing')
        epsilon = distances[kneedle.knee]
        print(f"Estimated epsilon based on knee detection: {epsilon:.4f} at index {kneedle.knee}")
    except:
        # Fall back to manual detection if kneed package isn't available
        # Calculate the second derivative
        second_derivative = np.diff(np.diff(distances))
        # Find the index of the maximum second derivative
        knee_point = np.argmax(second_derivative) + 1  # +1 adjustment for diff
        epsilon = distances[knee_point]
        print(f"Estimated epsilon based on second derivative: {epsilon:.4f} at index {knee_point}")

    return epsilon, min_samples

def apply_dbscan(X_scaled, epsilon, min_samples=5):
    """Apply DBSCAN clustering"""
    print("\nApplying DBSCAN clustering...")

    # Train the model
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"DBSCAN with epsilon={epsilon:.4f}, min_samples={min_samples} complete")
    print(f"Found {n_clusters} clusters and {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")

    # Calculate silhouette score (only for non-noise points)
    silhouette = 0
    if n_clusters > 1:
        # If we have noise points
        if n_noise > 0:
            mask = labels != -1
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
        else:
            silhouette = silhouette_score(X_scaled, labels)

        print(f"Silhouette score (excluding noise): {silhouette:.4f}")
    else:
        print("DBSCAN found only one cluster or only noise points. Cannot calculate silhouette score.")

    # Create result dictionary
    result = {
        'model': dbscan,
        'labels': labels,
        'silhouette': silhouette if n_clusters > 1 else 0,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'epsilon': epsilon,
        'min_samples': min_samples
    }
    
    # Create directory if it doesn't exist
    os.makedirs('results/dbscan', exist_ok=True)
    
    # Save the cluster labels
    np.save(f'results/dbscan/dbscan_labels_eps{epsilon:.4f}_min{min_samples}.npy', labels)

    return result