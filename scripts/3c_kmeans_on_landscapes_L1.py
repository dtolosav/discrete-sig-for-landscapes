"""K-means clustering on landscapes with L_1 distance.

Loads landscapes from .lan files, performs k-means clustering using L_1 distance
on landscape functions, and computes ARI/NMI scores against sequence similarity labels.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add src to path and import landscapes module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    import landscapes as pl
except ImportError:
    print("ERROR: 'landscapes' module not found. Expected at src/landscapes.py")
    sys.exit(1)

start_time = datetime.now()
LANDSCAPE_DIR = "data/external/landscapes"

if not os.path.isdir(LANDSCAPE_DIR):
    raise FileNotFoundError(f"Landscape directory not found: {LANDSCAPE_DIR}")

print(f"Using landscape directory: {LANDSCAPE_DIR}\n")


def load_classifications(csv_path):
    """Load chain-to-representative classification mapping."""
    df = pd.read_csv(csv_path)
    if 'Chain' not in df.columns or 'Representative' not in df.columns:
        raise ValueError(f"CSV must have 'Chain' and 'Representative' columns. Found: {df.columns.tolist()}")
    return df.astype({'Chain': str})


def find_landscape_chains(landscape_dir, classification_df):
    """Find classification chains that have _1.lan files in landscape_dir."""
    available = set()
    for fp in glob.glob(os.path.join(landscape_dir, '*_1.lan')):
        chain = os.path.basename(fp)[:-6]  # Remove '_1.lan'
        available.add(chain)
    
    candidates = set(classification_df['Chain'].astype(str))
    matched = sorted(candidates & available)
    
    if not matched:
        raise ValueError(f"No classification chains found in {landscape_dir}")
    
    print(f"Found {len(matched)} of {len(candidates)} classification chains in landscape directory")
    return matched


def run_l1_kmeans(landscapes, labels, n_clusters=9, n_runs=100, max_iter=200, base_seed=42):
    """Run k-means with L_1 distance on landscape functions.
    
    Args:
        landscapes: list of PersistenceLandscape objects
        labels: array of true cluster labels
        n_clusters: number of clusters
        n_runs: number of random restarts
        max_iter: max iterations per run
        base_seed: random seed
    
    Returns:
        dict with clustering results (ARI, NMI, etc)
    """
    n = len(landscapes)
    rng = np.random.RandomState(base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_runs)
    
    all_scores = []
    total_dist_time = 0.0
    
    for run_idx, seed in enumerate(seeds):
        rs = np.random.RandomState(seed)
        
        # Initialize centers by sampling random landscapes
        init_idx = rs.choice(n, size=n_clusters, replace=False)
        centers = [landscapes[i] for i in init_idx]
        cluster_assignment = np.full(n, -1, dtype=int)
        
        for iteration in range(1, max_iter + 1):
            # Assignment: compute distances and assign to nearest center
            t0 = datetime.now()
            distances = np.zeros((n, n_clusters))
            for i in range(n):
                for k in range(n_clusters):
                    distances[i, k] = pl.distance(landscapes[i], centers[k], p=1)
            t1 = datetime.now()
            total_dist_time += (t1 - t0).total_seconds()
            
            new_assignment = np.argmin(distances, axis=1)
            
            # Check convergence
            if np.array_equal(new_assignment, cluster_assignment):
                break
            cluster_assignment = new_assignment
            
            # Update: set centers to coordinate-wise median of cluster members
            for k in range(n_clusters):
                members = [landscapes[i] for i in range(n) if cluster_assignment[i] == k]
                if members:
                    centers[k] = members[0] if len(members) == 1 else _median_landscape(members)
                else:
                    centers[k] = landscapes[rs.randint(0, n)]
        
        ari = adjusted_rand_score(labels, cluster_assignment)
        nmi = normalized_mutual_info_score(labels, cluster_assignment)
        all_scores.append({'ARI': ari, 'NMI': nmi, 'iterations': iteration})
        
        if (run_idx + 1) % 10 == 0:
            print(f"  Completed {run_idx + 1}/{n_runs} runs")
    
    ari_arr = np.array([s['ARI'] for s in all_scores])
    nmi_arr = np.array([s['NMI'] for s in all_scores])
    
    return {
        'all_scores': all_scores,
        'n_runs': n_runs,
        'n_clusters': n_clusters,
        'avg_ari': float(np.mean(ari_arr)),
        'std_ari': float(np.std(ari_arr)),
        'min_ari': float(np.min(ari_arr)),
        'max_ari': float(np.max(ari_arr)),
        'avg_nmi': float(np.mean(nmi_arr)),
        'std_nmi': float(np.std(nmi_arr)),
        'min_nmi': float(np.min(nmi_arr)),
        'max_nmi': float(np.max(nmi_arr)),
        'total_distance_time_s': total_dist_time,
        'run_duration_s': (datetime.now() - start_time).total_seconds(),
    }


def _median_landscape(landscapes):
    """Compute coordinate-wise median of a list of landscapes (simplified)."""
    # For now, return the first landscape as placeholder; 
    # proper median would require landscape algebra
    return landscapes[0]


if __name__ == '__main__':
    # Load classifications and find matching chains
    classification_csv = "data/external/trefoil_list.csv"
    classification_df = load_classifications(classification_csv)
    matched_chains = find_landscape_chains(LANDSCAPE_DIR, classification_df)
    
    # Merge to get labels for matched chains
    matched_df = pd.DataFrame({'Chain': matched_chains})
    merged = matched_df.merge(classification_df, on='Chain', how='left')
    
    # Filter out 'Other' or 'other' labels
    valid_mask = merged['Representative'].notna() & ~merged['Representative'].isin(['Other', 'other'])
    merged_filtered = merged[valid_mask]
    
    print(f"Filtered out {len(merged) - len(merged_filtered)} landscapes labeled 'Other'")
    print(f"Kept {len(merged_filtered)} landscapes for clustering")
    
    labels_array = merged_filtered['Representative'].values
    chains_to_cluster = merged_filtered['Chain'].tolist()
    
    print(f"\nLoading landscapes for {len(chains_to_cluster)} chains...")
    landscapes_list = []
    for chain in chains_to_cluster:
        lan_path = os.path.join(LANDSCAPE_DIR, f"{chain}_1.lan")
        lan = pl.load(lan_path)
        landscapes_list.append(lan)
    
    print(f"Running k-means with L_1 distance ({len(landscapes_list)} landscapes, 9 clusters, 100 runs)...\n")
    results = run_l1_kmeans(landscapes_list, labels_array, n_clusters=9, n_runs=100, max_iter=200)
    
    # Print results
    print("\n" + "="*60)
    print("L_1 K-MEANS CLUSTERING RESULTS")
    print("="*60)
    print(f"  Landscapes: {len(landscapes_list)}")
    print(f"  Clusters: {results['n_clusters']}")
    print(f"  Runs: {results['n_runs']}")
    print(f"  ARI: {results['avg_ari']:.4f} ± {results['std_ari']:.4f} (min {results['min_ari']:.4f}, max {results['max_ari']:.4f})")
    print(f"  NMI: {results['avg_nmi']:.4f} ± {results['std_nmi']:.4f} (min {results['min_nmi']:.4f}, max {results['max_nmi']:.4f})")
    print(f"  Distance compute time: {results['total_distance_time_s']:.2f} sec")
    print(f"  Total time: {results['run_duration_s']:.2f} sec")
    print("="*60)
    
    # Save results
    results_path = f"results/kmeans_landscapes_L1_I100_L15_W1_noOther_{start_time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(results_path, **results)
    print(f"\nResults saved to {results_path}")

    









