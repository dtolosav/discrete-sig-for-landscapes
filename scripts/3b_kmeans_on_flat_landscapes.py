"""
K-means clustering on flattened truncated landscapes (L=15) as feature vectors.

Loads pre-computed landscapes from NPZ file, flattens each landscape into a feature vector,
performs k-means clustering using scikit-learn, and computes ARI/NMI scores against
sequence similarity labels.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import time

# Start timing
start_time = datetime.now()

# Configuration
DATA_FILE = "/home/deniel/Math/signature-tensors-tda/results/tests/landscapes_array_L15_noOther_20260522_164313.npz"
CLASSIFICATION_CSV = "data/external/trefoil_list.csv"
N_CLUSTERS = 9
N_RUNS = 100
BASE_SEED = 42

print(f"Loading data from: {DATA_FILE}")
print(f"Using classification file: {CLASSIFICATION_CSV}")
print(f"Number of clusters: {N_CLUSTERS}")
print(f"Number of runs: {N_RUNS}")
print()

def load_landscapes_and_ordering(data_file):
    """Load landscapes array and ordering dictionary from NPZ file."""
    data = np.load(data_file, allow_pickle=True)
    landscapes_array = data['result_array']  # Shape: (n_samples, n_levels, n_grid_points)
    ordering_dict = data['ordering'].item()  # Dict: chain_name -> index
    return landscapes_array, ordering_dict

def load_classifications(csv_path):
    """Load chain-to-representative classification mapping."""
    df = pd.read_csv(csv_path)
    if 'Chain' not in df.columns or 'Representative' not in df.columns:
        raise ValueError(f"CSV must have 'Chain' and 'Representative' columns. Found: {df.columns.tolist()}")
    return df.astype({'Chain': str})

def prepare_data(landscapes_array, ordering_dict, classification_df):
    """Prepare flattened landscapes and corresponding labels."""
    # Get chain names from ordering dict (remove '_1.lan' suffix)
    chain_names = [key[:-6] for key in ordering_dict.keys()]

    # Create DataFrame with chains
    chains_df = pd.DataFrame({'Chain': chain_names})

    # Merge with classifications
    merged = chains_df.merge(classification_df, on='Chain', how='left')

    # Filter out 'Other' or 'other' labels
    valid_mask = merged['Representative'].notna() & ~merged['Representative'].isin(['Other', 'other'])
    merged_filtered = merged[valid_mask]

    print(f"Total landscapes in data: {len(chains_df)}")
    print(f"After filtering 'Other' labels: {len(merged_filtered)}")

    # Get indices of valid chains
    valid_chains = merged_filtered['Chain'].tolist()
    valid_indices = [ordering_dict[f"{chain}_1.lan"] for chain in valid_chains]

    # Extract valid landscapes and flatten
    valid_landscapes = landscapes_array[valid_indices]  # Shape: (n_valid, 15, len_grid)
    n_samples, n_levels, n_grid = valid_landscapes.shape
    flattened_landscapes = valid_landscapes.reshape(n_samples, n_levels * n_grid)  # Shape: (n_valid, 15*153696)

    # Get labels
    labels = merged_filtered['Representative'].values

    print(f"Flattened landscape shape: {flattened_landscapes.shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    print()

    return flattened_landscapes, labels

def run_kmeans_clustering(feature_matrix, true_labels, n_clusters=9, n_runs=100, base_seed=42):
    """Run k-means clustering multiple times and compute ARI/NMI scores."""

    # Generate seeds for reproducibility
    rng = np.random.RandomState(base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_runs)

    all_scores = []
    run_times = []

    print("Running k-means clustering...")
    for run_idx in range(n_runs):
        run_start = time.time()

        # Run k-means with specific seed
        kmeans = KMeans(n_clusters=n_clusters, random_state=int(seeds[run_idx]), n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        run_end = time.time()
        run_time = run_end - run_start
        run_times.append(run_time)

        # Compute scores
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)

        all_scores.append({'ARI': ari, 'NMI': nmi, 'run_time': run_time})

        if (run_idx + 1) % 10 == 0:
            print(f"  Completed {run_idx + 1}/{n_runs} runs")

    # Compute statistics
    ari_scores = np.array([s['ARI'] for s in all_scores])
    nmi_scores = np.array([s['NMI'] for s in all_scores])
    run_times = np.array(run_times)

    results = {
        'all_scores': all_scores,
        'n_runs': n_runs,
        'n_clusters': n_clusters,
        'avg_ari': float(np.mean(ari_scores)),
        'std_ari': float(np.std(ari_scores)),
        'min_ari': float(np.min(ari_scores)),
        'max_ari': float(np.max(ari_scores)),
        'avg_nmi': float(np.mean(nmi_scores)),
        'std_nmi': float(np.std(nmi_scores)),
        'min_nmi': float(np.min(nmi_scores)),
        'max_nmi': float(np.max(nmi_scores)),
        'avg_run_time': float(np.mean(run_times)),
        'std_run_time': float(np.std(run_times)),
        'total_time': float(np.sum(run_times)),
        'total_script_time': (datetime.now() - start_time).total_seconds(),
    }

    return results

def main():
    # Load data
    landscapes_array, ordering_dict = load_landscapes_and_ordering(DATA_FILE)
    classification_df = load_classifications(CLASSIFICATION_CSV)

    # Prepare feature matrix and labels
    feature_matrix, true_labels = prepare_data(landscapes_array, ordering_dict, classification_df)

    # Run clustering
    results = run_kmeans_clustering(feature_matrix, true_labels, N_CLUSTERS, N_RUNS, BASE_SEED)

    # Print results
    print("\n" + "="*70)
    print("K-MEANS CLUSTERING ON FLATTENED LANDSCAPES (L=15)")
    print("="*70)
    print(f"Landscapes: {feature_matrix.shape[0]}")
    print(f"Feature vector length: {feature_matrix.shape[1]}")
    print(f"Clusters: {results['n_clusters']}")
    print(f"Runs: {results['n_runs']}")
    print()
    print("ARI scores:")
    print(f"  Mean:     {results['avg_ari']:.4f}")
    print(f"  Std:      {results['std_ari']:.4f}")
    print(f"  Range:    [{results['min_ari']:.4f}, {results['max_ari']:.4f}]")
    print()
    print("NMI scores:")
    print(f"  Mean:     {results['avg_nmi']:.4f}")
    print(f"  Std:      {results['std_nmi']:.4f}")
    print(f"  Range:    [{results['min_nmi']:.4f}, {results['max_nmi']:.4f}]")
    print()
    print("Timing:")
    print(f"  Avg per run:   {results['avg_run_time']:.3f}s")
    print(f"  Std per run:   {results['std_run_time']:.3f}s")
    print(f"  Total k-means: {results['total_time']:.3f}s")
    print(f"  Total script:  {results['total_script_time']:.3f}s")
    print("="*70)

    # Save results
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    results_path = f"results/kmeans_flattened_landscapes_L15_I{N_RUNS}_{timestamp}.npz"
    np.savez(results_path, **results)
    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()