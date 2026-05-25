"""Perform k-means clustering on signature data and compute ARI/NMI scores.

Loads a signature NPZ file, performs k-means clustering with 9 clusters
(after standardizing the data), and computes Adjusted Rand Index (ARI) and
Normalized Mutual Information (NMI) scores compared to sequence similarity
classification.

Usage:
  python scripts/3a_kmeans_clustering.py --input /path/to/signatures.npz
  python scripts/3a_kmeans_clustering.py --input /path/to/signatures.npz --classification /path/to/classification.csv
  python scripts/3a_kmeans_clustering.py --input /path/to/signatures.npz --landscape /path/to/landscape.npz --classification /path/to/classification.csv --clusters 9 --iterations 100 --output /path/to/results.json

Arguments:
  --input, -i
      Required. Path to the signature NPZ file containing signature vectors.
  --landscape, -l
      Optional. Path to a landscape NPZ file providing ordering information.
      If omitted, the script will try to find a matching landscape automatically.
  --classification, -c
      Optional. Path to a classification CSV file. The file should contain at
      least 'Chain' and 'Representative' columns. Defaults to
      ../data/external/trefoil_list.csv relative to the script location.
  --clusters, -k
      Number of clusters to use for k-means. Default is 9.
  --iterations, -n
      Number of clustering iterations to run for averaging ARI/NMI scores.
      Default is 100.
  --output, -o
      Optional. Path to save results in .npz or .json format. If omitted,
      results are printed but not saved.
"""

import argparse
import sys
import pathlib
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def find_landscape_by_sample_count(sig_npz_path, n_samples):
    """Find a landscape NPZ file with matching sample count.
    
    This is useful for integrated signatures which may not have a direct filename match.
    """
    sig_path = pathlib.Path(sig_npz_path)
    project_root = sig_path.parent.parent
    
    # Search in data/ and results/ directories
    search_dirs = [
        sig_path.parent,
        project_root / 'data',
        project_root / 'results',
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for npz_file in search_dir.glob('*.npz'):
            # Skip signature files
            if npz_file.name.startswith('signatures_'):
                continue
            # Skip integrated landscape signatures
            if 'integrated' not in npz_file.name.lower():
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    
                    # Look for array keys
                    for array_key in ['Chen_array', 'result_array', 'array', 'lans']:
                        if array_key in data.files:
                            arr = data[array_key]
                            if isinstance(arr, np.ndarray) and arr.shape[0] == n_samples:
                                # Check if it has ordering
                                for ordering_key in ['Chen_ordering', 'ordering', 'Ordering']:
                                    if ordering_key in data.files:
                                        return npz_file
                except Exception:
                    # Skip files that can't be loaded
                    continue
    
    return None


def find_corresponding_landscape_npz(signature_npz_path):
    """Find the corresponding landscape NPZ file for a signature NPZ.
    
    Given a signature NPZ path like:
      signatures_all_landscapes_array_grid256_L15_W3_..._W3_....npz
    
    Return the landscape NPZ path like:
      all_landscapes_array_grid256_L15_W3_....npz
    """
    # Extract the stem up to the landscape date
    # Pattern: signatures_<landscape_base>_W<weight>_<date>_W<weight>_<date>
    sig_path = pathlib.Path(signature_npz_path)
    filename = sig_path.stem
    sig_parent = sig_path.parent
    project_root = sig_parent.parent
    
    # Try to extract the landscape filename
    # Remove the "signatures_" prefix
    if filename.startswith('signatures_'):
        base = filename[len('signatures_'):]
        
        # Split by underscore and look for date patterns (8 digits followed by 6 digits)
        parts = base.split('_')
        
        # Find the first date pattern YYYYMMDD_HHMMSS that's followed by _W
        # The landscape filename should end after this date pattern
        landscape_end_idx = -1
        for i in range(len(parts) - 2):
            if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 6 and parts[i+1].isdigit():
                # Found a date pattern: YYYYMMDD_HHMMSS
                # Check if the next part starts with W (indicating additional signature params)
                if i + 2 < len(parts) and parts[i + 2].startswith('W'):
                    landscape_end_idx = i + 1  # Include the time part
                    break
        
        if landscape_end_idx > 0:
            landscape_parts = parts[:landscape_end_idx + 1]
            landscape_stem = '_'.join(landscape_parts)
            landscape_filename = landscape_stem + '.npz'
            
            # Search in multiple locations
            search_paths = [
                sig_parent / landscape_filename,           # Same directory as signature
                project_root / 'data' / landscape_filename, # data/ directory
                project_root / landscape_filename,          # project root
            ]
            
            for landscape_path in search_paths:
                if landscape_path.exists():
                    return landscape_path
    
    return None


def load_signatures_and_ordering(sig_npz_path, landscape_npz_path=None):
    """Load signatures and corresponding chain ordering.
    
    Args:
        sig_npz_path: Path to signature NPZ file
        landscape_npz_path: Optional path to landscape NPZ with ordering.
                          If None, tries to find it automatically.
    
    Returns:
        signatures: numpy array of shape (n_samples, n_features)
        ordering: list of chain identifiers corresponding to each row
    """
    sig_data = np.load(sig_npz_path)
    
    # Load signatures
    if 'signatures' in sig_data.files:
        signatures = sig_data['signatures']
    else:
        raise KeyError(f"Could not find 'signatures' key in {sig_npz_path}. "
                      f"Available keys: {sig_data.files}")
    
    print(f"Loaded signatures from {sig_npz_path}")
    print(f"  Shape: {signatures.shape}")
    
    # Find and load landscape NPZ for ordering
    if landscape_npz_path is None:
        landscape_npz_path = find_corresponding_landscape_npz(sig_npz_path)
    
    # If we found a landscape but it doesn't have ordering, search for alternatives
    if landscape_npz_path is not None:
        try:
            lan_data = np.load(landscape_npz_path, allow_pickle=True)
            has_ordering = any(key in lan_data.files for key in ['Chen_ordering', 'ordering', 'Ordering'])
            if not has_ordering:
                # Try to find an alternative landscape file with ordering
                alternative = find_landscape_by_sample_count(sig_npz_path, signatures.shape[0])
                if alternative is not None:
                    landscape_npz_path = alternative
        except Exception:
            pass
    
    if landscape_npz_path is None:
        # Try sample count search
        landscape_npz_path = find_landscape_by_sample_count(sig_npz_path, signatures.shape[0])
    
    if landscape_npz_path is None:
        print("Warning: Could not find corresponding landscape NPZ file.")
        print("  Generated ordering will be sequential indices.")
        ordering = [f"Sample_{i}" for i in range(signatures.shape[0])]
    else:
        print(f"Loading landscape NPZ from {landscape_npz_path}")
        lan_data = np.load(landscape_npz_path, allow_pickle=True)
        
        # Look for ordering in various possible keys
        ordering = None
        ordering_key = None
        for key in ['Chen_ordering', 'ordering', 'Ordering']:
            if key in lan_data.files:
                ordering = lan_data[key]
                ordering_key = key
                break
        
        if ordering is None:
            print(f"Warning: Could not find ordering in landscape NPZ.")
            print(f"  Available keys: {lan_data.files}")
            ordering = [f"Sample_{i}" for i in range(signatures.shape[0])]
        else:
            # Handle numpy object arrays (e.g., dict stored in NPZ)
            if isinstance(ordering, np.ndarray) and ordering.dtype == 'O' and ordering.shape == ():
                ordering_dict = ordering.item()
                if isinstance(ordering_dict, dict):
                    # Extract chain IDs from filenames in the dictionary keys
                    # Filenames are like '1by7_A_1.lan' or '1by7_A_0.lan'
                    chain_ids = []
                    for filename in ordering_dict.keys():
                        # Remove .lan extension and the dimension suffix
                        chain_id = filename.replace('_1.lan', '').replace('_0.lan', '').replace('.lan', '')
                        chain_ids.append(chain_id)
                    ordering = chain_ids
                else:
                    ordering = [str(x) for x in ordering_dict]
            elif isinstance(ordering, np.ndarray):
                # Convert array to list of strings
                ordering = ordering.tolist() if isinstance(ordering, np.ndarray) else list(ordering)
                ordering = [str(x) for x in ordering]
            else:
                ordering = [str(x) for x in ordering]
    
    return signatures, ordering


def load_classification_data(classification_csv_path):
    """Load protein classification data from CSV.
    
    Expected CSV format:
        Chain,Depth,Length,N-tail,C-tail,Representative,...
    
    Returns:
        DataFrame with at least 'Chain' and 'Representative' columns
    """
    df = pd.read_csv(classification_csv_path)
    
    if 'Chain' not in df.columns or 'Representative' not in df.columns:
        raise ValueError(f"CSV must have 'Chain' and 'Representative' columns. "
                        f"Found: {df.columns.tolist()}")
    
    df['Chain'] = df['Chain'].astype(str)
    return df


def match_signatures_to_classification(signatures, ordering, classification_df):
    """Match signatures to their Representative labels.
    
    Args:
        signatures: numpy array of shape (n_samples, n_features)
        ordering: list of chain identifiers
        classification_df: DataFrame with 'Chain' and 'Representative' columns
    
    Returns:
        X: signatures as DataFrame
        labels: Series of Representative values for each signature
        valid_mask: boolean mask indicating which samples were successfully matched
    """
    # Create a DataFrame with chain identifiers
    df = pd.DataFrame({
        'Chain': ordering,
        'index': range(len(ordering))
    })
    
    # Merge with classification data
    merged = df.merge(classification_df[['Chain', 'Representative']], 
                      on='Chain', how='left')
    
    # Filter to only rows that matched
    valid_mask = merged['Representative'].notna()
    n_matched = valid_mask.sum()
    n_total = len(ordering)
    
    print(f"Matched {n_matched}/{n_total} signatures to classification labels")
    
    if n_matched == 0:
        raise ValueError("No signatures matched to classification data. "
                        "Check that chain identifiers in signature ordering "
                        "match those in classification CSV.")
    
    # Filter signatures and labels
    valid_indices = merged.loc[valid_mask, 'index'].values.astype(int)
    X = signatures[valid_indices]
    labels = merged.loc[valid_mask, 'Representative'].values
    
    return X, labels


def filter_out_other_classes(X, labels, other_label='Other'):
    """Remove samples labeled as 'Other'.
    
    Args:
        X: numpy array of samples
        labels: array-like of class labels
        other_label: the label to filter out (default 'Other')
    
    Returns:
        X_filtered: filtered samples
        labels_filtered: filtered labels
        n_removed: number of samples removed
        n_total: total number of samples before filtering
    """
    mask = np.array(labels) != other_label
    n_total = len(labels)
    n_removed = (~mask).sum()
    n_kept = mask.sum()
    
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    return X_filtered, labels_filtered, n_removed, n_total


def standardize_and_cluster(X, n_clusters=9, random_seed=42):
    """Standardize data and perform k-means clustering.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        n_clusters: number of clusters (default 9)
        random_seed: random seed for reproducibility
    
    Returns:
        X_scaled: standardized data as numpy array
        cluster_labels: cluster assignments for each sample
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return X_scaled, cluster_labels


def run_multiple_clusterings(X, true_labels, n_clusters=9, n_iterations=100, base_seed=42):
    """Run k-means clustering multiple times with different seeds and compute scores.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        true_labels: ground truth labels for ARI/NMI computation
        n_clusters: number of clusters (default 9)
        n_iterations: number of times to run clustering (default 100)
        base_seed: initial seed to generate random seeds (default 42)
    
    Returns:
        results_dict: dictionary with keys:
            - 'X_scaled': standardized data
            - 'all_scores': list of dicts, each with 'ARI' and 'NMI' keys
            - 'seeds': list of random seeds used
            - 'avg_ari': average ARI across all iterations
            - 'avg_nmi': average NMI across all iterations
            - 'std_ari': standard deviation of ARI
            - 'std_nmi': standard deviation of NMI
    """
    # Standardize once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Generate seeds for reproducibility
    rng = np.random.RandomState(base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_iterations).tolist()
    
    all_scores = []
    
    for i, seed in enumerate(seeds):
        # Perform clustering with this seed
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Compute scores
        scores = compute_scores(true_labels, cluster_labels)
        all_scores.append(scores)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed iteration {i + 1}/{n_iterations}")
    
    # Compute statistics
    ari_scores = [s['ARI'] for s in all_scores]
    nmi_scores = [s['NMI'] for s in all_scores]
    
    results_dict = {
        'X_scaled': X_scaled,
        'all_scores': all_scores,
        'seeds': seeds,
        'avg_ari': np.mean(ari_scores),
        'avg_nmi': np.mean(nmi_scores),
        'std_ari': np.std(ari_scores),
        'std_nmi': np.std(nmi_scores),
        'min_ari': np.min(ari_scores),
        'max_ari': np.max(ari_scores),
        'min_nmi': np.min(nmi_scores),
        'max_nmi': np.max(nmi_scores),
    }
    
    return results_dict


def compute_scores(true_labels, cluster_labels):
    """Compute ARI and NMI scores.
    
    Args:
        true_labels: ground truth labels (sequence similarity classification)
        cluster_labels: predicted cluster labels
    
    Returns:
        dict with 'ARI' and 'NMI' keys
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    return {'ARI': ari, 'NMI': nmi}


def save_clustering_results(results, output_path, metadata=None):
    """Save clustering results to a file (NPZ or JSON format).
    
    Args:
        results: results dictionary from run_multiple_clusterings()
        output_path: path to save results (extension determines format: .npz or .json)
        metadata: optional dictionary with additional metadata to save
    
    Returns:
        output_path: the path where results were saved
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        # Save as JSON (human-readable)
        import json
        
        # Prepare data for JSON serialization
        json_data = {
            'summary': {
                'avg_ari': float(results['avg_ari']),
                'std_ari': float(results['std_ari']),
                'min_ari': float(results['min_ari']),
                'max_ari': float(results['max_ari']),
                'avg_nmi': float(results['avg_nmi']),
                'std_nmi': float(results['std_nmi']),
                'min_nmi': float(results['min_nmi']),
                'max_nmi': float(results['max_nmi']),
            },
            'seeds': [int(s) for s in results['seeds']],
            'n_iterations': len(results['seeds']),
            'individual_scores': [
                {'iteration': i, 'ARI': float(s['ARI']), 'NMI': float(s['NMI'])}
                for i, s in enumerate(results['all_scores'], 1)
            ]
        }
        
        # Add metadata if provided
        if metadata:
            json_data['metadata'] = metadata
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    else:
        # Default: Save as NPZ
        if output_path.suffix.lower() != '.npz':
            output_path = output_path.with_suffix('.npz')
        
        # Prepare arrays for NPZ
        ari_scores = np.array([s['ARI'] for s in results['all_scores']])
        nmi_scores = np.array([s['NMI'] for s in results['all_scores']])
        seeds_array = np.array(results['seeds'], dtype=np.int64)
        
        # Create a dictionary with summary statistics
        summary = {
            'avg_ari': np.float64(results['avg_ari']),
            'std_ari': np.float64(results['std_ari']),
            'min_ari': np.float64(results['min_ari']),
            'max_ari': np.float64(results['max_ari']),
            'avg_nmi': np.float64(results['avg_nmi']),
            'std_nmi': np.float64(results['std_nmi']),
            'min_nmi': np.float64(results['min_nmi']),
            'max_nmi': np.float64(results['max_nmi']),
            'n_iterations': np.int32(len(results['seeds'])),
        }
        
        # Add metadata if provided
        if metadata:
            summary['metadata'] = metadata
        
        # Save NPZ file
        np.savez(output_path,
                 summary_stats=summary,
                 ari_scores=ari_scores,
                 nmi_scores=nmi_scores,
                 seeds=seeds_array,
                 allow_pickle=True)
        
        print(f"Results saved to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Perform k-means clustering on signatures and compute ARI/NMI scores')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to signature NPZ file')
    parser.add_argument('--landscape', '-l', type=str, default=None,
                       help='Path to landscape NPZ file with ordering. '
                            'If omitted, will try to find it automatically.')
    parser.add_argument('--classification', '-c', type=str, default=None,
                       help='Path to classification CSV file. '
                            'Defaults to ../data/external/trefoil_list.csv relative to script location.')
    parser.add_argument('--clusters', '-k', type=int, default=9,
                       help='Number of clusters (default 9)')
    parser.add_argument('--iterations', '-n', type=int, default=100,
                       help='Number of clustering iterations to run for averaging scores (default 100)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save results (.npz or .json format). If omitted, results are not saved.')
    args = parser.parse_args()
    
    # Resolve paths
    sig_npz = pathlib.Path(args.input)
    if not sig_npz.exists():
        print(f"Error: Signature NPZ file not found: {sig_npz}", file=sys.stderr)
        sys.exit(1)
    
    landscape_npz = None
    if args.landscape:
        landscape_npz = pathlib.Path(args.landscape)
        if not landscape_npz.exists():
            print(f"Error: Landscape NPZ file not found: {landscape_npz}", file=sys.stderr)
            sys.exit(1)
    
    # Default classification CSV
    if args.classification is None:
        script_dir = pathlib.Path(__file__).resolve().parent.parent
        # Try to find in same workspace
        default_paths = [
            script_dir.parent / 'Signature tensors in TDA' / 'data' / 'external' / 'trefoil_list.csv',
            script_dir.parent / 'Signature tensors in TDA_pysiglib' / 'data' / 'external' / 'trefoil_list.csv',
            pathlib.Path('/home/deniel/Math/Signature tensors in TDA/data/external/trefoil_list.csv'),
        ]
        classification_csv = None
        for path in default_paths:
            if path.exists():
                classification_csv = path
                break
        
        if classification_csv is None:
            print("Error: Could not find default classification CSV at:", file=sys.stderr)
            for path in default_paths:
                print(f"  {path}", file=sys.stderr)
            print("Please provide --classification argument.", file=sys.stderr)
            sys.exit(1)
    else:
        classification_csv = pathlib.Path(args.classification)
        if not classification_csv.exists():
            print(f"Error: Classification CSV not found: {classification_csv}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Classification CSV: {classification_csv}")
    
    # Load data
    signatures, ordering = load_signatures_and_ordering(sig_npz, landscape_npz)
    classification_df = load_classification_data(classification_csv)
    
    # Match and filter
    X, labels = match_signatures_to_classification(signatures, ordering, classification_df)
    
    # Remove 'Other' classified samples
    print(f"\nRemoving proteins classified as 'Other'...")
    X, labels, n_removed, n_total = filter_out_other_classes(X, labels, other_label='Other')
    n_kept = n_total - n_removed
    print(f"  Samples discarded: {n_removed}")
    print(f"  Samples kept: {n_kept}")
    
    if n_kept == 0:
        print("Error: No samples remaining after removing 'Other' class.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nPerforming k-means clustering with {args.clusters} clusters ({args.iterations} iterations for averaging)...")
    # Run multiple clusterings and compute average scores
    results = run_multiple_clusterings(X, labels, n_clusters=args.clusters, 
                                       n_iterations=args.iterations, base_seed=42)
    
    # Print results
    print("\n" + "="*60)
    print("CLUSTERING RESULTS")
    print("="*60)
    print(f"Input signatures: {sig_npz.name}")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of clusters: {args.clusters}")
    print(f"Number of classes (sequence similarity): {len(np.unique(labels))}")
    print(f"Number of iterations: {args.iterations}\n")
    
    print("AVERAGED METRICS (across all iterations):")
    print(f"  Adjusted Rand Index (ARI):       {results['avg_ari']:>8.4f} ± {results['std_ari']:.4f}")
    print(f"  Normalized Mutual Information:   {results['avg_nmi']:>8.4f} ± {results['std_nmi']:.4f}")
    print(f"\nARI range:  [{results['min_ari']:.4f}, {results['max_ari']:.4f}]")
    print(f"NMI range:  [{results['min_nmi']:.4f}, {results['max_nmi']:.4f}]")
    
    print(f"\nRandom seeds used ({len(results['seeds'])} total):")
    print(f"  {results['seeds']}")
    print("="*60)
    
    # Save results if output path specified
    if args.output:
        metadata = {
            'input_file': sig_npz.name,
            'n_samples': len(labels),
            'n_clusters': args.clusters,
            'n_classes': len(np.unique(labels)),
            'n_iterations': args.iterations,
        }
        save_clustering_results(results, args.output, metadata=metadata)
    
    return results


if __name__ == '__main__':
    main()
