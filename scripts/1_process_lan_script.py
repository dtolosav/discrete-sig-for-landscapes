"""
Process landscapes from .lan files and save them as time-series of critical points in numpy arrays stores as an NPZ file.

This script processes persistence landscapes with flexible options:
  - Choose between critical-point or uniform grid discretization
  - Optionally integrate landscapes after evaluation
  - Optionally exclude proteins labeled "Other" from classification
  - Save results with filenames that reflect selected options

Usage:
  python scripts/1_process_lan_script.py --directory <path> [options]

Example:
  python scripts/1_process_lan_script.py --directory <lan_directory> --trunc-level <level> --grid-type <grid_type> --grid-size <size> --integrate --exclude-other --classification <classification_csv> --output-dir <output_directory>

Arguments:
  --directory, -d       str   required   Path to directory containing .lan files
  --trunc-level, -L     int   default=15  Number of persistence levels to keep
  --grid-type           str   default=critical  choices=[critical, uniform]
                        Grid type: 'critical' uses critical points, 'uniform' uses a fixed-size grid
  --grid-size, -g       int   default=None  Number of points in the uniform grid (required when --grid-type=uniform)
  --integrate, -I       flag   Integrate the landscapes after evaluation
  --exclude-other       flag   Exclude samples labeled "Other" in the classification CSV
  --classification, -c  str   default=None  Path to classification CSV (required when --exclude-other is used)
  --output-dir, -o      str   default=None  Output directory (defaults to ../data)
"""

import argparse
import sys
import pathlib
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.process_lan import load_lan, eval_landscape, lan2array, lan2array_batch
from src.integrate import integrate_lan_on_grid


def load_classification_data(classification_csv_path):
    """Load protein classification data from CSV.
    
    Expected CSV format with at least 'Chain' and 'Representative' columns.
    """
    df = pd.read_csv(classification_csv_path)
    
    if 'Chain' not in df.columns or 'Representative' not in df.columns:
        raise ValueError(f"CSV must have 'Chain' and 'Representative' columns. "
                        f"Found: {df.columns.tolist()}")
    
    df['Chain'] = df['Chain'].astype(str)
    return df


def extract_chain_id_from_filename(filename):
    """Extract chain ID from landscape filename.
    
    Expected format: <chain_id>_0.lan or <chain_id>_1.lan
    """
    # Remove .lan extension and the dimension suffix
    base = filename.replace('_1.lan', '').replace('_0.lan', '').replace('.lan', '')
    return base


def filter_ordering_by_classification(ordering, classification_df, exclude_other=False):
    """Filter ordering dictionary by classification data.
    
    Args:
        ordering: dict mapping filename to index
        classification_df: DataFrame with 'Chain' and 'Representative' columns
        exclude_other: if True, remove samples labeled 'Other'
    
    Returns:
        valid_indices: array of valid indices to keep
        valid_filenames: list of valid filenames
        n_removed: number of samples removed
    """
    valid_indices = []
    valid_filenames = []
    n_removed = 0
    
    for filename, idx in ordering.items():
        chain_id = extract_chain_id_from_filename(filename)
        
        # Find matching classification
        matches = classification_df[classification_df['Chain'] == chain_id]
        
        if len(matches) == 0:
            # No match in classification - skip
            n_removed += 1
            continue
        
        rep_label = matches.iloc[0]['Representative']
        
        # Skip 'Other' if requested
        if exclude_other and rep_label == 'Other':
            n_removed += 1
            continue
        
        valid_indices.append(idx)
        valid_filenames.append(filename)
    
    return np.array(valid_indices), valid_filenames, n_removed


def process_landscapes(directory, trunc_level, grid_type='critical', grid_size=None,
                       integrate=False, exclude_other=False, classification_csv=None,
                       output_dir=None):
    """Process landscapes from .lan files and optionally filter/integrate them.
    
    Args:
        directory: path to directory containing .lan files
        trunc_level: number of persistence levels to keep
        grid_type: 'critical' (use critical points) or 'uniform' (use uniform grid)
        grid_size: size of uniform grid (only used if grid_type='uniform')
        integrate: whether to integrate the landscapes
        exclude_other: whether to exclude samples labeled 'Other'
        classification_csv: path to classification CSV (required if exclude_other=True)
        output_dir: directory to save results (defaults to ../data)
    
    Returns:
        result_dict: dictionary with keys:
            - 'array': the landscape array
            - 'grid': the grid (epsilon or critical points)
            - 'ordering': mapping of filenames to indices
            - 'filename': the suggested output filename
    """
    
    # Optionally pre-filter files before loading landscapes
    filtered_file_list = None
    if exclude_other:
        if classification_csv is None:
            raise ValueError("classification_csv required when exclude_other=True")

        print(f"\nLoading classification data from {classification_csv}...")
        classification_df = load_classification_data(classification_csv)

        print("Filtering filenames to exclude 'Other' proteins before grid computation...")
        filtered_file_list = []
        for fname in os.listdir(directory):
            if not fname.endswith('1.lan'):
                continue
            chain_id = extract_chain_id_from_filename(fname)
            matches = classification_df[classification_df['Chain'] == chain_id]
            if len(matches) == 0:
                # No classification; skip as conservative behavior.
                continue
            rep_label = matches.iloc[0]['Representative']
            if rep_label == 'Other':
                continue
            filtered_file_list.append(fname)

        filtered_file_list = sorted(filtered_file_list)
        print(f"  Keeping {len(filtered_file_list)} landscapes (excluding Other)")

    # Load landscapes (with optional pre-filtering)
    print(f"Loading landscapes from {directory}...")
    if grid_type == 'critical':
        print(f"  Using critical points grid")
        result_array, ordering, global_grid = lan2array_batch(
            directory, trunc_level, grid=None, file_list=filtered_file_list
        )
    elif grid_type == 'uniform':
        if grid_size is None:
            raise ValueError("grid_size must be specified when grid_type='uniform'")
        print(f"  Using uniform grid with {grid_size} points")
        result_array, ordering, global_grid = lan2array_batch(
            directory, trunc_level, grid=grid_size, file_list=filtered_file_list
        )
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    print(f"  Loaded {result_array.shape[0]} landscapes")
    print(f"  Grid size: {len(global_grid)}")
    print(f"  Array shape: {result_array.shape}")

    # If pre-filtering applied, skip the old post-filtering block
    if exclude_other:
        # Rebuild ordering mapping to keep sequential indices
        ordering = {fn: i for i, fn in enumerate(sorted(ordering.keys()))}
        print(f"  Final ordering has {len(ordering)} entries")
    else:
        # Existing behavior: no filtering after collection
        pass
    
    # Optionally integrate
    if integrate:
        print(f"\nIntegrating landscapes...")
        # result_array has shape (N, trunc_level, T)
        # For each landscape, integrate across the grid
        integrated_array = np.zeros_like(result_array)
        
        for i in range(result_array.shape[0]):
            # For each level k, integrate the landscape function
            for k in range(result_array.shape[1]):
                # Get the landscape values at all grid points for this level
                lam_local = result_array[i, k, :]  # shape (T,)
                
                # Reshape for integrate_lan_on_grid which expects (T, d) where d is number of levels
                # Here we treat each level separately
                lam_2d = lam_local.reshape(-1, 1)
                
                # Integrate
                integrated_vals = integrate_lan_on_grid(global_grid, lam_2d, global_grid)
                integrated_array[i, k, :] = integrated_vals[:, 0]
        
        result_array = integrated_array
        print(f"  Integration complete")
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build filename with appropriate components
    output_filename = 'landscapes_array'
    
    if grid_type == 'uniform':
        output_filename += f'_grid{grid_size}'
    
    output_filename += f'_L{trunc_level}'
    
    if integrate:
        output_filename += '_integrated'
    
    if exclude_other:
        output_filename += '_noOther'
    
    output_filename += f'_{timestamp}.npz'
    
    if output_dir is None:
        output_dir = pathlib.Path(__file__).parent.parent / 'data'
    else:
        output_dir = pathlib.Path(output_dir)
    
    output_path = output_dir / output_filename
    
    return {
        'array': result_array,
        'grid': global_grid,
        'ordering': ordering,
        'filename': output_filename,
        'filepath': output_path,
    }


def save_landscapes(result_dict, output_path=None):
    """Save processed landscapes to NPZ file.
    
    Args:
        result_dict: dictionary from process_landscapes()
        output_path: path to save (uses result_dict['filepath'] if None)
    """
    if output_path is None:
        output_path = result_dict['filepath']
    else:
        output_path = pathlib.Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_path}...")
    np.savez(
        output_path,
        result_array=result_dict['array'],
        ordering=result_dict['ordering'],
        global_epsilon=result_dict['grid'],
        allow_pickle=True
    )
    print(f"✓ Saved successfully")
    print(f"  Array shape: {result_dict['array'].shape}")
    print(f"  Filename: {result_dict['filename']}")


def main():
    parser = argparse.ArgumentParser(
        description='Process landscapes from .lan files with flexible options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use critical points grid
  python process_lan_script.py --directory /path/to/landscapes --trunc-level 15

  # Use uniform grid with 256 points
  python process_lan_script.py --directory /path/to/landscapes --trunc-level 15 \\
                                --grid-type uniform --grid-size 256

  # Integrate and exclude 'Other' proteins
  python process_lan_script.py --directory /path/to/landscapes --trunc-level 15 \\
                                --grid-type uniform --grid-size 256 \\
                                --integrate --exclude-other \\
                                --classification /path/to/classification.csv
        """)
    
    parser.add_argument('--directory', '-d', type=str, required=True,
                       help='Path to directory containing .lan files')
    parser.add_argument('--trunc-level', '-L', type=int, default=15,
                       help='Number of persistence levels to keep (default 15)')
    parser.add_argument('--grid-type', type=str, default='critical',
                       choices=['critical', 'uniform'],
                       help='Type of grid: critical (critical points) or uniform (default critical)')
    parser.add_argument('--grid-size', '-g', type=int, default=None,
                       help='Size of uniform grid (required if grid-type is uniform)')
    parser.add_argument('--integrate', '-I', action='store_true',
                       help='Integrate the landscapes')
    parser.add_argument('--exclude-other', action='store_true',
                       help='Exclude samples labeled "Other" in classification')
    parser.add_argument('--classification', '-c', type=str, default=None,
                       help='Path to classification CSV (required if --exclude-other is used)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (defaults to ../data)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.grid_type == 'uniform' and args.grid_size is None:
        print("Error: --grid-size must be specified when --grid-type is 'uniform'", file=sys.stderr)
        sys.exit(1)
    
    if args.exclude_other and args.classification is None:
        print("Error: --classification must be specified when --exclude-other is used", file=sys.stderr)
        sys.exit(1)
    
    # Resolve paths
    directory = pathlib.Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    
    classification_csv = None
    if args.classification:
        classification_csv = pathlib.Path(args.classification)
        if not classification_csv.exists():
            print(f"Error: Classification CSV not found: {classification_csv}", file=sys.stderr)
            sys.exit(1)
    
    # Process landscapes
    try:
        result = process_landscapes(
            directory=str(directory),
            trunc_level=args.trunc_level,
            grid_type=args.grid_type,
            grid_size=args.grid_size,
            integrate=args.integrate,
            exclude_other=args.exclude_other,
            classification_csv=str(classification_csv) if classification_csv else None,
            output_dir=args.output_dir,
        )
        
        # Save results
        save_landscapes(result)
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Directory: {directory}")
        print(f"Truncation level: {args.trunc_level}")
        print(f"Grid type: {args.grid_type}")
        if args.grid_type == 'uniform':
            print(f"Grid size: {args.grid_size}")
        print(f"Integrated: {args.integrate}")
        print(f"Excluded 'Other': {args.exclude_other}")
        print(f"Output file: {result['filepath'].name}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()