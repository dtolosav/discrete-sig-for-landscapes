"""Compute discrete signatures for landscapes.

Loads an NPZ, computes the signature of each 
landscape using FRUITS, stacks results into an array of shape
(num_landscapes, signature_length), and writes the output to the
project `results/` directory.

Usage:
  python scripts/2a_compute_discrete_sigs.py --input path/to/lans.npz --weight 3 --csv
"""

# import all dependencies
import fruits
import matplotlib.pyplot as plt
import numpy as np
import re
from IPython.display import display, Math, Latex
import os
from datetime import datetime
import argparse
import sys
import pathlib
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to lans NPZ.')
    parser.add_argument('--weight', '-w', type=int, default=3,
                        help='Signature truncation weight (integer).')
    parser.add_argument('--csv', action='store_true', help='Also save signatures as CSV')
    args = parser.parse_args()
    start_time = datetime.now()

    project_root = pathlib.Path(__file__).resolve().parent.parent
    results_dir = project_root / 'results'
    data_dir = project_root / 'data'

    if args.input:
        npz_path = pathlib.Path(args.input)
    else:
        print('No NPZ found in data/. Provide --input', file=sys.stderr)
        sys.exit(1)

    print(f'Loading landscapes NPZ: {npz_path}')
    data = np.load(npz_path, allow_pickle=True)

    if 'result_array' in data:
        lans = data['result_array']
    elif 'lans' in data:
        lans = data['lans']
    else:
        # fall back to common keys
        for k in data.files:
            if 'lans' in k or 'Lans' in k:
                lans = data[k]
                break
        else:
            raise KeyError('Could not find landscape array in NPZ (expected key result_array)')

    lans = np.asarray(lans)
    if lans.ndim != 3:
        raise ValueError('lans array must be 3D: (num_landscapes, num_levels, len_grid)')

    num_landscapes = lans.shape[0]
    trunc_level = lans.shape[1]
    print(f'Landscapes loaded: {num_landscapes} landscapes; shape {lans.shape[1:]} per landscape')

    # compute signatures for all landscapes
    fruit = fruits.Fruit("my_fruit")
    fruit.add(fruits.preparation.transform.INC)
    Sigma = fruits.ISS(
        fruits.words.of_weight(args.weight, dim=trunc_level),
        mode=fruits.ISSMode.EXTENDED,
    )
    fruit.add(Sigma)
    fruit.add(fruits.sieving.END)

    fruit.fit(lans)
    signatures = fruit.transform(lans)
    print(f'{fruit.summary}')

    print(f'Computed signatures shape: {signatures.shape}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f'discrete_signatures_weight{args.weight}_{npz_path.stem}_{timestamp}'
    output_path = results_dir / f'{output_name}.npz'
    np.savez(output_path, signatures=signatures, weight=args.weight,
             ordering=data.get('ordering'), global_epsilon=data.get('global_epsilon'),
             allow_pickle=True)
    print(f'Signatures saved to NPZ: {output_path}')

    # record end time and write metadata
    end_time = datetime.now()
    meta = {
        'script': 'compute_discrete_sigs.py',
        'input': str(npz_path),
        'output': str(output_path),
        'weight': args.weight,
        'start': start_time.isoformat(),
        'end': end_time.isoformat(),
        'duration_sec': (end_time - start_time).total_seconds()
    }
    meta_path = results_dir / f"{output_name}_meta.json"
    with open(meta_path, 'w') as mf:
        json.dump(meta, mf, indent=2)
    print(f'Wrote metadata to {meta_path}, duration: {meta["duration_sec"]:.2f} seconds')

    if args.csv:
        out_csv = results_dir / f'{output_name}.csv'
        np.savetxt(out_csv, signatures, delimiter=',')
        print(f'Saved signatures CSV to: {out_csv}')


if __name__ == '__main__':
    main()
