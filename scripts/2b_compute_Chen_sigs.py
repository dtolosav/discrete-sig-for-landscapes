"""Compute signatures for landscapes.

Loads an NPZ, computes the signature of each 
landscape using `pysiglib.sig`, stacks results into an array of shape
(num_landscapes, signature_length), and writes the output to the
project `results/` directory.

Usage:
  python scripts/2b_compute_Chen_sigs.py --input path/to/lans.npz --weight 3
"""

import argparse
import sys
import numpy as np
import pathlib
from datetime import datetime
import json
from pysiglib import sig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', type=str, default=None,
        help='Path to lans NPZ (optional). If omitted, the latest file in results/ is used.',
    )
    parser.add_argument(
        '--weight', '-w', type=int, default=3,
        help='Signature truncation weight (integer).',
    )
    parser.add_argument(
        '--csv', action='store_true', help='Also save signatures as CSV'
    )
    args = parser.parse_args()

    # start timing
    start_time = datetime.now()

    project_root = pathlib.Path(__file__).resolve().parent.parent
    results_dir = project_root / 'results'
    data_dir = project_root / 'data'

    if args.input:
        npz_path = pathlib.Path(args.input)
    else:
        print('No NPZ found in data/. Provide --input', file=sys.stderr)
        sys.exit(1)
        
    print(f'Loading landcapes NPZ: {npz_path}')
    data = np.load(npz_path)

    if 'result_array' in data:
        lans = data['result_array']
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
        raise ValueError('lans array must be 3D: (num_landscapes, len_grid, num_levels)')

    num_landscapes = lans.shape[0]

    print(f'Landscapes loaded: {num_landscapes} landscapes; shape {lans.shape[1:]} per landscape')

    # sigs = sig(lans, args.weight)

    sigs = []
    for i in range(num_landscapes):
        L = lans[i].T  # shape after transposing (len_grid, num_levels)
        s = sig(L, args.weight)
        sigs.append(np.asarray(s))

    # ensure consistent length
    sigs = [np.atleast_1d(s) for s in sigs]
    sig_len = max(s.shape[0] for s in sigs)
    sig_array = np.zeros((num_landscapes, sig_len), dtype=float)
    for i, s in enumerate(sigs):
        sig_array[i, : s.shape[0]] = s

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = npz_path.stem
    out_base = f'signatures_{base}_W{args.weight}_{ts}'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_npz = results_dir / f'{out_base}.npz'
    np.savez(out_npz, signatures=sig_array, weight=args.weight)
    print(f'Saved signatures NPZ to: {out_npz} (shape {sig_array.shape})')

    # record end time and write metadata
    end_time = datetime.now()
    meta = {
        'script': 'compute_lan_sigs.py',
        'input': str(npz_path),
        'output': str(out_npz),
        'weight': args.weight,
        'start': start_time.isoformat(),
        'end': end_time.isoformat(),
        'duration_sec': (end_time - start_time).total_seconds()
    }
    meta_path = results_dir / f"{out_base}_meta.json"
    with open(meta_path, 'w') as mf:
        json.dump(meta, mf, indent=2)
    print(f'Wrote metadata to {meta_path}, duration: {meta["duration_sec"]:.2f} seconds')

    if args.csv:
        out_csv = results_dir / f'{out_base}.csv'
        np.savetxt(out_csv, sig_array, delimiter=',')
        print(f'Saved signatures CSV to: {out_csv}')


if __name__ == '__main__':
    main()
