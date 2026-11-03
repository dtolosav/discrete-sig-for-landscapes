"""Recreate the paper's tables and figures from immutable campaign records.

The bundle deliberately omits restart centres. We verify every bundled file,
recompute scores from assignments and check minimum-objective selection. This
does not repeat optimisation or certify the omitted centre files.
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.input_audit import sha256_file
from src.revision_experiment import save_json
from src.revision_reporting import depth_pairs

NAMES = dict(dlfm='DLFM', chen='Ordinary Chen', time_chen='Time-augmented Chen',
             integrated_chen='Integrated Chen', flattened='Flattened landscapes',
             functional_l1=r'Functional $L_1$ $k$-medians')


def read(path):
    return json.loads(Path(path).read_text())


def campaign_rows(bundle, name, selected):
    """Summarise original records without changing the original manifest."""
    directory = Path(bundle) / name
    identity = read(directory / 'manifest.json')['identity']
    if name == 'timing':
        # The original timing manifest records chain order but not class labels.
        reference = read(Path(bundle) / 'x1/manifest.json')['identity']
        assert identity['chains'] == reference['chains']
        assert identity['input_sha256'] == reference['input_sha256']
        classes = reference['classes']
    else:
        classes = identity['classes']
    specs = {s['id']: s for s in identity['specifications']}
    rows, all_fits = [], {}
    for spec_id in selected:
        spec = specs[spec_id]
        folder = directory / spec_id
        summary = read(folder / 'summary.json')
        assert summary['specification'] == spec
        fits = [read(folder / f'run-{i:03d}.json') for i in range(summary['runs'])]
        expected_seeds = identity['start_seeds'][:summary['runs']]
        assert len(expected_seeds) == len(fits)
        scores = []
        for fit, seeds in zip(fits, expected_seeds):
            assert [s['seed'] for s in fit['starts']] == seeds
            best = int(np.argmin([s['objective'] for s in fit['starts']]))
            assert best == fit['best_start']
            assert fit['labels'] == fit['starts'][best]['labels']
            assert fit['objective'] == fit['starts'][best]['objective']
            scores.append([adjusted_rand_score(classes, fit['labels']),
                           normalized_mutual_info_score(classes, fit['labels'], average_method='arithmetic')])
            np.testing.assert_allclose(scores[-1], [fit['ARI'], fit['NMI']], rtol=0, atol=1e-14)
        scores = np.array(scores)
        row = dict(spec, runs=len(fits), initialisations=len(expected_seeds[0]),
                   raw_dimension=summary['raw_dimension'], retained_dimension=summary['retained_dimension'],
                   representation_seconds=summary['representation_seconds'],
                   clustering_median_seconds=float(np.median([f['seconds'] for f in fits])))
        for j, metric in enumerate(('ARI', 'NMI')):
            row[metric + '_mean'] = float(scores[:, j].mean())
            row[metric + '_std'] = float(scores[:, j].std(ddof=0))
            np.testing.assert_allclose([row[metric + '_mean'], row[metric + '_std']],
                [summary[metric + '_mean'], summary[metric + '_std']], rtol=0, atol=1e-14)
        rows.append(row)
        all_fits[spec_id] = fits
    return identity, rows, all_fits


def write_table(output, name, rows, latex):
    with (output / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / (name + '.tex')).write_text('\n'.join(latex) + '\n')


def score(row, metric):
    return f"${row[metric + '_mean']:.4f} \\pm {row[metric + '_std']:.4f}$"


def split_observations(bundle, identity, fits):
    """Recompute the appendix's descriptive observations within 4qef_A."""
    import re
    import fruits
    low = min((f for f in fits if f['ARI'] < .65), key=lambda f: f['objective'])
    high = min((f for f in fits if f['ARI'] > .9), key=lambda f: f['objective'])
    members = np.array(identity['classes']) == '4qef_A'
    assignments = np.array(low['labels'])[members]
    groups = sorted(set(assignments), key=lambda j: -sum(assignments == j))
    assert len(groups) == 2
    chains = np.array(identity['chains'])[members]
    with (bundle / 'reference/x6_membership.csv').open(newline='') as stream:
        records = {r['Chain']: r for r in csv.DictReader(stream)}
    with (bundle / 'reference/trefoil_list.csv').open(newline='') as stream:
        classification = {r['Chain']: r for r in csv.DictReader(stream)}
    medians = {}
    for letter, group in zip('AB', groups):
        selected = chains[assignments == group]
        assert all(records[c]['Group'] == letter for c in selected)
        values = {key: float(np.median([float(classification[c][key]) for c in selected]))
                  for key in ('Length', 'N-tail', 'C-tail', 'Depth')}
        values['Level 11 area'] = float(np.median([float(records[c]['Level 11 area']) for c in selected]))
        medians[letter] = dict(size=len(selected), **values)
    with np.load(bundle / 'x6/primary-dlfm/features.npz') as archive:
        features, keep = archive['scaled'][members], archive['keep']
    delta = features[assignments == groups[1]].mean(0) - features[assignments == groups[0]].mean(0)
    iss = fruits.ISS(fruits.words.of_weight(3, dim=15), mode=fruits.ISSMode.EXTENDED)
    highest = [max(int(a or b) for a, b in re.findall(r'\((\d+)\)|([1-9])', iss.label(i)))
               for i in range(iss.n_iterated_sums()) if keep[i]]
    energy = delta**2
    return dict(groups=medians, later_levels_fraction=float(energy[np.array(highest) >= 11].sum() / energy.sum()),
                objective_relative_difference=abs(low['objective'] - high['objective']) / high['objective'])


def report_paper(bundle, output):
    bundle, output = Path(bundle), Path(output)
    index = read(bundle / 'index.json')
    for relative, expected in index['sha256'].items():
        assert sha256_file(bundle / relative) == expected, relative
    output.mkdir(parents=True, exist_ok=True)
    campaigns = {name: campaign_rows(bundle, name, selected) for name, selected in index['scopes'].items()}
    x1, rows, fits = campaigns['x1']
    l1, l1rows, _ = campaigns['x1_l1']
    assert x1['chains'] == l1['chains'] and x1['classes'] == l1['classes']
    assert len(rows) == 5 and len(l1rows) == 1
    assert all(r['runs'] == 100 for r in rows) and l1rows[0]['runs'] == 10
    timing = {r['method']: r for r in campaigns['timing'][1] + l1rows}
    primary = []
    latex = []
    for row in rows + l1rows:
        method = row['method']
        times = timing[method]
        assert times['runs'] == 10 and times['initialisations'] == 10
        entry = dict(row, representation_seconds=times['representation_seconds'],
                     clustering_median_seconds=times['clustering_median_seconds'])
        primary.append(entry)
        dimensions = '--' if method == 'functional_l1' else f"{row['raw_dimension']:,} / {row['retained_dimension']:,}"
        encoding = '--' if method == 'functional_l1' else f"{times['representation_seconds']:,.2f}"
        name = NAMES[method]
        if method == 'time_chen':
            name = r'\shortstack[l]{Time-augmented\\Chen}'
        elif method == 'functional_l1':
            name = r'\shortstack[l]{Functional $L_1$\\$k$-medians}'
        latex.append(' & '.join([name, score(row, 'ARI'), score(row, 'NMI'), dimensions,
                                 encoding, f"{times['clustering_median_seconds']:,.3f}"]) + r' \\')
    write_table(output, 'x1_summary_rows', primary, latex)
    x6, appendix, x6fits = campaigns['x6']
    assert x6['chains'] == x1['chains'] and len(appendix) == 23
    assert all(r['runs'] == 100 and r['initialisations'] == 10 for r in appendix)
    write_table(output, 'x6_comparison_rows', appendix[:6], [
        f"{NAMES[r['method']]} & {score(r, 'ARI')} & {score(r, 'NMI')} " + r'\\' for r in appendix[:6]])
    write_table(output, 'x6_sensitivity_rows', appendix[6:], [
        ' & '.join([NAMES[r['method']], str(r['levels']),
                    str(r['weight']) if r['method'] not in ('flattened', 'functional_l1') else '--',
                    str(r['points']) if r['method'] == 'flattened' else '--',
                    score(r, 'ARI'), score(r, 'NMI')]) + r' \\' for r in appendix[6:]])
    with np.load(bundle / 'x1/primary-dlfm/features.npz') as archive:
        features = archive['scaled']
    _, depth = depth_pairs(features, x1['depth'], x1['classes'])
    groups = {'all': 'All pairs', 'within_classes': 'Within classes', 'between_classes': 'Between classes'}
    depthrows = [dict(group=groups.get(key, key.removeprefix('class:')), **value) for key, value in depth.items()]
    write_table(output, 'x1_depth_rows', depthrows, [
        r['group'].replace('_', r'\_') + f" & {r['pairs']:,} & {r['spearman_rho']:.3f} " + r'\\'
        for r in depthrows])
    from src.paper_plots import plot_x1, plot_x6
    metadata = dict(x1=plot_x1(x1, fits['primary-dlfm'], features, output),
                    x6=plot_x6(x6, x6fits['primary-dlfm'], output), depth=depth,
                    x6_split=split_observations(bundle, x6, x6fits['primary-dlfm']))
    # PCA signs depend on the implementation; the pinned environment must match
    # the actual plotted coordinates as well as the explained variances.
    with np.load(bundle / 'reference/pca_coordinates.npz') as original, np.load(output / 'pca_coordinates.npz') as actual:
        np.testing.assert_array_equal(actual['chains'], original['chains'])
        np.testing.assert_allclose(actual['coordinates'], original['coordinates'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual['explained_variance_ratio'], original['explained_variance_ratio'], rtol=1e-10, atol=1e-12)
    reference = read(bundle / 'reference/figure_metadata.json')
    assert metadata['x1']['selected_run_zero_based'] == reference['selected_run_zero_based']
    save_json(output / 'paper_metadata.json', metadata)
    print('Verified 29 configurations; wrote four tables and three computational figures.')
