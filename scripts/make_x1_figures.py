"""Verified x1 author-review figures and primary uniform-grid error audit.

No clustering is refitted. The displayed partition is nearest the median saved
clustering objective, independently of sequence labels. PCA is descriptive.
"""
import argparse
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.revision_reporting import collect_summaries, depth_pairs
from src.revision_experiment import save_json, save_npz
from src.input_audit import sha256_file
from src.process_lan import load_lan
from src.revision_features import evaluate, reconstruction_errors
from src.functional_l1 import as_functional, l1_distance


def plot_pca(xyz, ratios, classes, depths):
    """Draw identical PCA coordinates with two colourings and aligned legends."""
    names = sorted(set(classes))
    colors = plt.get_cmap('tab10').colors
    fig = plt.figure(figsize=(10, 5.6))
    ax1 = fig.add_axes([.015, .25, .40, .68], projection='3d')
    ax2 = fig.add_axes([.545, .25, .40, .68], projection='3d')
    for name, color in zip(names, colors):
        ax1.scatter(*xyz[classes == name].T, s=19, color=color, label=name, alpha=.8, depthshade=False)
    for name, mask, color in [('Shallow (<0.005)', depths < .005, '#357db0'),
                               ('Intermediate', (depths >= .005) & (depths <= .05), '#a9adb3'),
                               ('Deep (>0.05)', depths > .05, '#d05a37')]:
        ax2.scatter(*xyz[mask].T, s=19, color=color, label=name, alpha=.8, depthshade=False)
    for ax in (ax1, ax2):
        ax.set_xlabel(f'PC1 ({ratios[0]:.1%})')
        ax.set_ylabel(f'PC2 ({ratios[1]:.1%})')
        ax.set_zlabel(f'PC3 ({ratios[2]:.1%})')
        ax.view_init(elev=23, azim=-58)
    ax1.set_title('Sequence-similarity class')
    ax2.set_title('Knot-depth category')
    ax1.legend(fontsize=10, ncol=3, loc='upper center',
               bbox_to_anchor=(.215, .20), bbox_transform=fig.transFigure)
    ax2.legend(fontsize=10, loc='upper center',
               bbox_to_anchor=(.745, .20), bbox_transform=fig.transFigure)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign, out = args.run_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest, summaries = collect_summaries(campaign)
    assert manifest['config']['cohort'] == 'x1' and len(summaries) == 5
    assert all(s['runs'] == 100 and s['initialisations'] == 10 for s in summaries)
    classes = np.asarray(manifest['classes'])
    depths = np.asarray(manifest['depth'])
    names = sorted(set(classes))
    colors = plt.get_cmap('tab10').colors
    fits = [json.loads((campaign / 'primary-dlfm' / f'run-{i:03d}.json').read_text()) for i in range(100)]
    objectives = np.array([f['objective'] for f in fits])
    chosen = int(np.argmin(np.abs(objectives - np.median(objectives))))
    labels = np.array(fits[chosen]['labels'])
    order = sorted(set(labels), key=lambda j: (-sum(labels == j), j))
    counts = np.array([[sum((labels == j) & (classes == c)) for j in order] for c in names])
    ari = np.array([f['ARI'] for f in fits])
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42})

    def save(fig, name):
        for extension in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{extension}', dpi=190, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout='constrained',
                             gridspec_kw={'width_ratios': [1.35, 1]})
    bottom = np.zeros(len(order))
    for name, row, color in zip(names, counts, colors):
        axes[0].bar(np.arange(len(order)), row, bottom=bottom, label=name, color=color)
        bottom += row
    axes[0].set(xticks=np.arange(9), xticklabels=np.arange(1, 10),
                xlabel='Cluster (ordered by size)', ylabel='Number of proteins',
                title=f'Median-objective solution (ARI {ari[chosen]:.3f})', ylim=(0, 1.08 * max(bottom)))
    axes[0].legend(title='Sequence class', ncol=3, fontsize=8,
                   loc='upper right', frameon=False)
    # Include every run, including rare lower-agreement outcomes.
    lower = max(0., np.floor(ari.min() * 20) / 20)
    hist_counts, _, _ = axes[1].hist(ari, bins=np.linspace(lower, 1, 26), color='#397fa0', edgecolor='white')
    assert hist_counts.sum() == 100
    axes[1].axvline(ari[chosen], color='#b13e32', ls='--', label='Displayed solution')
    axes[1].set(xlim=(lower, 1), xlabel='Adjusted Rand Index', ylabel='Number of runs',
                title='All 100 ten-start runs')
    axes[1].legend(fontsize=9, loc='upper left', frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'x1_cluster_composition')

    with np.load(campaign / 'primary-dlfm/features.npz') as archive:
        features = archive['scaled']
    pca = PCA(n_components=3, svd_solver='full')
    xyz = pca.fit_transform(features)
    ratios = pca.explained_variance_ratio_
    fig = plot_pca(xyz, ratios, classes, depths)
    save(fig, 'x1_pca_views')
    save_npz(out / 'pca_coordinates.npz', coordinates=xyz, explained_variance_ratio=ratios,
             chains=manifest['chains'])

    pairs, depth = depth_pairs(features, depths, classes)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), layout='constrained')
    masks = [np.ones(len(pairs['i']), dtype=bool), pairs['within_class'], ~pairs['within_class']]
    for ax, name, title, mask in zip(axes, ('all', 'within_classes', 'between_classes'),
                                    ('All pairs', 'Within sequence classes', 'Between sequence classes'), masks):
        plot = ax.hexbin(pairs['signature_distance'][mask], pairs['depth_difference'][mask],
                        gridsize=25, mincnt=1, cmap='Blues', norm=LogNorm())
        fig.colorbar(plot, ax=ax, label='Pair count (log scale)', shrink=.8)
        ax.set(xlabel='Standardised DLFM distance', title=f'{title}\nSpearman ρ = {depth[name]["spearman_rho"]:.3f}')
    axes[0].set_ylabel('Absolute knot-depth difference')
    save(fig, 'x1_depth_association')

    captions = {
        'x1_cluster_composition': 'DLFM clustering of 363 x1 protein chains at 15 landscape levels and weight 3. Left: the saved solution whose objective is closest to the median objective across all 100 ten-start runs, taking the first run in case of a tie. This illustrates a typical objective value and uses no sequence labels for selection. Each run still selects its best of ten starts by minimum objective. Bars show sequence-class counts and clusters are ordered by size. Right: ARI values for every run, including the lower-agreement outcomes; the dashed line identifies the displayed solution.',
        'x1_pca_views': f'First three principal components of the standardised x1 DLFM vectors, explaining {sum(ratios):.1%} of variance. Coordinates are identical in both panels; colours indicate sequence class and knot-depth category. PCA uses no class or depth labels and is not used as input to clustering.',
        'x1_depth_association': 'Standardised x1 DLFM distances and absolute knot-depth differences for all unordered pairs, pairs within sequence classes, and pairs between classes. Each distinct pair is counted once. Shading and colour bars show pair counts on logarithmic scales. Axis ranges differ between panels to show the variation within each group. Correlations are descriptive; pairs sharing a protein are not independent observations.'}
    for name, caption in captions.items():
        (out / f'{name}_caption.txt').write_text(caption + '\n')
    metadata = dict(selected_run_zero_based=chosen, selection='closest to median clustering objective; first tie',
                    median_objective=float(np.median(objectives)),
                    ARI=fits[chosen]['ARI'], NMI=fits[chosen]['NMI'],
                    objective=fits[chosen]['objective'], counts=counts, class_order=names,
                    cluster_order=[int(j) for j in order], runs_ARI_above_09=int(sum(ari > .9)),
                    ARI_min=float(ari.min()), ARI_median=float(np.median(ari)), ARI_max=float(ari.max()),
                    explained_variance_ratio=ratios, cumulative_variance=float(sum(ratios)), depth=depth,
                    campaign_manifest_sha256=sha256_file(campaign / 'manifest.json'),
                    script_sha256=sha256_file(Path(__file__)),
                    features_sha256=sha256_file(campaign / 'primary-dlfm/features.npz'))
    save_json(out / 'figure_metadata.json', metadata)
    print(json.dumps({k: metadata[k] for k in ('selected_run_zero_based', 'ARI', 'NMI', 'runs_ARI_above_09', 'cumulative_variance')}, indent=2), flush=True)

    # The x6 approximation errors cannot be reused for the x1 baseline.
    source = ROOT / 'data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x1'
    assert all(sha256_file(source / f'{c}_1.lan') == expected for c, expected in manifest['input_sha256'].items())
    landscapes = [load_lan(source / f'{c}_1.lan', 15) for c in manifest['chains']]
    support = manifest['config']['support_L']
    grid = np.linspace(0, support, 1024)
    sampled = evaluate(landscapes, grid)
    sup = reconstruction_errors(landscapes, grid)
    functional = as_functional(landscapes, support)
    mass = np.array([sum(np.trapz(y, x) for x, y in lan) for lan in functional])
    assert np.all(mass > 0)
    absolute = np.array([sum(l1_distance(curve, (grid, sampled[i, r])) for r, curve in enumerate(lan))
                         for i, lan in enumerate(functional)])
    audit = dict(cohort='x1', points=1024, support_L=support,
                 max_sup_error=float(sup.max()), max_relative_L1_error=float((absolute / mass).max()),
                 chains=manifest['chains'], sup_error=sup, L1_error=absolute, landscape_mass=mass,
                 manifest_sha256=metadata['campaign_manifest_sha256'])
    save_json(out / 'uniform_grid_audit.json', audit)
    save_json(out / 'artifact_sha256.json', {p.name: sha256_file(p) for p in sorted(out.iterdir())
               if p.is_file() and p.name != 'artifact_sha256.json'})
    print('Figures and x1 uniform-grid audit complete.', flush=True)


if __name__ == '__main__':
    main()
