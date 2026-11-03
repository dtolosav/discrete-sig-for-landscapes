"""The three computational figures in the revised paper (no refitting)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.revision_experiment import save_npz

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



def plot_x1(manifest, fits, features, out):
    classes = np.asarray(manifest['classes'])
    depths = np.asarray(manifest['depth'])
    names = sorted(set(classes))
    colors = plt.get_cmap('tab10').colors
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

    pca = PCA(n_components=3, svd_solver='full')
    xyz = pca.fit_transform(features)
    ratios = pca.explained_variance_ratio_
    fig = plot_pca(xyz, ratios, classes, depths)
    save(fig, 'x1_pca_views')
    save_npz(out / 'pca_coordinates.npz', coordinates=xyz, explained_variance_ratio=ratios,
             chains=manifest['chains'])

    return dict(selected_run_zero_based=chosen, ARI=fits[chosen]['ARI'], NMI=fits[chosen]['NMI'],
                counts=counts, runs_ARI_above_09=int(sum(ari > .9)),
                cumulative_variance=float(sum(ratios)), explained_variance_ratio=ratios)


def plot_x6(m, fits, out):
    classes=np.array(m['classes']);names=sorted(set(classes))
    ari=np.array([f['ARI'] for f in fits])
    low=np.flatnonzero(ari<.65);high=np.flatnonzero(ari>.9)
    selected=[min(g,key=lambda i:fits[i]['objective']) for g in (high,low)]
    plt.rcParams.update({'font.size':11,'axes.titlesize':11})
    fig=plt.figure(figsize=(8,6.6),layout='constrained')
    gs=fig.add_gridspec(2,2)
    axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0])]
    legend_ax=fig.add_subplot(gs[1,1]);legend_ax.axis('off')
    colors=plt.get_cmap('tab10').colors
    metadata=[]
    for ax,i,title,n in zip(axs,selected,['Higher sequence agreement','Lower sequence agreement'],[len(high),len(low)]):
        labels=np.array(fits[i]['labels'])
        order=sorted(set(labels),key=lambda j:(-sum(labels==j),j))
        counts=np.array([[sum((labels==j)&(classes==c)) for j in order] for c in names])
        bottom=np.zeros(9)
        for name,values,col in zip(names,counts,colors):
            ax.bar(np.arange(9),values,bottom=bottom,label=name,color=col,width=.8)
            bottom+=values
        ax.set(xticks=np.arange(9),xticklabels=np.arange(1,10),xlabel='Cluster (ordered by size)',
               ylim=(0,235),title=f'{title} ({n} runs)\nDisplayed ARI = {ari[i]:.3f}')
        ax.spines[['top','right']].set_visible(False)
        metadata.append(dict(run_zero_based=int(i),ARI=float(ari[i]),objective=fits[i]['objective'],
                             cluster_order=[int(j) for j in order],class_order=names,counts=counts.tolist()))
    axs[0].set_ylabel('Number of proteins')
    axs[2].hist(ari,bins=np.linspace(.55,1,19),color='#397fa0',edgecolor='white')
    axs[2].set(xlim=(.55,1),xlabel='Adjusted Rand Index',ylabel='Number of runs',title='All 100 ten-start runs')
    axs[2].spines[['top','right']].set_visible(False)
    handles,labels=axs[0].get_legend_handles_labels()
    legend_ax.legend(handles,labels,loc='upper left',ncol=2,
                     title='Sequence-similarity class',frameon=False,fontsize=10)
    legend_ax.text(.04,.10,f'{len(high)} higher-agreement runs\n{len(low)} lower-agreement runs\n{len(fits)-len(high)-len(low)} intermediate runs',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(out/f'x6_clustering_families.{ext}',dpi=200)
    plt.close(fig)
    return dict(selected=metadata, higher_runs=len(high), lower_runs=len(low), intermediate_runs=len(fits)-len(high)-len(low))
