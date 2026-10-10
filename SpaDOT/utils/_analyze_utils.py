import os
import anndata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import wot # Waddington OT, original package
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

def KMeans_Clustering(adata, n_clusters, seed=2026, n_init=10):
    """
    Perform KMeans clustering on the spatial data.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the spatial data.
    n_clusters : int
        The number of clusters to form.
    
    Returns
    -------
    adata : anndata.AnnData
        The AnnData object with the clustering results added to the obs.
    """
    tps = sorted(adata.obs['timepoint'].unique())
    tp_adata_list = []
    for i, tp in enumerate(tps):
        # perform KMeans clustering
        tp_adata = adata[adata.obs['timepoint'] == tp].copy()
        tp_kmeans = KMeans(n_clusters=n_clusters[i],
                           random_state=seed, n_init=n_init,
                           init='k-means++', algorithm='lloyd').fit(tp_adata.X)
        tp_adata.obs['kmeans'] = tp_kmeans.labels_.astype(str)
        tp_adata_list.append(tp_adata)
    # merge all timepoints
    merged_adata = anndata.concat(tp_adata_list, axis=0)
    return merged_adata


def Adaptive_clustering(args, adata, min_clusters=4, max_clusters=20, wss_threshold=0.1):
    """
    Perform adaptive clustering using KMeans and calculate WSS (Within-Cluster Sum of Squares).
    Then calculate the difference between consecutive clusters and the ratio between consecutive differences.

    Parameters:
    -----------
    adata : anndata.AnnData
        The AnnData object containing the data to cluster.
    max_clusters : int, optional
        The maximum number of clusters to test. Default is 20.
    min_clusters : int, optional
        The minimum number of clusters to test. Default is 5.

    Returns:
    --------
    dict
        A dictionary containing WSS values, differences, and ratios for each number of clusters.
    """

    seed = getattr(args, 'seed', 2026)
    n_init = getattr(args, 'n_init', 10)
    tps = sorted(adata.obs['timepoint'].unique())
    tp_adata_list = []
    for i, tp in enumerate(tps):
        # perform KMeans clustering
        tp_adata = adata[adata.obs['timepoint'] == tp].copy()
        # Store WSS for each number of clusters
        wss = []
        # Perform KMeans clustering for each number of clusters
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init,
                            init='k-means++', algorithm='lloyd').fit(tp_adata.X)
            wss.append(kmeans.inertia_)
        # Calculate wss and ratios
        wss_diff = -np.diff(wss)
        wss_diff_ratios = [wss_diff[i] / wss_diff[i + 1] for i in range(len(wss_diff)-1)]
        wss_df = pd.DataFrame({
            'clusters': range(min_clusters, max_clusters + 1),
            'wss': wss,
            'wss_diff': [None] + list(wss_diff),
            'wss_diff_ratio': [None] + list(wss_diff_ratios) + [None]
        })
        wss_range = wss_df['wss'].max() - wss_df['wss'].min()
        wss_diff_threshold = wss_threshold * wss_range
        filtered_wss_df = wss_df[wss_df['wss_diff'] > wss_diff_threshold]
        max_idx = filtered_wss_df['wss_diff_ratio'].idxmax()
        wss_cluster = filtered_wss_df['clusters'][max_idx]
        highlight_wss = filtered_wss_df['wss'][max_idx]
        # plot wss per cluster
        plt.figure(figsize=(10, 6))
        plt.plot(wss_df['clusters'], wss_df['wss'], marker='o')
        plt.scatter(wss_cluster, highlight_wss, color='red', s=100, label='Selected Cluster')
        plt.title('WSS vs Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WSS')
        plt.xticks(wss_df['clusters'])
        plt.grid()
        plt.savefig(args.output_dir+os.sep+args.prefix+str(tp)+'_WSS_vs_Clusters.png')
        plt.close()
        tp_adata.obs['kmeans'] = KMeans(n_clusters=wss_cluster,
                                        random_state=seed, n_init=n_init,
                                        init='k-means++', algorithm='lloyd').fit(tp_adata.X).labels_.astype(str)
        tp_adata_list.append(tp_adata)
    # merge all timepoints
    merged_adata = anndata.concat(tp_adata_list, axis=0)
    return merged_adata


def OT_analysis(args, adata):
    """
    Perform optimal transport analysis on the spatial data.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the spatial data.    
    Returns
    -------
    adata : anndata.AnnData
        The AnnData object with the OT analysis results added to the obs.
    """
    adata.obs['day'] = adata.obs['timepoint'].astype('category').cat.codes
    adata.obs['cell_growth_rate'] = 1 # initialize as 1
    # larger epsilon0 to avoid overflow
    ot_model = wot.ot.OTModel(adata, epsilon = 0.05, epsilon0 = 1, lambda1 = 0.1,lambda2 = 5, growth_iters=3) # imbalanced OT with imbalanced row and comparatiely balanced target
    ot_model.compute_all_transport_maps(tmap_out=args.output_dir+os.sep+'OT') # compute transport maps
    tmap_model = wot.tmap.TransportMapModel.from_directory(args.output_dir+os.sep+'OT')
    # generate region dict
    adata.obs['SpaDOT_pred_labels'] = adata.obs['timepoint'].astype('str')+'_'+adata.obs['kmeans'].astype('str')
    latent_cell_sets = adata.obs.groupby('SpaDOT_pred_labels').apply(lambda x: x.index.tolist()).to_dict()
    days = sorted(adata.obs['day'].unique())
    for tp_i in range(len(days)-1):
        prev_day = days[tp_i]
        next_day = days[tp_i+1]
        prev_day_populations = tmap_model.population_from_cell_sets(latent_cell_sets, at_time=prev_day)
        next_day_populations = tmap_model.population_from_cell_sets(latent_cell_sets, at_time=next_day)
        transition_table = tmap_model.transition_table(prev_day_populations, next_day_populations) # aggregated OT matrix
        transition_table.write_h5ad(args.output_dir+os.sep+args.prefix+'transition_table_'+str(prev_day)+'_'+str(next_day)+'.h5ad')

def _cluster_key(value):
    """Sort numeric cluster IDs numerically, with a stable string fallback."""
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def _annotation_matches(adata):
    """Match each predicted cluster to its most frequent annotation for display only."""
    if 'annotation' not in adata.obs:
        return {}, {}
    labels = adata.obs['annotation']
    valid = labels.notna() & labels.astype(str).str.strip().ne('')
    names = sorted(labels[valid].astype(str).unique())
    if not names:
        return {}, {}
    if len(names) <= 10:
        colors = sns.color_palette('tab10', 10)
    elif len(names) <= 20:
        colors = sns.color_palette('tab20', 20)
    else:
        colors = sns.husl_palette(len(names))
    palette = {name: colors[i] for i, name in enumerate(names)}
    obs = adata.obs.loc[valid, ['timepoint', 'kmeans', 'annotation']].astype(str)
    counts = obs.groupby(['timepoint', 'kmeans', 'annotation'], observed=True).size()
    matches = {}
    for (tp, cluster), group in counts.groupby(level=[0, 1], sort=False):
        options = sorted(((int(n), str(label)) for (_, _, label), n in group.items()),
                         key=lambda item: (-item[0], item[1]))
        matches[f'{tp}_{cluster}'] = options[0][1]
    return matches, palette


def plot_domains(args, adata):
    """Plot spatial domains in the original layout with annotation-matched colors."""
    matches, palette = _annotation_matches(adata)
    display_labels = getattr(args, 'display_labels', {}) or {}
    display_palette = getattr(args, 'display_palette', {}) or {}
    stage_titles = getattr(args, 'stage_titles', {}) or {}
    records = []
    domain_names = []
    for tp in adata.obs['timepoint'].unique():
        tp_adata = adata[adata.obs['timepoint'] == tp].copy()
        tp_adata.obs['pixel_x'] = tp_adata.obsm['spatial'][:, 0]
        tp_adata.obs['pixel_y'] = tp_adata.obsm['spatial'][:, 1]
        clusters = tp_adata.obs['kmeans'].astype(str)
        tp_adata.obs['kmeans'] = clusters
        cluster_ids = sorted(clusters.unique(), key=_cluster_key)
        fallback = sns.color_palette('tab10', max(10, len(cluster_ids)))
        colors = {}
        for i, cluster in enumerate(cluster_ids):
            domain = f'{tp}_{cluster}'
            label = display_labels.get(domain, matches.get(domain))
            visual_palette = display_palette if display_labels else palette
            colors[cluster] = visual_palette[label] if label in visual_palette else fallback[i]
            domain_names.append((str(tp), str(cluster), domain,
                                 display_labels.get(domain, domain),
                                 stage_titles.get(str(tp), str(tp)),
                                 matches.get(domain, '').replace('\n', ' ').replace('\r', ' ').strip()))
        hue = 'kmeans'
        if display_labels:
            hue = '_display_label'
            tp_adata.obs[hue] = clusters.map(
                lambda cluster: display_labels.get(f'{tp}_{cluster}', cluster))
            colors = {display_labels.get(f'{tp}_{cluster}', cluster): colors[cluster]
                      for cluster in cluster_ids}
        plt.figure(figsize=(5, 5))
        sns.scatterplot(data=tp_adata.obs, x='pixel_x', y='pixel_y',
                        hue=hue, palette=colors, s=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Time point: {}'.format(stage_titles.get(str(tp), tp)))
        plt.tight_layout()
        plt.savefig(args.output_dir+os.sep+args.prefix+str(tp)+'_domains.png')
        plt.close()
        for spot, cluster in zip(tp_adata.obs.index, clusters):
            label = matches.get(f'{tp}_{cluster}', '').replace('\n', ' ')
            records.append((spot, str(tp), cluster, label))
    pd.DataFrame(records, columns=['spot', 'timepoint', 'cluster', 'matched_annotation']).to_csv(
        os.path.join(args.output_dir, f'{args.prefix}domains.csv'), index=False)
    pd.DataFrame(domain_names, columns=['timepoint', 'cluster', 'domain_id',
                                        'display_name', 'stage', 'matched_annotation']).to_csv(
        os.path.join(args.output_dir, f'{args.prefix}domain_display_mapping.csv'), index=False)


def _transition_order(mass, rows, columns, row_order, matches=None):
    """Align matched labels first, then strongest flows; place extras near sources."""
    row_order = [name for name in row_order if name in rows] + [
        name for name in rows if name not in row_order]
    row_indices = [rows.index(name) for name in row_order]
    ordered_mass = mass[row_indices, :]
    alignment = ordered_mass.copy()
    if matches:
        for i, row in enumerate(row_order):
            for j, column in enumerate(columns):
                if matches.get(row) and matches.get(row) == matches.get(column):
                    alignment[i, j] += 2.0
    matched_rows, matched_cols = linear_sum_assignment(-alignment)
    anchors = {int(col): float(row) for row, col in zip(matched_rows, matched_cols)}
    for col in range(len(columns)):
        if col not in anchors:
            weight = ordered_mass[:, col]
            anchors[col] = (float(np.dot(np.arange(len(row_order)), weight) / weight.sum())
                            if weight.sum() else float(len(row_order)))
    col_order = [columns[i] for i in sorted(range(len(columns)),
                                           key=lambda i: (anchors[i], i not in matched_cols, _cluster_key(columns[i])))]
    return row_order, col_order

def plot_OT(args, adata):
    """Plot original transition-score dots and a numbered score matrix."""
    matches, palette = _annotation_matches(adata)
    display_labels = getattr(args, 'display_labels', {}) or {}
    display_palette = getattr(args, 'display_palette', {}) or {}
    stage_titles = getattr(args, 'stage_titles', {}) or {}
    visual_matches = display_labels if display_labels else matches
    visual_palette = display_palette if display_labels else palette
    days = sorted(adata.obs['day'].unique())
    prior_order = None
    for tp_i in range(len(days)-1):
        prev_day = days[tp_i]
        next_day = days[tp_i+1]
        transition_table = anndata.read_h5ad(args.output_dir+os.sep+args.prefix+'transition_table_'+str(prev_day)+'_'+str(next_day)+'.h5ad')
        mass = transition_table.X
        mass = mass.toarray() if hasattr(mass, 'toarray') else np.asarray(mass)
        mass = np.asarray(mass, dtype=float)
        rows = list(map(str, transition_table.obs_names))
        columns = list(map(str, transition_table.var_names))
        if prior_order is None:
            prior_order = sorted(rows, key=lambda name: (visual_matches.get(name, ''),
                                                           _cluster_key(name.rsplit('_', 1)[-1])))
        row_order, col_order = _transition_order(mass, rows, columns, prior_order, visual_matches)
        prior_order = col_order
        row_idx = [rows.index(name) for name in row_order]
        col_idx = [columns.index(name) for name in col_order]
        ordered_mass = mass[np.ix_(row_idx, col_idx)]
        row_sum = mass.sum(axis=1, keepdims=True)
        col_sum = mass.sum(axis=0, keepdims=True)
        row_conditional = np.divide(mass, row_sum, out=np.zeros_like(mass), where=row_sum != 0)
        col_conditional = np.divide(mass, col_sum, out=np.zeros_like(mass), where=col_sum != 0)
        ordered_score = np.minimum(row_conditional, col_conditional)[np.ix_(row_idx, col_idx)]
        base = os.path.join(args.output_dir, f'{args.prefix}transition_')
        pd.DataFrame(ordered_mass, index=row_order, columns=col_order).to_csv(
            f'{base}mass_ordered_{prev_day}_{next_day}.csv')
        pd.DataFrame(ordered_score, index=row_order, columns=col_order).to_csv(
            f'{base}probability_min_ordered_{prev_day}_{next_day}.csv')
        display_rows = [visual_matches.get(name, name) if display_labels else name for name in row_order]
        display_cols = [visual_matches.get(name, name) if display_labels else name for name in col_order]
        if display_labels:
            if len(set(display_rows)) != len(display_rows) or len(set(display_cols)) != len(display_cols):
                raise ValueError('Display names must be unique within each timepoint')
            pd.DataFrame(ordered_score, index=display_rows, columns=display_cols).to_csv(
                f'{base}probability_min_named_{prev_day}_{next_day}.csv')
        prev_key = rows[0].rsplit('_', 1)[0]
        next_key = columns[0].rsplit('_', 1)[0]
        prev_title = stage_titles.get(prev_key, prev_day)
        next_title = stage_titles.get(next_key, next_day)
        plt.figure(figsize=(ordered_score.shape[1] * 0.8, ordered_score.shape[0] * 0.8))
        for i in range(ordered_score.shape[0]):
            for j in range(ordered_score.shape[1]):
                value = ordered_score[i, j]
                color = 'grey' if value < 0.25 else plt.cm.Reds(value)
                plt.scatter(j, i, s=value * 500, c=[color], edgecolors='black', alpha=0.8)
        plt.xticks(range(ordered_score.shape[1]), display_cols, rotation=45, ha='right')
        plt.yticks(range(ordered_score.shape[0]), display_rows)
        ax = plt.gca()
        for tick, name in zip(ax.get_xticklabels(), col_order):
            tick.set_color(visual_palette.get(visual_matches.get(name), '#303030'))
        for tick, name in zip(ax.get_yticklabels(), row_order):
            tick.set_color(visual_palette.get(visual_matches.get(name), '#303030'))
        plt.xlabel('{} Domains'.format(next_title))
        plt.ylabel('{} Domains'.format(prev_title))
        plt.title('Transition Probability Dotplot')
        plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'),
                     label='Transition Probability', ax=ax)
        plt.tight_layout()
        plt.savefig(args.output_dir+os.sep+args.prefix+
                    f'transition_dotplot_{prev_day}_{next_day}.png')
        plt.close()
        fig, ax = plt.subplots(figsize=(max(5, len(columns) * 0.85),
                                        max(4, len(rows) * 0.75)))
        image = ax.imshow(ordered_score, cmap='Reds', origin='lower', vmin=0, vmax=1)
        ax.set_xticks(range(len(columns)), display_cols, rotation=45, ha='right')
        ax.set_yticks(range(len(rows)), display_rows)
        for tick, name in zip(ax.get_xticklabels(), col_order):
            tick.set_color(visual_palette.get(visual_matches.get(name), '#303030'))
        for tick, name in zip(ax.get_yticklabels(), row_order):
            tick.set_color(visual_palette.get(visual_matches.get(name), '#303030'))
        midpoint = 0.6
        for i in range(len(rows)):
            for j in range(len(columns)):
                value = ordered_score[i, j]
                ax.text(j, i, f'{value:.3g}', ha='center', va='center', fontsize=7,
                        color='white' if value > midpoint else 'black')
        ax.set_xlabel('{} Domains'.format(next_title))
        ax.set_ylabel('{} Domains'.format(prev_title))
        ax.set_title('Analyzed conditional transition score')
        fig.colorbar(image, ax=ax, label='Minimum conditional probability')
        fig.tight_layout()
        fig.savefig(args.output_dir+os.sep+args.prefix+
                    f'transition_probability_min_{prev_day}_{next_day}.png')
        plt.close(fig)
