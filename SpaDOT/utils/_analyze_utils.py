import os
import anndata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import wot # Waddington OT, original package
import matplotlib.pyplot as plt
import seaborn as sns

def KMeans_Clustering(adata, n_clusters):
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
    tps = adata.obs['timepoint'].unique()
    tps.sort()
    tp_adata_list = []
    for i, tp in enumerate(tps):
        # perform KMeans clustering
        tp_adata = adata[adata.obs['timepoint'] == tp].copy()
        tp_kmeans = KMeans(n_clusters=n_clusters[i], 
                           random_state=1993, 
                           n_init=10).fit(tp_adata.X)
        tp_adata.obs['kmeans'] = tp_kmeans.labels_.astype(str)
        tp_adata_list.append(tp_adata)
    # merge all timepoints
    merged_adata = anndata.concat(tp_adata_list, axis=0)
    return merged_adata

def Adaptive_Clustering(adata):
    """
    Perform adaptive clustering on the spatial data based on Elbow method.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the spatial data.    
    Returns
    -------
    adata : anndata.AnnData
        The AnnData object with the clustering results added to the obs.
    """
    
    # Perform adaptive clustering
    
    return adata

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
    ot_model = wot.ot.OTModel(adata, epsilon = 0.01, lambda1 = 0.1,lambda2 = 5, growth_iters=3) # imbalanced OT with imbalanced row and comparatiely balanced target
    ot_model.compute_all_transport_maps(tmap_out=args.output_dir+os.sep+'OT') # compute transport maps
    tmap_model = wot.tmap.TransportMapModel.from_directory(args.output_dir+os.sep+'OT')
    # generate region dict
    adata.obs['SpaDOT_pred_labels'] = adata.obs['timepoint'].astype('str')+'_'+adata.obs['kmeans'].astype('str')
    latent_cell_sets = adata.obs.groupby('SpaDOT_pred_labels').apply(lambda x: x.index.tolist()).to_dict()
    days = adata.obs['day'].unique()
    days.sort()
    for tp_i in range(len(days)-1):
        prev_day = days[tp_i]
        next_day = days[tp_i+1]
        prev_day_populations = tmap_model.population_from_cell_sets(latent_cell_sets, at_time=prev_day)
        next_day_populations = tmap_model.population_from_cell_sets(latent_cell_sets, at_time=next_day)
        transition_table = tmap_model.transition_table(prev_day_populations, next_day_populations) # aggregated OT matrix
        transition_table.write_h5ad(args.output_dir+os.sep+'transition_table_'+str(prev_day)+'_'+str(next_day)+'.h5ad')

def plot_domains(args, adata):
    """
    Plot the spatial domains of the cells.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the spatial data.    
    Returns
    -------
    None
    """
    for tp in adata.obs['timepoint'].unique():
        tp_adata = adata[adata.obs['timepoint'] == tp].copy()
        tp_adata.obs['pixel_x'] = tp_adata.obsm['spatial'][:, 0]
        tp_adata.obs['pixel_y'] = tp_adata.obsm['spatial'][:, 1]
        # plot the spatial domains
        plt.figure(figsize=(5, 5))
        sns.scatterplot(data=tp_adata.obs, x='pixel_x', y='pixel_y', 
                        hue='kmeans', palette='tab10', s=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Time point: {}'.format(tp))
        plt.tight_layout()
        plt.savefig(args.output_dir+os.sep+args.prefix+str(tp)+'_domains.png')
        plt.close()

def plot_OT(args, adata):
    """
    Plot the optimal transport results.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the spatial data.    
    Returns
    -------
    None
    """
    days = adata.obs['day'].unique()
    days.sort()
    for tp_i in range(len(days)-1):
        prev_day = days[tp_i]
        next_day = days[tp_i+1]
        transition_table = anndata.read_h5ad(args.output_dir+os.sep+'transition_table_'+str(prev_day)+'_'+str(next_day)+'.h5ad')
        # --- normalize by column sum
        transition_prob = transition_table.X
        transition_prob = transition_prob/transition_prob.sum(axis=0, keepdims=True)
        transition_prob_col_norm_df = pd.DataFrame(transition_prob, index=transition_table.obs_names, columns=transition_table.var_names)
        # --- normalize by row sum
        transition_prob = transition_table.X
        transition_prob = transition_prob/transition_prob.sum(axis=1, keepdims=True)
        transition_prob_row_norm_df = pd.DataFrame(transition_prob, index=transition_table.obs_names, columns=transition_table.var_names)
        # --- obtain minimum of both normalized transition probabilities
        transition_prob_min_df = np.minimum(transition_prob_col_norm_df.values, transition_prob_row_norm_df.values)
        # Generate a dotplot for transition_prob_min_df with dynamic figure size
        plt.figure(figsize=(transition_prob_min_df.shape[1]*0.8, transition_prob_min_df.shape[0]*0.8))
        for i in range(transition_prob_min_df.shape[0]):
            for j in range(transition_prob_min_df.shape[1]):
                value = transition_prob_min_df[i, j]
                color = 'grey' if value < 0.2 else plt.cm.Reds(value)
                plt.scatter(j, i, s=value * 500, c=[color], edgecolors='black', alpha=0.8)
        plt.xticks(range(transition_prob_min_df.shape[1]), transition_table.var_names, rotation=45, ha='right')
        plt.yticks(range(transition_prob_min_df.shape[0]), transition_table.obs_names)
        plt.xlabel('{} Domains'.format(next_day))
        plt.ylabel('{} Domains'.format(prev_day))
        plt.title('Transition Probability Dotplot')
        plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), label='Transition Probability')
        plt.tight_layout()
        plt.savefig(args.output_dir+os.sep+f'transition_dotplot_{prev_day}_{next_day}.png')
        plt.close()