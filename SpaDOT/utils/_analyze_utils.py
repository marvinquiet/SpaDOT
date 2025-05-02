import os
import anndata
import scipy as sp
import sklearn.neighbors
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler

def obtain_loc_tp_info(adata):
    '''
    Obtain the location and time point information from the adata object.
    The location is standardized and concatenated with the time point information.
    The time point information is one-hot encoded.
    '''
    adata.obs['timepoint_numeric'] = adata.obs['timepoint'].astype('category').cat.codes
    time_info = np.array(adata.obs['timepoint_numeric']).astype('int')
    time_mat = np.zeros((time_info.size, time_info.max()+1))
    time_mat[np.arange(time_info.size), time_info] = 1
    n_tp = time_mat.shape[1]
    # scale locations per time point
    loc = adata.obsm['spatial']
    loc_scaled = np.zeros(loc.shape, dtype=np.float64)
    for i in range(n_tp):
        scaler = StandardScaler()
        tp_loc = loc[time_mat[:,i]==1, :]
        tp_loc = scaler.fit_transform(tp_loc)
        loc_scaled[time_mat[:,i]==1, :] = tp_loc
    loc = loc_scaled
    loc = np.concatenate((loc, time_mat), axis=1)
    return loc, time_mat, n_tp



# 
def preprocess_adata(args, adata, get_SVG=False):
    if args.feature_selection:
        SVGs = select_SVGs(args, adata)


    SVG_genes = select_SVGs(args, adata, get_SVG=get_SVG)
    adata = adata[:, SVG_genes].copy()
    # filter genes
    if not scipy.sparse.issparse(adata.X):
        adata.layers['counts'] = scipy.sparse.csr_matrix(adata.X)
    else:
        adata.layers['counts'] = adata.X
    # --- preprocess data
    tp_adata_list = []
    for tp in args.timepoints:
        tp_adata = adata[adata.obs['timepoint'] == tp]
        sc.pp.normalize_total(tp_adata, target_sum=1e-4)
        sc.pp.log1p(tp_adata)
        tp_adata_list.append(tp_adata)
    combined_genes = set().union(*(obj.var_names for obj in tp_adata_list))
    combined_genes = list(combined_genes)
    combined_genes.sort()
    with open(args.result_dir+os.sep+'selected_genes.txt', 'w') as f:
        for item in combined_genes:
            f.write("%s\n" % item)  
    new_tp_adata_list = []
    for tp_adata in tp_adata_list:
        tp_adata = tp_adata[:, combined_genes]
        sc.pp.scale(tp_adata)
        new_tp_adata_list.append(tp_adata)
    concat_adata = anndata.concat(new_tp_adata_list)
    return concat_adata






def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=100, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    # coor = pd.DataFrame(adata.obs[['pixel_x', 'pixel_y']])
    assert 'spatial' in adata.obsm.keys()
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        # n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
        n_neighbors=max_neigh + 1, algorithm='auto').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.layers['counts'].toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G


def Stats_Spatial_Net(adata, result_dir):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.savefig(result_dir+os.sep+'spatial_net_stats.png')
    plt.close()


def rbf_adjacency_matrix(features, sigma=1.0, sparsity=0.1):
    """
    Builds an adjacency matrix using the RBF kernel.

    Args:
        features (np.ndarray): Node features of shape (N, F) where N is the number of nodes.
        sigma (float): Bandwidth parameter for the RBF kernel.
        threshold (float): Minimum weight to consider an edge (optional).

    Returns:
        np.ndarray: Adjacency matrix of shape (N, N).
    """
    assert sparsity >= 0 and sparsity <= 1
    # Compute squared Euclidean distance
    distances = np.sum(features**2, axis=1)[:, None] + np.sum(features**2, axis=1) - 2 * np.dot(features, features.T)
    # Apply the RBF kernel
    adjacency_matrix = np.exp(-distances / (2 * sigma**2))
    # Optional: Apply threshold to sparsify the graph
    threshold = np.percentile(adjacency_matrix, 100*(1-sparsity))
    adjacency_matrix[adjacency_matrix < threshold] = 0
    return adjacency_matrix