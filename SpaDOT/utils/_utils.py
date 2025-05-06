# --- other utilitities functions ---
import os
import yaml
# from importlib.resources import files
import sklearn
import scipy as sp
import numpy as np
import pandas as pd
import torch
import random
from torch.backends import cudnn

def set_seed(seed=1993):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed) # for stabilization

def seed_worker(worker_id):
    np.random.seed(1993)
    random.seed(1993)

def load_model_config(args):
    '''
    Load the model configuration from the config.yaml file provided in the package.
    '''
    if args.config:
        config_path = args.config
    else:
        # Load the default config.yaml from the package
        # config_path = files("SpaDOT").joinpath("config.yaml")
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _Cal_Spatial_Net(adata, k_cutoff=None, max_neigh=30):
    """
    Construct the spatial neighbor networks using KNN.
    Parameters

    adata : anndata.AnnData
        The AnnData object containing the spatial data.
    """

    print('Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='auto').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    indices = indices[:, 1:k_cutoff + 1]
    distances = distances[:, 1:k_cutoff + 1]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
    print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    X = pd.DataFrame(adata.layers['counts'].toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    # add self-loop
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + np.eye(G.shape[0])  # add self-loop
    adata.uns['adj'] = G

def _save_inducing_points(args, inducing_points_dict):
    """
    Write the inducing points to a file.
    """
    # Convert the dictionary into a DataFrame directly
    inducing_points_list = []
    for key, value in inducing_points_dict.items():
        # Create a DataFrame for each key-value pair
        df = pd.DataFrame(value)
        df.columns = ['norm-pixel_x', 'norm-pixel_y']
        df["timepoint"] = key  # Add the key as a new column
        inducing_points_list.append(df)
    # Combine all DataFrames into a single DataFrame
    inducing_points_df = pd.concat(inducing_points_list, ignore_index=True)

    # Save to a file
    inducing_points_df.to_csv(args.output_dir+os.sep+args.prefix+"inducing_points.csv", index=False)