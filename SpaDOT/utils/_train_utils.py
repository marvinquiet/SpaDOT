import os
from time import time
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
from torch import optim
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm
from collections import OrderedDict

# --- my package
from SpaDOT.utils import _utils
from SpaDOT.utils.OT_loss.ot_solvers import compute_transport_map # Waddington, faster C version
from SpaDOT.model import SpaDOT

class MyDataset(Dataset):
    def __init__(self, pos, counts, tp_ix, dtype=torch.float64):
        self.dataset = TensorDataset(torch.tensor(pos, dtype=dtype), 
                                     torch.tensor(counts, dtype=dtype), 
                                     torch.tensor(tp_ix, dtype=torch.int))
    def __getitem__(self, index):
        pos, counts, tp_ix  = self.dataset[index]
        return pos, counts, tp_ix
    def __len__(self):
        return len(self.dataset)
    

def prepare_dataloader(adata, model_config):
    # --- obtain location and time point information
    loc = _obtain_tp_loc_info(adata)
    # sample some inducing points
    inducing_idx = random.sample(range(loc.shape[0]), model_config['inducing_point_nums']) 
    inducing_points = loc[inducing_idx, :]
    # --- generate mapper between tp_mat index and real timepoint -> in case the timepoint is not numeric or continous
    idx_to_tp = dict(enumerate(model_config['timepoints']))
    tp_to_idx = {v:k for k,v in idx_to_tp.items()} 
    idx = np.argmax(loc[:, 2:], axis=1)    
    tp_list = [idx_to_tp[_] for _ in idx]
    tp_index_dict = dict() # tp index in timepoint list
    for tp in model_config['timepoints']:
        tp_ix = [_ for _ in range(len(tp_list)) if tp_list[_] == tp]
        tp_index_dict[tp] = tp_ix
    # create inducing point dict / number of training sample dict for separate decoder
    inducing_points_dict, N_train_dict = OrderedDict(), OrderedDict()
    inducing_points_tp = np.argmax(inducing_points[:, 2:], axis=1)
    for _, tp in enumerate(model_config['timepoints']):
        idx = tp_to_idx[tp]
        tp_inducing_idx = np.where(inducing_points_tp == idx)[0]
        inducing_points_dict[tp] = inducing_points[tp_inducing_idx, :2]
        N_train_dict[tp] = np.sum(adata.obs['timepoint'] == tp) # number of training data in each timepoint
    # --- generate timepoint-specific dataloader
    dataloaders_dict, adj_dict, dataset_dict = OrderedDict(), OrderedDict(), OrderedDict()
    for tp in model_config['timepoints']:
        tp_ix = tp_index_dict[tp]
        tp_adata = adata[tp_ix].copy()
        tp_loc = loc[tp_ix, :2]
        tp_dataset = MyDataset(tp_loc, tp_adata.X, tp_ix)
        dataset_dict[tp] = tp_dataset
        # --- Neighborloader for graph data
        _utils._Cal_Spatial_Net(tp_adata, k_cutoff=min(model_config['max_neighbors'], model_config['knn_cutoff']*round(1/1000*tp_adata.n_obs)),
                               max_neigh=model_config['max_neighbors'])
        tp_adj = torch.tensor(tp_adata.uns['adj'], dtype=model_config['dtype'])
        tp_edge_index, _ = dense_to_sparse(tp_adj)
        tp_graph_data = Data(
            x=torch.tensor(tp_adata.X, dtype=model_config['dtype']),
            # edge_index=tp_edge_index.clone().to(dtype=torch.long), # torch copy warning
            edge_index=torch.tensor(tp_edge_index, dtype=torch.long),
            data_index=torch.tensor(tp_ix, dtype=torch.int),
            loc=torch.tensor(tp_loc, dtype=model_config['dtype'])
        )
        tp_graph_loader = NeighborLoader(
            tp_graph_data, num_neighbors=[max(30, model_config['knn_cutoff']*round(1/1000*tp_adata.n_obs))] * 2, 
            batch_size = model_config['batch_size'],
            subgraph_type="induced",
            worker_init_fn=_utils.seed_worker
        )
        dataloaders_dict[tp] = tp_graph_loader 
        adj_dict[tp] = tp_adj
    return {
        "inducing_points": inducing_points_dict,
        "N_train": N_train_dict,
        "dataloaders": dataloaders_dict,
        "adjacency_matrices": adj_dict,
        "datasets": dataset_dict
    }


# --- obtain latent representations
def get_latent(model, model_config, adata, dataloader_dict):
    '''Obtain latent representations
    :param model: SpaDOT model
    :param model_config: model configuration
    :return: latent representations
    '''
    model.eval()
    with torch.no_grad():
        latent_adata_list = list()
        for tp in model_config['timepoints']:
            tp_loc, tp_y, tp_idx = dataloader_dict['datasets'][tp].dataset.tensors
            tp_adj = dataloader_dict['adjacency_matrices'][tp]
            tp_edge_index, _ = dense_to_sparse(tp_adj)
            latent_samples = model.all_latent_samples(tp_loc, tp_y, tp_edge_index, tp)  
            tp_latent_adata = sc.AnnData(latent_samples, obs=adata[tp_idx.numpy()].obs)
            tp_latent_adata.obsm['spatial'] = adata.obsm['spatial'][tp_idx.numpy()]
            latent_adata_list.append(tp_latent_adata)
        latent_adata = anndata.concat(latent_adata_list)
    return latent_adata

def _obtain_tp_loc_info(adata):
    '''
    Obtain the location and time point information from the adata object.
    The location is standardized and concatenated with the time point information.
    The time point information is one-hot encoded.
    '''
    # in case the timepoint is not numeric
    adata.obs['timepoint_numeric'] = adata.obs['timepoint'].astype('category').cat.codes
    tp_info = np.array(adata.obs['timepoint_numeric']).astype('int')
    tp_mat = np.zeros((tp_info.size, tp_info.max()+1))
    tp_mat[np.arange(tp_info.size), tp_info] = 1
    n_tp = tp_mat.shape[1]
    # scale locations per time point
    loc = adata.obsm['spatial']
    loc_scaled = np.zeros(loc.shape, dtype=np.float64)
    for i in range(n_tp):
        scaler = StandardScaler()
        tp_loc = loc[tp_mat[:,i]==1, :]
        tp_loc = scaler.fit_transform(tp_loc)
        loc_scaled[tp_mat[:,i]==1, :] = tp_loc
    loc = loc_scaled
    loc = np.concatenate((loc, tp_mat), axis=1)
    return loc


def _beta_cycle_linear(n_iter, start=0.0, stop=1, n_cycle=10, ratio=1):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def train_SpaDOT(dataloader_dict, model_config):
    SpaDOT_model = SpaDOT.SpaDOT(model_config, dataloader_dict).to(model_config['device'])
    print(SpaDOT_model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, SpaDOT_model.parameters()), lr=model_config['lr'])

    # --- prepare training
    beta1s = _beta_cycle_linear(model_config['maxiter'], stop=model_config['beta1'])
    tp_indexed_list = list(enumerate(model_config['timepoints']))
    
    # --- initialize dictionary to store loss
    loss_dict = OrderedDict()
    loss_names = ['elbo', 'Recon', 'SVGP_KL', 'GAT_KL', 'alignment', 'KMeans', 'OT']
    for epoch in range(model_config['maxiter']):
        loss_dict[epoch] = OrderedDict()
        for name in loss_names:
            loss_dict[epoch][name] = 0

    print("Training SpaDOT model...")
    train_starttime = time()
    for epoch in tqdm(range(model_config['maxiter'])):
        beta1 = beta1s[epoch]
        # --- start training
        SpaDOT_model.train()
        ep_starttime = time()
        tp_loss_dict = OrderedDict()
        # --- random shuffle timepoints
        random.shuffle(tp_indexed_list) 
        for tp_i, tp in tp_indexed_list:
            tp_loss_dict[tp] = OrderedDict()
            for name in loss_names:
                tp_loss_dict[tp][name] = 0
            tp_dataloader = dataloader_dict['dataloaders'][tp]
            for _, batch in enumerate(tp_dataloader):
                # y: gene expression; x: location; tp_ix: timepoint index; edge_index: graph edge index
                y_batch, x_batch, tp_ix, edge_index_batch = batch.x, batch.loc, batch.data_index, batch.edge_index
                x_batch, y_batch, edge_index_batch = x_batch.to(model_config['device']), y_batch.to(model_config['device']), edge_index_batch.to(model_config['device'])
                tp_ix, adj_idx,  = tp_ix[:batch.batch_size], batch.n_id[:batch.batch_size] # subset to seed nodes only
                # --- forward SVGPVAE and GATVAE to obtain latent space
                tp_Recon_val, tp_SVGP_KL_val, tp_GAT_KL_val, tp_alignment_val, tp_p_m = \
                SpaDOT_model.forward(x=x_batch, y=y_batch, edge_index=edge_index_batch, 
                                     tp=tp, batch_size=batch.batch_size)
                # --- calculate KMeans loss from latent space
                tp_KMeans_val = torch.tensor(0, dtype=model_config['dtype'], device=model_config['device'])
                if epoch >= 1: # add Kmeans loss after the first epoch once the latent space is generated
                    kmeans_loss = _compute_kmeans_loss(SpaDOT_model, model_config, tp, tp_ix, tp_p_m)
                    tp_KMeans_val += kmeans_loss
                # --- calculate OT loss from latent space
                tp_OT_val = torch.tensor(0, dtype=model_config['dtype'], device=model_config['device'])
                if epoch >= model_config['ot_epoch']:
                    if tp_i != 0:
                        tp_loss = _compute_OT_loss(SpaDOT_model, model_config, tp, tp_ix, tp_p_m, model_config['timepoints'][tp_i-1])
                        tp_OT_val += tp_loss
                tp_elbo_val = model_config['lambda1'] * tp_Recon_val - \
                    beta1 * tp_SVGP_KL_val + \
                    model_config['beta2'] * tp_GAT_KL_val + \
                    model_config['omiga1'] * tp_alignment_val + \
                    model_config['omiga2'] * tp_KMeans_val + \
                    model_config['omiga3'] * tp_OT_val
                # --- backward
                optimizer.zero_grad()
                tp_elbo_val.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(SpaDOT_model.parameters(), 0.3) # stabilize training
                optimizer.step()
                # --- update loss in each timepoint
                for name in loss_names:
                    tp_loss_dict[tp][name] += locals()[f'tp_{name}_val'].detach().cpu().item()
            # --- update loss in loss dictionary
            for name in loss_names:
                tp_loss_dict[tp][name] /= len(tp_dataloader)
                loss_dict[epoch][f'{name}'] += tp_loss_dict[tp][name]
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}: Training time: {int(time()-ep_starttime)} seconds, ELBO: {loss_dict[epoch]["elbo"]:.8f}, Recon loss: {loss_dict[epoch]["Recon"]:.8f}, SVGP KL loss: {loss_dict[epoch]["SVGP_KL"]:.8f}, GAT KL loss: {loss_dict[epoch]["GAT_KL"]:.8f}, Alignment loss: {loss_dict[epoch]["alignment"]:.8f}, Kmeans loss: {loss_dict[epoch]["KMeans"]:.8f}, OT loss: {loss_dict[epoch]["OT"]:.8f}')
        # --- update Kmeans centroids
        _update_Kmeans(SpaDOT_model, model_config, dataloader_dict)
        # --- update OT matrix 
        if (epoch + 1) % model_config['ot_config']['ot_epochs'] == 0:
            _update_OT_matrix(SpaDOT_model, model_config)

    print('Training finished...')
    print('Training time: %d seconds.' % int(time() - train_starttime))
    loss_df = pd.DataFrame.from_dict(loss_dict)
    return SpaDOT_model, loss_df


# K-Means related
def _compute_kmeans_loss(model, model_config, tp, tp_ix, latent):
    '''Compute KMeans loss between assignment and centroids
    :param tp: time point
    :param tp_ix: time point specific index
    :param latent: GP posterior latent space

    :return: K-Means loss
    '''
    cluster_index_dict = model.kmeans_index_dict
    cluster_assignments = [cluster_index_dict[tp][_.item()] for _ in tp_ix]
    cluster_centers = torch.tensor(model.kmeans_center_dict[tp], dtype=model_config['dtype'], device=model_config['device'])
    # Compute K-means loss (mean of squared distances)
    kmeans_loss = torch.sum(torch.norm(latent - cluster_centers[cluster_assignments]) ** 2 / latent.shape[1] / len(set(cluster_assignments)) )
    return kmeans_loss

def _update_Kmeans(model, model_config, dataloader_dict):
    model.eval()
    with torch.no_grad():
        for tp in model_config['timepoints']:
            tp_loc, tp_y, tp_idx = dataloader_dict['datasets'][tp].dataset.tensors
            tp_adj = dataloader_dict['adjacency_matrices'][tp]
            tp_edge_index, _ = dense_to_sparse(tp_adj)
            latent_samples = model.all_latent_samples(tp_loc, tp_y, tp_edge_index, tp)  
            # adaptive KMeans
            tp_kmeans = KMeans(n_clusters=model_config['n_clusters'], 
                               random_state=model_config['seed'], 
                               n_init=10).fit(latent_samples)
            model.kmeans_center_dict[tp] = tp_kmeans.cluster_centers_
            model.kmeans_cluster_dict[tp] = tp_kmeans.labels_.tolist()
            model.kmeans_index_dict[tp] = dict(zip(tp_idx.numpy(), tp_kmeans.labels_))

# OT related
def _compute_OT_loss(model, model_config, cur_tp, tp_ix, tp_p_m, prev_tp):
    '''Compute OT loss between timepoints
    :param model:  model
    :param tp: current timepoints
    :param tp_ix: index of current tp to obtain hidden clusters
    :param tp_p_m: posterior mean of current timepoints
    :prev_tp: previous timepoint
    :return: OT transport cost
    '''
    cur_tp_cluster_latent = dict()
    cur_tp_clusters = [model.kmeans_index_dict[cur_tp][_] for _ in tp_ix.numpy()]
    cluster_list = list(set(model.kmeans_cluster_dict[cur_tp]))
    cluster_list.sort()
    for idx, cluster in enumerate(cur_tp_clusters):
        if cluster in cur_tp_cluster_latent:
            cur_tp_cluster_latent[cluster].append(tp_p_m[idx])
        else:
            cur_tp_cluster_latent[cluster] = [tp_p_m[idx]]
    cur_tp_cluster_center = []
    for cluster in cluster_list:
        if cluster not in cur_tp_cluster_latent:
            cur_tp_cluster_center.append(torch.tensor(model.kmeans_center_dict[cur_tp][cluster], 
                                                      dtype=model_config['dtype'], device=model_config['device']))
        else:
            cur_tp_cluster_center.append(torch.mean(torch.stack(cur_tp_cluster_latent[cluster]), dim=0))
    cur_tp_cluster_center = torch.stack(cur_tp_cluster_center)
    gamma = model.gammas[f"{prev_tp}_{cur_tp}"] # obtain transport plan
    gamma = gamma / gamma.sum(axis=1, keepdims=True) # normalize rows
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)  # prune invalid values
    gamma = torch.tensor(gamma, dtype=model_config['dtype']).to(model_config['device'])
    cost_matrix = torch.cdist(
        torch.tensor(model.kmeans_center_dict[prev_tp], dtype=model_config['dtype']).to(model_config['device']), 
        cur_tp_cluster_center, 
        p=2) # (prev_tp_cluster, cur_tp_cluster)
    transport_cost = torch.mean(gamma*cost_matrix)
    return transport_cost

def _update_OT_matrix(model, model_config):
    '''Compute OT matrix
    '''
    model.eval()
    timepoints = model_config['timepoints']
    for tp_i, tp in enumerate(timepoints):
        if tp_i == len(timepoints) - 1: break
        cur_tp = tp
        next_tp = timepoints[tp_i+1]
        gamma = compute_transport_map(model.kmeans_center_dict[cur_tp],
                                       model.kmeans_center_dict[next_tp], 
                                       model_config['ot_config'], G=None)
        model.gammas[f"{timepoints[tp_i]}_{timepoints[tp_i+1]}"] = gamma