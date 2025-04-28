import os
import argparse
import anndata
from time import time
import random
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
sns.set_palette(cc.glasbey)

import torch
from torch import optim
import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

# --- load my package
import SpaDOT.data_processer as data_processer
from OT_loss.ot_solvers import compute_transport_map # Waddington

class MyDataset(Dataset):
    def __init__(self, pos, counts, tp_ix):
        self.dataset = TensorDataset(torch.tensor(pos, dtype=torch.float64), torch.tensor(counts, dtype=torch.float64), torch.tensor(tp_ix, dtype=torch.int))
    def __getitem__(self, index):
        pos, counts, tp_ix  = self.dataset[index]
        return pos, counts, tp_ix
    def __len__(self):
        return len(self.dataset)


def parse_spaDOT_args():
    parser = argparse.ArgumentParser(description='SVGPVAE+MSE+OT pipeline')
    parser.add_argument('--maxiter', type=int, default=100, help='number of epochs trained')
    parser.add_argument('--batch_size', type=int, default=512, help='size of each batch')
    parser.add_argument('--z_dim', type=int, default=20, help='latent dimension')
    parser.add_argument('--beta', type=float, default=1, help='control weight on KL loss in VAE: beta / z_dim')
    parser.add_argument('--kernel_type', type=str, default='Gaussian', help='type of Kernel')
    parser.add_argument('--ot_epoch', type=int, default=50, help='when to start adding OT loss')
    parser.add_argument('--kmeans_epoch', type=int, default=1, help='when to start adding KMeans loss')
    parser.add_argument('--inducing_point_nums', type=int, default=1200, help='number of inducing points')
    args, _ = parser.parse_known_args()
    args.n_clusters = 10 # use a large K for KMeans
    args.gradient_clipping = 0.5 # slow down the training
    # default settings for the network
    args.lr = 3e-4
    args.encoder_layers = [256, 64] # more parameters
    args.decoder_layers = [64, 256]
    # sample inducing point
    args.fix_inducing_points = True
    args.fixed_gp_params = True
    args.model_file = 'model.pt'
    args.final_latent_file = 'final_latent.txt'
    # configuration for OT
    ot_config = {
        "growth_iters": 3,
        "ot_epochs": 10,
        "epsilon": 0.05, # larger epsilon can result in a blurred transport plan
        "lambda1": 0.1, # row imbalance
        "lambda2": 5, # column imbalance
        "epsilon0": 1,
        "tau": 1000,
        "scaling_iter": 3000,
        "inner_iter_max": 50,
        "tolerance": 1e-8,
        "max_iter": 1e7,
        "batch_size": 5,
        "extra_iter": 1000,
        "numItermax": 1000000,
        "use_Py": False,
        "use_C": True,
        "profiling": False,
        "method": "waddington" # Default
    }  # specified in tutorial
    args.ot_config = ot_config
    return args

def beta_cycle_linear(n_iter, start=0.0, stop=1, n_cycle=10, ratio=1):
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

def prepare_dataloader(adata, args, k_cutoff=6):
    # obtain loc, batch_mat
    loc, batch_mat = data_processer.obtain_loc_batch_info(adata)
    # sample some inducing points
    sampled_inducing_idx = random.sample(range(loc.shape[0]), args.inducing_point_nums)   # confirmed seed works! always the same inducing points
    initial_inducing_points = loc[sampled_inducing_idx, :]
    np.savetxt(args.result_dir+os.sep+"sampled_inducing_points.txt", initial_inducing_points, delimiter=",")
    # batch <-> timepoint mapper
    batch_to_tp_dict = dict(zip(np.argmax(batch_mat, axis=1), adata.obs['timepoint'].values))
    tp_to_batch_dict = dict(zip(adata.obs['timepoint'].values, np.argmax(batch_mat, axis=1)))
    batch_idx = np.argmax(batch_mat, axis=1)    
    tps_list = [batch_to_tp_dict[idx] for idx in batch_idx]
    tp_index_dict = dict() # tp index in timepoint list
    for tp in args.timepoints:
        tp_ix = [_ for _ in range(len(tps_list)) if tps_list[_] == tp]
        tp_index_dict[tp] = tp_ix
    # create inducing point dict / number of training sample dict for separate decoder
    initial_inducing_points_dict, N_train_dict, kernel_scale_dict = dict(), dict(), dict()
    initial_inducing_points_batch = np.argmax(initial_inducing_points[:, 2:], axis=1)
    for _, tp in enumerate(args.timepoints):
        batch_id = tp_to_batch_dict[tp]
        tp_inducing_idx = np.where(initial_inducing_points_batch == batch_id)[0]
        initial_inducing_points_dict[tp] = initial_inducing_points[tp_inducing_idx, :2]
        N_train_dict[tp] = np.sum(adata.obs['timepoint'] == tp) # number of training data in each timepoint
        kernel_scale_dict[tp] = float(args.kernel_scale[_])
    # --- generate timepoint-specific dataloader
    dataloaders_dict, adj_dict, dataset_dict = dict(), dict(), dict()
    for tp in args.timepoints:
        tp_ix = tp_index_dict[tp]
        tp_adata = adata[tp_ix].copy()
        tp_loc = loc[tp_ix, :2]
        tp_dataset = MyDataset(tp_loc, tp_adata.X, tp_ix)
        dataset_dict[tp] = tp_dataset
        # --- Neighborloader for graph data
        data_processer.Cal_Spatial_Net(tp_adata, k_cutoff=max(30, k_cutoff*round(1/1000*tp_adata.n_obs)), model='KNN')
        tp_adj = torch.tensor(tp_adata.uns['adj'], dtype=torch.double)
        tp_edge_index, _ = dense_to_sparse(tp_adj)
        tp_graph_data = Data(
            x=torch.tensor(tp_adata.X, dtype=torch.double),
            edge_index=torch.tensor(tp_edge_index, dtype=torch.long),
            data_index=torch.tensor(tp_ix, dtype=torch.int),
            loc=torch.tensor(tp_loc, dtype=torch.double)
        )
        tp_graph_loader = NeighborLoader(
            tp_graph_data, num_neighbors=[max(30, k_cutoff*round(1/1000*tp_adata.n_obs))] * 2, batch_size=args.batch_size,
            subgraph_type="induced"
        )
        dataloaders_dict[tp] = tp_graph_loader 
        adj_dict[tp] = tp_adj
    return initial_inducing_points_dict, kernel_scale_dict, N_train_dict, dataloaders_dict, adj_dict, dataset_dict


def train_spaDOT(model, args, logging, adata):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # --- prepare training
    minimum_dataloader_length = min([len(model.dataloaders_dict[tp]) for tp in args.timepoints])
    logging.info('Minimum dataloader length among timepoints: %d' % minimum_dataloader_length)
    beta1s = beta_cycle_linear(args.maxiter, stop=args.beta1)
    # beta2s = beta_cycle_linear(args.maxiter, stop=args.beta2)
    tp_indexed_list = list(enumerate(args.timepoints))
    
    # --- initialize dictionary to store loss
    loss_dict = dict()
    loss_names = ['elbo', 'SVGP_recon', 'SVGP_KL', 'GAT_recon', 'GAT_KL', 'alignment', 'KMeans', 'OT']
    for epoch in range(args.maxiter):
        loss_dict[epoch] = dict()
        for name in loss_names:
            loss_dict[epoch][name] = 0

    logging.info("Start training...")
    train_starttime = time()
    for epoch in tqdm(range(args.maxiter)):
        logging.info('--- Epoch %d' % (epoch+1))
        model.beta1 = beta1s[epoch]
        model.beta2 = args.beta2
        logging.info('Model beta1: %.6f, Model beta2: %.6f' % (model.beta1, model.beta2))
        # --- start training
        model.train()
        ep_starttime = time()
        tp_loss_dict = dict()
        # --- random shuffle timepoints
        random.shuffle(tp_indexed_list)   # confirm same order if set seed
        for tp_i, tp in tp_indexed_list:
            tp_loss_dict[tp] = dict()
            for name in loss_names:
                tp_loss_dict[tp][name] = 0
            tp_dataloader = model.dataloaders_dict[tp]
            tp_adj = model.adj_dict[tp]
            for batch_idx, batch in enumerate(tp_dataloader):
                # if batch_idx == minimum_dataloader_length:
                #     break
                # y: gene expression; x: location; tp_ix: timepoint index; edge_index: graph edge index
                y_batch, x_batch, tp_ix, edge_index_batch = batch.x, batch.loc, batch.data_index, batch.edge_index
                x_batch, y_batch, edge_index_batch = x_batch.to(model.device), y_batch.to(model.device), edge_index_batch.to(model.device)
                tp_ix, adj_idx,  = tp_ix[:batch.batch_size], batch.n_id[:batch.batch_size] # subset to seed nodes only
                adj_batch = tp_adj[adj_idx, :][:, adj_idx]
                adj_batch = adj_batch.to(model.device)
                # --- forward SVGPVAE and GATVAE to obtain latent space
                tp_SVGP_recon_val, tp_SVGP_KL_val, tp_GAT_KL_val, tp_alignment_val, tp_p_m, tp_SVGP_p_m, tp_GAT_p_m  = \
                model.forward(x=x_batch, y=y_batch, edge_index=edge_index_batch, tp=tp, batch_size=batch.batch_size)
                # --- calculate reconstructed graph from tp_p_m
                tp_recon_graph = torch.sigmoid(torch.mm(tp_GAT_p_m, tp_GAT_p_m.t()))
                tp_GAT_recon_val = F.binary_cross_entropy(tp_recon_graph, adj_batch, reduction='sum') / batch.batch_size
                # --- calculate KMeans loss from latent space
                tp_KMeans_val = torch.tensor(0, dtype=torch.float64, device=model.device)
                if args.kmeans_epoch is not None and epoch >= args.kmeans_epoch:
                    kmeans_loss = compute_kmeans_loss(model, tp, tp_ix, tp_p_m)
                    tp_KMeans_val += kmeans_loss
                # --- calculate OT loss from latent space
                tp_OT_val = torch.tensor(0, dtype=torch.float64, device=model.device)
                if args.ot_epoch is not None and epoch >= args.ot_epoch:
                    if tp_i != 0:
                        tp_loss = compute_OT_loss(model, tp, tp_ix, tp_p_m, args.timepoints[tp_i-1])
                        tp_OT_val += tp_loss
                tp_elbo_val = args.lambda1 * tp_SVGP_recon_val - model.beta1 * tp_SVGP_KL_val
                tp_elbo_val += args.lambda2 * tp_GAT_recon_val + model.beta2 * tp_GAT_KL_val
                tp_elbo_val += args.omiga1 * tp_alignment_val + args.omiga2 * tp_KMeans_val + args.omiga3 * tp_OT_val
                # --- backward
                optimizer.zero_grad()
                tp_elbo_val.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                optimizer.step()
                # --- update loss in each timepoint
                for name in loss_names:
                    tp_loss_dict[tp][name] += locals()[f'tp_{name}_val'].detach().cpu().item()
            # scheduler.step()
            # --- update loss in loss dictionary
            for name in loss_names:
                tp_loss_dict[tp][name] /= len(tp_dataloader)
                loss_dict[epoch][f'{name}'] += tp_loss_dict[tp][name]
        logging.info('Training time:{:.2f}, ELBO:{:.8f}, SVGP Recon loss:{:.8f}, SVGP KLD loss:{:.8f}, GAT Recon loss:{:.8f}, GAT KLD loss:{:.8f}, \
                    Alignment loss: {:.8f}, Kmeans loss: {:.8f}, OT loss: {:.8f}'.format(
                    int(time()-ep_starttime), loss_dict[epoch]['elbo'], loss_dict[epoch]['SVGP_recon'], loss_dict[epoch]['SVGP_KL'],
                    loss_dict[epoch]['GAT_recon'], loss_dict[epoch]['GAT_KL'], 
                    loss_dict[epoch]['alignment'], loss_dict[epoch]['KMeans'], loss_dict[epoch]['OT']))
        for tp in args.timepoints:
            logging.info('Timepoint {}, ELBO:{:.8f}, SVGP Recon loss:{:.8f}, SVGP KLD loss:{:.8f}, GAT Recon loss:{:.8f}, GAT KLD loss:{:.8f}, \
                    Alignment loss: {:.8f}, Kmeans loss: {:.8f}, OT loss: {:.8f}'.format(
                    tp, tp_loss_dict[tp]['elbo'], tp_loss_dict[tp]['SVGP_recon'], tp_loss_dict[tp]['SVGP_KL'], 
                    tp_loss_dict[tp]['GAT_recon'], tp_loss_dict[tp]['GAT_KL'], 
                    tp_loss_dict[tp]['alignment'], tp_loss_dict[tp]['KMeans'], tp_loss_dict[tp]['OT']))

        # --- update Kmeans
        update_Kmeans(model, args)
        # --- Update OT matrix
        if (epoch + 1) % args.ot_config["ot_epochs"] == 0:
            update_OT_matrix(model, args)
        # --- evaluate
        if (epoch + 1) % 10 == 0 and args.latent_record:
            _ = do_eval(model, args, logging, adata,
                prefix=str(epoch)+'_', plot=True) # for visualization purpose
    #     # # --- early stopping
    #     if epoch > args.ot_epoch and epoch > args.kmeans_epoch:
    #         # train_loss = recon_loss_val + adj_recon_val
    #         train_loss = elbo_val
    #         if train_loss < model.best_loss - model.min_delta:
    #             # model.best_loss = elbo_val 
    #             model.best_loss = train_loss # or tracking recon_loss_val+adj_recon_val
    #             no_improve_epochs = 0
    #             torch.save(model.state_dict(), args.result_dir+os.sep+"best_model.pth")
    #         else:
    #             no_improve_epochs += 1
    #             if no_improve_epochs >= model.early_stopping_patience:
    #                 print("Early stopping triggered")
    #                 break
    # model.load_state_dict(torch.load(args.result_dir+os.sep+"best_model.pth"))

    logging.info('Training time: %d seconds.' % int(time() - train_starttime))
    loss_df = pd.DataFrame.from_dict(loss_dict)
    loss_df.to_csv(args.result_dir+os.sep+'loss.csv')
    return model

def test_spaDOT(args, latent_adata):
    ARI_dict = dict()
    ARI_list = list()
    for tp in args.timepoints:
        tp_adata = latent_adata[latent_adata.obs['timepoint'] == tp].copy()
        tp_emb = tp_adata.X
        tp_kmeans = KMeans(n_clusters=len(set(tp_adata.obs['annotation'])), random_state=0).fit(tp_emb)
        tp_adata.obs['kmeans'] = tp_kmeans.labels_.astype(str)
        tp_ARI = adjusted_rand_score(tp_adata.obs['annotation'], tp_adata.obs['kmeans'])
        print('%s Kmeans ARI: %.3f' % (tp, tp_ARI))
        ARI_dict[tp] = tp_ARI
        ARI_list.append(tp_ARI)
        plt.figure(figsize=(5,5))
        ax = sns.scatterplot(data=tp_adata.obs, x='pixel_x', y='pixel_y', hue='kmeans', s=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(args.result_dir+os.sep+str(tp)+'_Kmeans_separate_clustering.png')
        plt.close()
    pd.DataFrame(ARI_dict, index=[0]).to_csv(args.result_dir+os.sep+'spaDOT_ARI.csv', index=False)
    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=["timepoint", "annotation"])
    plt.savefig(args.result_dir+os.sep+"spaDOT_umap.png")
    plt.close()
    return np.mean(ARI_list)

def update_Kmeans(model, args):
    model.eval()
    with torch.no_grad():
        for tp in args.timepoints:
            tp_loc, tp_y, tp_idx = model.dataset_dict[tp].dataset.tensors
            tp_adj = model.adj_dict[tp]
            tp_edge_index, _ = dense_to_sparse(tp_adj)
            latent_samples = model.all_latent_samples(tp_loc, tp_y, tp_edge_index, tp)  
            # adaptive KMeans
            tp_kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init=10).fit(latent_samples)
            model.kmeans_center_dict[tp] = tp_kmeans.cluster_centers_
            model.kmeans_cluster_dict[tp] = tp_kmeans.labels_.tolist()
            model.kmeans_index_dict[tp] = dict(zip(tp_idx.numpy(), tp_kmeans.labels_))


def do_eval(model, args, logging, adata,
            prefix='', plot=False):
    model.eval()
    with torch.no_grad():
        latent_adata_list = list()
        for tp in args.timepoints:
            tp_loc, tp_y, tp_idx = model.dataset_dict[tp].dataset.tensors
            tp_adj = model.adj_dict[tp]
            tp_edge_index, _ = dense_to_sparse(tp_adj)
            latent_samples = model.all_latent_samples(tp_loc, tp_y, tp_edge_index, tp)  
            tp_latent_adata = sc.AnnData(latent_samples, obs=adata[tp_idx.numpy()].copy().obs)
            tp_latent_adata.obs['annotation'] = tp_latent_adata.obs['annotation'].astype('category')
            # adaptive KMeans
            tp_kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(tp_latent_adata.X)
            model.kmeans_center_dict[tp] = tp_kmeans.cluster_centers_
            model.kmeans_cluster_dict[tp] = tp_kmeans.labels_.tolist()
            model.kmeans_index_dict[tp] = dict(zip(tp_idx.numpy(), tp_kmeans.labels_))

            # --- evaluate based on annotation information
            tp_kmeans = KMeans(n_clusters=len(set(tp_latent_adata.obs['annotation'])), random_state=0).fit(tp_latent_adata.X)
            tp_latent_adata.obs['kmeans'] = tp_kmeans.labels_.astype(str)
            if plot:
                logging.info('Timepoint %s Seprate KMeans ARI: %.3f' % (tp, adjusted_rand_score(tp_latent_adata.obs['annotation'], tp_latent_adata.obs['kmeans'])))
                plt.figure(figsize=(4, 4))
                ax = sns.scatterplot(data=tp_latent_adata.obs, x='pixel_x', y='pixel_y', hue='kmeans', s=10)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(args.latent_records_dir+os.sep+prefix+str(tp)+'_Kmeans_separate_clustering.png')
                plt.close()
            latent_adata_list.append(tp_latent_adata)
            latent_adata = anndata.concat(latent_adata_list)
    return latent_adata


# # --- compute KMeans loss
def compute_kmeans_loss(model, tp, tp_ix, latent):
    '''Compute KMeans loss between assignment and centroids
    :param tp: time point
    :param tp_ix: time point specific index
    :param latent: GP posterior latent space

    :return: K-Means loss
    '''
    cluster_index_dict = model.kmeans_index_dict
    cluster_assignments = [cluster_index_dict[tp][_.item()] for _ in tp_ix]
    cluster_centers = torch.tensor(model.kmeans_center_dict[tp], device=model.device)
    # Compute K-means loss (mean of squared distances)
    kmeans_loss = torch.sum(torch.norm(latent - cluster_centers[cluster_assignments]) ** 2 / latent.shape[1] / len(set(cluster_assignments)) )
    return kmeans_loss


# --- compute OT loss
def compute_OT_loss(model, cur_tp, tp_ix, tp_p_m, prev_tp):
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
            cur_tp_cluster_center.append(torch.tensor(model.kmeans_center_dict[cur_tp][cluster], dtype=model.dtype, device=model.device))
        else:
            cur_tp_cluster_center.append(torch.mean(torch.stack(cur_tp_cluster_latent[cluster]), dim=0))
    cur_tp_cluster_center = torch.stack(cur_tp_cluster_center)
    gamma = model.gammas[f"{prev_tp}_{cur_tp}"] # obtain transport plan
    gamma = gamma / gamma.sum(axis=1, keepdims=True) # normalize rows
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)  # prune invalid values
    gamma = torch.tensor(gamma, dtype=model.dtype).to(model.device)
    cost_matrix = torch.cdist(
        torch.tensor(model.kmeans_center_dict[prev_tp], dtype=model.dtype).to(model.device), 
        cur_tp_cluster_center, 
        p=2) # (prev_tp_cluster, cur_tp_cluster)
    transport_cost = torch.mean(gamma*cost_matrix)
    return transport_cost

def update_OT_matrix(model, args):
    '''Compute OT matrix
    '''
    model.eval()
    timepoints = args.timepoints
    for tp_i, tp in enumerate(timepoints):
        if tp_i == len(timepoints) - 1: break
        cur_tp = tp
        next_tp = timepoints[tp_i+1]
        gamma = compute_transport_map(model.kmeans_center_dict[cur_tp], model.kmeans_center_dict[next_tp], args.ot_config, G=None)
        model.gammas[f"{timepoints[tp_i]}_{timepoints[tp_i+1]}"] = gamma

def compute_OT_heatmap(args, latent_adata, prefix=''):
    '''Compute OT heatmap
    '''
    timepoints = list(set(latent_adata.obs['timepoint']))
    timepoints.sort()
    tp_kmeans_dict = dict()
    for tp in timepoints:
        tp_adata = latent_adata[latent_adata.obs['timepoint']==tp].copy()
        tp_kmeans = KMeans(n_clusters=len(set(tp_adata.obs['annotation'])), random_state=0, n_init=10).fit(tp_adata.X)
        tp_kmeans_dict[tp] = tp_kmeans
    for tp_i in range(len(timepoints)-1):
        cur_tp = timepoints[tp_i]
        next_tp = timepoints[tp_i+1]
        # regularization on C
        C = pairwise_distances(tp_kmeans_dict[cur_tp].cluster_centers_, tp_kmeans_dict[next_tp].cluster_centers_, metric="sqeuclidean", n_jobs=-1)
        C = C**2
        C = C / np.median(C)
        gamma = compute_transport_map(tp_kmeans_dict[cur_tp].cluster_centers_, tp_kmeans_dict[next_tp].cluster_centers_, args.ot_config, C=C, G=None)  ## do not set growth rate for now
        ## write OT matrix
        np.save(prefix+'gamma_tp{}_tp{}_final_OT.npy'.format(cur_tp, next_tp), gamma)
        plt.figure()
        ax = sns.heatmap(gamma, linewidth=0.5, annot=True)
        plt.savefig(prefix+'gamma_tp{}_tp{}_final_heatmap.png'.format(cur_tp, next_tp))
        plt.close()

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Kernel same as in SpatialPCA
'''
class Kernel(nn.Module):
    def __init__(self, kernel_type='Gaussian', scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(Kernel, self).__init__()
        self.kernel_type = kernel_type
        self.fixed_scale = fixed_scale
        if self.fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        d = torch.cdist(x, y, p=2) # Euclidean distance
        # kernel type
        if self.kernel_type == 'Gaussian':
            if self.fixed_scale:
                res = torch.exp(-1*torch.square(d)/self.scale)
            else:
                res = torch.exp(-1*torch.square(d)/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        if self.kernel_type == 'Cauchy':
            if self.fixed_scale:
                res = 1 / (1 + 1*torch.square(d)/self.scale)
            else:
                res = 1 / (1 + 1*torch.square(d)/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        if self.kernel_type == 'Quadratic':
            if self.fixed_scale:
                res = 1 - 1 * torch.square(d) / (1 * torch.square(d) + self.scale)
            else:
                res = 1 - 1 * torch.square(d) / (1 * torch.square(d) + torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        if self.kernel_type == 'Matern':
            
            pass
        return res

    def print_scale(self):
        print(self.scale)