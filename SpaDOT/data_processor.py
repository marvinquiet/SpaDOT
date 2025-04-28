import os
import random
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from .data_processer import preprocess_adata, obtain_loc_batch_info, Cal_Spatial_Net

class DataProcessor:
    def __init__(self, args):
        self.args = args

    def prepare_data(self, adata, k_cutoff=6):
        loc, batch_mat = obtain_loc_batch_info(adata)
        sampled_inducing_idx = random.sample(range(loc.shape[0]), self.args.inducing_point_nums)
        initial_inducing_points = loc[sampled_inducing_idx, :]
        np.savetxt(self.args.result_dir + os.sep + "sampled_inducing_points.txt", initial_inducing_points, delimiter=",")

        batch_to_tp_dict = dict(zip(np.argmax(batch_mat, axis=1), adata.obs['timepoint'].values))
        tp_to_batch_dict = dict(zip(adata.obs['timepoint'].values, np.argmax(batch_mat, axis=1)))
        batch_idx = np.argmax(batch_mat, axis=1)
        tps_list = [batch_to_tp_dict[idx] for idx in batch_idx]

        tp_index_dict = {tp: [i for i, t in enumerate(tps_list) if t == tp] for tp in self.args.timepoints}
        initial_inducing_points_dict, N_train_dict, kernel_scale_dict = self._create_inducing_point_dicts(adata, loc, tp_to_batch_dict, initial_inducing_points)

        dataloaders_dict, adj_dict, dataset_dict = self._create_dataloaders(adata, loc, tp_index_dict, k_cutoff)
        return initial_inducing_points_dict, kernel_scale_dict, N_train_dict, dataloaders_dict, adj_dict, dataset_dict

    def _create_inducing_point_dicts(self, adata, loc, tp_to_batch_dict, initial_inducing_points):
        initial_inducing_points_batch = np.argmax(initial_inducing_points[:, 2:], axis=1)
        initial_inducing_points_dict, N_train_dict, kernel_scale_dict = {}, {}, {}

        for i, tp in enumerate(self.args.timepoints):
            batch_id = tp_to_batch_dict[tp]
            tp_inducing_idx = np.where(initial_inducing_points_batch == batch_id)[0]
            initial_inducing_points_dict[tp] = initial_inducing_points[tp_inducing_idx, :2]
            N_train_dict[tp] = np.sum(adata.obs['timepoint'] == tp)
            kernel_scale_dict[tp] = float(self.args.kernel_scale[i])

        return initial_inducing_points_dict, N_train_dict, kernel_scale_dict

    def _create_dataloaders(self, adata, loc, tp_index_dict, k_cutoff):
        dataloaders_dict, adj_dict, dataset_dict = {}, {}, {}

        for tp in self.args.timepoints:
            tp_ix = tp_index_dict[tp]
            tp_adata = adata[tp_ix].copy()
            tp_loc = loc[tp_ix, :2]
            tp_dataset = TensorDataset(torch.tensor(tp_loc, dtype=torch.float64), torch.tensor(tp_adata.X, dtype=torch.float64), torch.tensor(tp_ix, dtype=torch.int))
            dataset_dict[tp] = tp_dataset

            Cal_Spatial_Net(tp_adata, k_cutoff=max(30, k_cutoff * round(1 / 1000 * tp_adata.n_obs)), model='KNN')
            tp_adj = torch.tensor(tp_adata.uns['adj'], dtype=torch.double)
            tp_edge_index, _ = dense_to_sparse(tp_adj)
            tp_graph_data = Data(
                x=torch.tensor(tp_adata.X, dtype=torch.double),
                edge_index=torch.tensor(tp_edge_index, dtype=torch.long),
                data_index=torch.tensor(tp_ix, dtype=torch.int),
                loc=torch.tensor(tp_loc, dtype=torch.double)
            )
            tp_graph_loader = NeighborLoader(
                tp_graph_data, num_neighbors=[max(30, k_cutoff * round(1 / 1000 * tp_adata.n_obs))] * 2, batch_size=self.args.batch_size,
                subgraph_type="induced"
            )
            dataloaders_dict[tp] = tp_graph_loader
            adj_dict[tp] = tp_adj

        return dataloaders_dict, adj_dict, dataset_dict