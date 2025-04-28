import torch
from torch import optim
from tqdm.auto import tqdm
import pandas as pd
from .trainer import beta_cycle_linear, compute_kmeans_loss, compute_OT_loss, update_Kmeans, update_OT_matrix, do_eval
from time import time
import random

class Trainer:
    def __init__(self, model, args, logging):
        self.model = model
        self.args = args
        self.logging = logging
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        self.loss_dict = {}
        self.loss_names = ['elbo', 'SVGP_recon', 'SVGP_KL', 'GAT_recon', 'GAT_KL', 'alignment', 'KMeans', 'OT']

    def train(self, adata):
        self._initialize_loss_dict()
        beta1s = beta_cycle_linear(self.args.maxiter, stop=self.args.beta1)
        tp_indexed_list = list(enumerate(self.args.timepoints))
        train_starttime = time()
        for epoch in tqdm(range(self.args.maxiter)):
            self.logging.info(f'--- Epoch {epoch + 1}')
            self.model.beta1 = beta1s[epoch]
            self.model.beta2 = self.args.beta2
            self.logging.info(f'Model beta1: {self.model.beta1:.6f}, Model beta2: {self.model.beta2:.6f}')
            self.model.train()
            tp_loss_dict = self._train_epoch(tp_indexed_list, epoch)
            # Log epoch losses
            self._log_epoch_losses(epoch, tp_loss_dict)
            # Update KMeans and OT matrix
            update_Kmeans(self.model, self.args)
            if (epoch + 1) % self.args.ot_config["ot_epochs"] == 0:
                update_OT_matrix(self.model, self.args)
        self.logging.info(f'Training time: {int(time() - train_starttime)} seconds.')
        pd.DataFrame.from_dict(self.loss_dict).to_csv(self.args.result_dir + os.sep + 'loss.csv')
        return self.model

    def _initialize_loss_dict(self):
        for epoch in range(self.args.maxiter):
            self.loss_dict[epoch] = {name: 0 for name in self.loss_names}

    def _train_epoch(self, tp_indexed_list, epoch):
        tp_loss_dict = {tp: {name: 0 for name in self.loss_names} for tp in self.args.timepoints}
        random.shuffle(tp_indexed_list)

        for tp_i, tp in tp_indexed_list:
            tp_dataloader = self.model.dataloaders_dict[tp]
            tp_adj = self.model.adj_dict[tp]
            for batch in tp_dataloader:
                self._train_batch(batch, tp, tp_adj, tp_loss_dict[tp])

        for tp in self.args.timepoints:
            for name in self.loss_names:
                tp_loss_dict[tp][name] /= len(self.model.dataloaders_dict[tp])
                self.loss_dict[epoch][name] += tp_loss_dict[tp][name]

        return tp_loss_dict

    def _train_batch(self, batch, tp, tp_adj, tp_loss_dict):
        y_batch, x_batch, tp_ix, edge_index_batch = batch.x, batch.loc, batch.data_index, batch.edge_index
        x_batch, y_batch, edge_index_batch = x_batch.to(self.model.device), y_batch.to(self.model.device), edge_index_batch.to(self.model.device)
        tp_ix, adj_idx = tp_ix[:batch.batch_size], batch.n_id[:batch.batch_size]

        # Forward pass
        tp_SVGP_recon_val, tp_SVGP_KL_val, tp_GAT_KL_val, tp_alignment_val, tp_p_m= \
            self.model.forward(x=x_batch, y=y_batch, edge_index=edge_index_batch, tp=tp, batch_size=batch.batch_size)

        # Compute losses
        tp_KMeans_val = compute_kmeans_loss(self.model, tp, tp_ix, tp_p_m) if epoch >= self.args.kmeans_epoch else 0
        tp_OT_val = compute_OT_loss(self.model, tp, tp_ix, tp_p_m, self.args.timepoints[tp_i - 1]) if epoch >= self.args.ot_epoch and tp_i != 0 else 0

        tp_elbo_val = self.args.lambda1 * tp_SVGP_recon_val - self.model.beta1 * tp_SVGP_KL_val + self.model.beta2 * tp_GAT_KL_val
        tp_elbo_val += self.args.omiga1 * tp_alignment_val + self.args.omiga2 * tp_KMeans_val + self.args.omiga3 * tp_OT_val

        # Backward pass
        self.optimizer.zero_grad()
        tp_elbo_val.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
        self.optimizer.step()

        # Update losses
        for name in self.loss_names:
            tp_loss_dict[name] += locals()[f'tp_{name}_val'].detach().cpu().item()

    def _log_epoch_losses(self, epoch, tp_loss_dict):
        self.logging.info(f"Epoch {epoch + 1} Losses:")
        for tp in self.args.timepoints:
            self.logging.info(f"Timepoint {tp}: {tp_loss_dict[tp]}")