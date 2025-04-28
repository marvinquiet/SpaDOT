import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SVGP
import utils as utils

import model_utils as model_utils

class SpaDOT(nn.Module):
    def __init__(self, input_dim, SVGP_z_dim, GNN_z_dim, 
                 # VAE parameters
                 encoder_layers, decoder_layers,
                 # GP parameters
                 fixed_inducing_points, initial_inducing_points, 
                 fixed_gp_params, kernel_type, kernel_scale, N_train_dict, 
                 dtype, device, jitter=1e-2):
        super(spaDOT, self).__init__()
        self.input_dim = input_dim
        self.dtype = dtype
        # --- separate SVGP z_dim and GNN_z_dim
        self.SVGP_z_dim = SVGP_z_dim
        self.GNN_z_dim = GNN_z_dim
        self.device = device
        # build VAE
        self.SVGPEncoder = model_utils.SVGPEncoder(input_dim=input_dim, SVGP_z_dim=SVGP_z_dim, encoder_layers=encoder_layers).to(dtype=dtype)
        self.GATEncoder = model_utils.GATEncoder(input_dim=input_dim, GNN_z_dim=GNN_z_dim).to(dtype=dtype)
        # --- combine decoder
        self.decoder = model_utils.Decoder(input_dim, SVGP_z_dim+GNN_z_dim, decoder_layers).to(dtype=dtype)
        # build SVGP
        self.svgp_dict = nn.ModuleDict()
        for tp in initial_inducing_points.keys():
            self.svgp_dict[tp] = SVGP(fixed_inducing_points=fixed_inducing_points, 
                    initial_inducing_points=initial_inducing_points[tp],
                    fixed_gp_params=fixed_gp_params, kernel_type=kernel_type, kernel_scale=kernel_scale[tp],
                    N_train=N_train_dict[tp], jitter=jitter, dtype=dtype, device=device)
        # --- add OT related parameters
        self.gammas = {}
        self.kmeans_center_dict = dict()
        self.kmeans_cluster_dict = dict()
        self.kmeans_index_dict = dict()
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x, y, edge_index, tp, batch_size):
        """
        Forward pass.
        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of preprocessed counts.
        edge_index: mini-batch of edges.
        
        Both x and y is center scaled for better performance.
        """ 
        b = batch_size
        # --- obtain GNN latent
        GNN_mu, GNN_var = self.GATEncoder(y, edge_index)
        GNN_mu, GNN_var = GNN_mu[:b, :], GNN_var[:b, :]
        GNN_latent_sample = GNN_mu + torch.randn_like(GNN_mu) * torch.sqrt(GNN_var)

        # --- obtain SVGP latent
        x = x[:b]
        y = y[:b]
        SVGP_qnet_mu, SVGP_qnet_var = self.SVGPEncoder(y) 
        inside_elbo_recon, inside_elbo_kl = [], []
        SVGP_p_m, SVGP_p_v = [], []
        for l in range(self.SVGP_z_dim):
            p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp_dict[tp].approximate_posterior_params(x, x,
                                                                    SVGP_qnet_mu[:, l], SVGP_qnet_var[:, l])
            inside_elbo_recon_l, inside_elbo_kl_l = self.svgp_dict[tp].variational_loss(x=x, y=SVGP_qnet_mu[:, l],
                                                                    noise=SVGP_qnet_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)
            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            SVGP_p_m.append(p_m_l)
            SVGP_p_v.append(p_v_l)
        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)
        inside_elbo = inside_elbo_recon - (b / self.svgp_dict[tp].N_train) * inside_elbo_kl
        SVGP_p_m = torch.stack(SVGP_p_m, dim=1)
        SVGP_p_v = torch.stack(SVGP_p_v, dim=1)
        # cross entropy term
        ce_term = model_utils.gauss_cross_entropy(SVGP_p_m, SVGP_p_v, SVGP_qnet_mu, SVGP_qnet_var)
        ce_term = torch.sum(ce_term)
        if ce_term.item() > inside_elbo.item(): # force KL to be negative (Recon - beta*KL), stablize training
            SVGP_KL_term = (- ce_term + inside_elbo) / self.SVGP_z_dim # normalize by number of dimensions
        else:
            SVGP_KL_term = (ce_term - inside_elbo) / self.SVGP_z_dim
        SVGP_latent_sample = SVGP_p_m + torch.randn_like(SVGP_p_m) * torch.sqrt(SVGP_p_v)

        # --- GNN latent
        latent_sample = torch.cat([SVGP_latent_sample, GNN_latent_sample], dim=1)
        # --- Reconstruction loss
        feature_recon = self.decoder(latent_sample)
        recon_loss = torch.sum((y-feature_recon)**2) / self.input_dim
        alignment_loss = F.mse_loss(SVGP_latent_sample.norm(dim=1) / self.SVGP_z_dim, GNN_latent_sample.norm(dim=1) / self.GNN_z_dim, reduction='sum')
        # adj_recon = torch.sigmoid(torch.mm(latent_sample, latent_sample.t())) # reconstruct adjacency matrix
        GNN_KL_term = -0.5 * torch.sum(1 + torch.log(GNN_var) - GNN_mu.pow(2) - GNN_var) / self.GNN_z_dim
        return recon_loss, SVGP_KL_term, GNN_KL_term, alignment_loss, latent_sample, SVGP_latent_sample, GNN_latent_sample

    def all_latent_samples(self, X, Y, edge_index, tp):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        """ 
        X = torch.tensor(X, dtype=self.dtype).to(self.device)
        Y = torch.tensor(Y, dtype=self.dtype).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        SVGP_qnet_mu, SVGP_qnet_var = self.SVGPEncoder(Y)
        SVGP_p_m, SVGP_p_v = [], []
        for l in range(self.SVGP_z_dim):
            p_m_l, p_v_l, _, _ = self.svgp_dict[tp].approximate_posterior_params(X, X, SVGP_qnet_mu[:, l], SVGP_qnet_var[:, l])
            SVGP_p_m.append(p_m_l)
            SVGP_p_v.append(p_v_l)
        SVGP_p_m = torch.stack(SVGP_p_m, dim=1)
        SVGP_p_v = torch.stack(SVGP_p_v, dim=1)

        GNN_mu, _ = self.GATEncoder(Y, edge_index)
        p_m = torch.cat((SVGP_p_m, GNN_mu), dim=1) # do I need to standardize this as well?
        latent_samples = p_m.data.cpu().detach().numpy()
        return latent_samples


    # def batching_latent_samples(self, X, Y, edge_index, tp, batch_size=512):
    #     """
    #     Output latent embedding.

    #     Parameters:
    #     -----------
    #     X: array_like, shape (n_spots, 2)
    #         Location information.
    #     Y: array_like, shape (n_spots, n_genes)
    #         Preprocessed count matrix.
    #     """ 
    #     # X = torch.tensor(X, dtype=self.dtype)
    #     # Y = torch.tensor(Y, dtype=self.dtype)
    #     latent_samples = []
    #     num = X.shape[0]
    #     num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
    #     for batch_idx in range(num_batch):
    #         xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
    #         ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
    #         qnet_mu, qnet_var = self.encoder(ybatch, edge_index_batch)
    #         p_m, p_v = [], []
    #         for l in range(self.z_dim):
    #             p_m_l, p_v_l, _, _ = self.svgp_dict[tp].approximate_posterior_params(xbatch, xbatch, qnet_mu[:, l], qnet_var[:, l])
    #             p_m.append(p_m_l)
    #             p_v.append(p_v_l)
    #         p_m = torch.stack(p_m, dim=1)
    #         p_v = torch.stack(p_v, dim=1)
    #         # SAMPLE
    #         latent_samples.append(p_m.data.cpu().detach())
    #     latent_samples = torch.cat(latent_samples, dim=0)
    #     return latent_samples.numpy()
