import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SVGPEncoder, GATEncoder
from .decoder import Decoder
from .svgp import SVGP

class SpaDOT(nn.Module):
    def __init__(self, config):
        super(SpaDOT, self).__init__()
        self.config = config

        # Build encoders
        self.SVGPEncoder = SVGPEncoder(
            input_dim=config.input_dim,
            SVGP_z_dim=config.SVGP_z_dim,
            hidden_dims=config.encoder_layers
        ).to(dtype=config.dtype)

        self.GATEncoder = GATEncoder(
            input_dim=config.input_dim,
            GAT_z_dim=config.GAT_z_dim
        ).to(dtype=config.dtype)

        # Build decoder
        self.decoder = Decoder(
            input_dim=config.input_dim,
            z_dim=config.SVGP_z_dim + config.GNN_z_dim,
            decoder_layers=config.decoder_layers
        ).to(dtype=config.dtype)

        # Build SVGP
        self.svgp_dict = nn.ModuleDict({
            tp: SVGP(
                config=config,
                initial_inducing_points=config.initial_inducing_points[tp],
                N_train=config.N_train_dict[tp],
                kernel_scale=config.kernel_scale[tp]
            ) for tp in config.initial_inducing_points.keys()
        })

        # OT-related parameters
        self.gammas = {}
        self.kmeans_center_dict = {}
        self.kmeans_cluster_dict = {}
        self.kmeans_index_dict = {}
        self.to(config.device)

    def forward(self, x, y, edge_index, tp, batch_size):
        # SVGP latent
        SVGP_qnet_mu, SVGP_qnet_var = self.SVGPEncoder(y[:batch_size])
        inside_elbo_recon, inside_elbo_kl = [], []
        SVGP_p_m, SVGP_p_v = [], []
        for l in range(self.SVGP_z_dim):
            p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp_dict[tp].approximate_posterior_params(x[:batch_size], x[:batch_size],
                                                                    SVGP_qnet_mu[:, l], SVGP_qnet_var[:, l])
            inside_elbo_recon_l, inside_elbo_kl_l = self.svgp_dict[tp].variational_loss(x=x[:batch_size], y=SVGP_qnet_mu[:, l],
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
        inside_elbo = inside_elbo_recon - (batch_size / self.svgp_dict[tp].N_train) * inside_elbo_kl
        SVGP_p_m = torch.stack(SVGP_p_m, dim=1)
        SVGP_p_v = torch.stack(SVGP_p_v, dim=1)
        SVGP_latent_sample = SVGP_p_m + torch.randn_like(SVGP_p_m) * torch.sqrt(SVGP_p_v)
        ce_term = self._gauss_cross_entropy(SVGP_p_m, SVGP_p_v, SVGP_qnet_mu, SVGP_qnet_var)
        ce_term = torch.sum(ce_term)
        diff = ce_term - inside_elbo
        SVGP_KL = (-diff if ce_term.item() > inside_elbo.item() else diff) / self.SVGP_z_dim #  force KL to be negative (Recon - beta*KL), stablize training

        # GAT latent
        GAT_mu, GAT_var = self.GATEncoder(y, edge_index)
        GAT_mu, GAT_var = GAT_mu[:batch_size, :], GAT_var[:batch_size, :]
        GAT_latent_sample = GAT_mu + torch.randn_like(GAT_mu) * torch.sqrt(GAT_var)
        GAT_KL = -0.5 * torch.sum(1 + torch.log(GAT_var) - GAT_mu.pow(2) - GAT_var) / self.GAT_z_dim

        # Concatenated latent samples
        final_latent = torch.cat([SVGP_latent_sample, GAT_latent_sample], dim=1)
        # Reconstruction loss
        recon_loss = torch.sum((y-self.decoder(final_latent))**2) / self.input_dim
        # Alignment loss
        alignment_loss = F.mse_loss(SVGP_latent_sample.norm(dim=1) / self.SVGP_z_dim, 
                                    GAT_latent_sample.norm(dim=1) / self.GNN_z_dim, reduction='sum')
        return recon_loss, SVGP_KL, GAT_KL, alignment_loss, final_latent
    
    def _gauss_cross_entropy(self, mu1, var1, mu2, var2):
        """
        Computes the element-wise cross entropy
        Given q(z) ~ N(z| mu1, var1)
        returns E_q[ log N(z| mu2, var2) ]
        args:
            mu1:  mean of expectation (batch, tmax, 2) tf variable
            var1: var  of expectation (batch, tmax, 2) tf variable
            mu2:  mean of integrand (batch, tmax, 2) tf variable
            var2: var of integrand (batch, tmax, 2) tf variable
        returns:
            cross_entropy: (batch, tmax, 2) tf variable
        """
        term0 = 1.8378770664093453  # log(2*pi)
        term1 = torch.log(var2)
        term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2
        cross_entropy = -0.5 * (term0 + term1 + term2)
        return cross_entropy
