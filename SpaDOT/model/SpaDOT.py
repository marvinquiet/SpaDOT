import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SVGPEncoder, GATEncoder
from .decoder import Decoder
from .svgp import SVGP

class SpaDOT(nn.Module):
    def __init__(self, model_config, dataloader_dict):
        super(SpaDOT, self).__init__()
        self.input_dim = model_config['input_dim']
        self.SVGP_z_dim = model_config['z_dim'] // 2
        self.GAT_z_dim = model_config['z_dim'] // 2
        self.dtype = model_config['dtype']
        self.device = model_config['device']

        # Build encoders
        self.SVGPEncoder = SVGPEncoder(
            input_dim=self.input_dim,
            SVGP_z_dim=self.SVGP_z_dim,
            hidden_dims=model_config['svgp_encoder_layers']
        ).to(dtype=self.dtype)

        self.GATEncoder = GATEncoder(
            input_dim=self.input_dim,
            GAT_z_dim=self.GAT_z_dim,
            hidden_dim=model_config['gat_encoder_hidden'],
            num_heads=model_config['gat_attention_heads']
        ).to(dtype=self.dtype)

        # Build decoder
        self.decoder = Decoder(
            input_dim=self.input_dim,
            z_dim=self.SVGP_z_dim + self.GAT_z_dim, # in case input z_dim is odd
            decoder_layers=model_config['decoder_layers']
        ).to(dtype=self.dtype)

        # Build SVGP
        self.svgp_dict = nn.ModuleDict({
            str(tp): SVGP(
                model_config=model_config,
                inducing_points=dataloader_dict['inducing_points'][tp],
                N_train=dataloader_dict['N_train'][tp]
            ).to(model_config['device']) for tp in model_config['timepoints']
        })
        # Kmeans/OT-related parameters
        self.gammas = {}
        self.kmeans_center_dict = {}
        self.kmeans_cluster_dict = {}
        self.kmeans_index_dict = {}

    def forward(self, x, y, edge_index, tp, batch_size):
        # SVGP latent
        SVGP_qnet_mu, SVGP_qnet_var = self.SVGPEncoder(y[:batch_size])
        inside_elbo_recon, inside_elbo_kl = [], []
        SVGP_p_m, SVGP_p_v = [], []
        for l in range(self.SVGP_z_dim):
            p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp_dict[str(tp)].approximate_posterior_params(x[:batch_size], x[:batch_size],
                                                                    SVGP_qnet_mu[:, l], SVGP_qnet_var[:, l])
            inside_elbo_recon_l, inside_elbo_kl_l = self.svgp_dict[str(tp)].variational_loss(x=x[:batch_size], y=SVGP_qnet_mu[:, l],
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
        inside_elbo = inside_elbo_recon - (batch_size / self.svgp_dict[str(tp)].N_train) * inside_elbo_kl
        SVGP_p_m = torch.stack(SVGP_p_m, dim=1)
        SVGP_p_v = torch.stack(SVGP_p_v, dim=1)
        ce_term = self._gauss_cross_entropy(SVGP_p_m, SVGP_p_v, SVGP_qnet_mu, SVGP_qnet_var)
        ce_term = torch.sum(ce_term)
        diff = ce_term - inside_elbo
        SVGP_KL = (-diff if ce_term.item() > inside_elbo.item() else diff) / self.SVGP_z_dim #  force KL to be negative (Recon - beta*KL), stablize training
        SVGP_latent_sample = SVGP_p_m + torch.randn_like(SVGP_p_m) * torch.sqrt(SVGP_p_v)

        # GAT latent
        GAT_m, GAT_v = self.GATEncoder(y, edge_index)
        GAT_m, GAT_v = GAT_m[:batch_size, :], GAT_v[:batch_size, :]
        GAT_latent_sample = GAT_m + torch.randn_like(GAT_m) * torch.sqrt(GAT_v)
        GAT_KL = -0.5 * torch.sum(1 + torch.log(GAT_v) - GAT_m.pow(2) - GAT_v) / self.GAT_z_dim

        # Concatenated latent samples
        final_latent = torch.cat([SVGP_latent_sample, GAT_latent_sample], dim=1)
        # Reconstruction loss
        recon_loss = torch.sum((y[:batch_size]-self.decoder(final_latent))**2) / self.input_dim
        # Alignment loss
        alignment_loss = F.mse_loss(SVGP_latent_sample.norm(dim=1) / self.SVGP_z_dim, 
                                    GAT_latent_sample.norm(dim=1) / self.GAT_z_dim, 
                                    reduction='sum')
        return recon_loss, SVGP_KL, GAT_KL, alignment_loss, final_latent
    
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
        # solve warinings
        X = torch.tensor(X, dtype=self.dtype, device=self.device)
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        SVGP_qnet_mu, SVGP_qnet_var = self.SVGPEncoder(Y)
        SVGP_p_m, SVGP_p_v = [], []
        for l in range(self.SVGP_z_dim):
            p_m_l, p_v_l, _, _ = self.svgp_dict[str(tp)].approximate_posterior_params(X, X, SVGP_qnet_mu[:, l], SVGP_qnet_var[:, l])
            SVGP_p_m.append(p_m_l)
            SVGP_p_v.append(p_v_l)
        SVGP_p_m = torch.stack(SVGP_p_m, dim=1)
        SVGP_p_v = torch.stack(SVGP_p_v, dim=1)

        GAT_m, _ = self.GATEncoder(Y, edge_index)
        p_m = torch.cat((SVGP_p_m, GAT_m), dim=1)
        latent_samples = p_m.data.cpu().detach().numpy()
        return latent_samples
    
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
