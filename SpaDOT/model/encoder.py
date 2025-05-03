import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SVGPEncoder(nn.Module):
    def __init__(self, input_dim, SVGP_z_dim, hidden_dims):
        super(SVGPEncoder, self).__init__()
        # Create a sequential network with specified hidden dimensions
        layers = [input_dim] + hidden_dims
        SVGP_encoder_net = []
        for i in range(1, len(layers)):
            linear_layer = nn.Linear(layers[i-1], layers[i])
            nn.init.xavier_uniform_(linear_layer.weight)
            SVGP_encoder_net.append(linear_layer)
            # normalization
            SVGP_encoder_net.append(nn.BatchNorm1d(layers[i]))
            # activation
            SVGP_encoder_net.append(nn.LeakyReLU())
        self.SVGP_encoder_net = nn.Sequential(*SVGP_encoder_net)
        self.SVGP_fc = nn.Linear(hidden_dims[-1], SVGP_z_dim*2)
        nn.init.xavier_uniform_(self.SVGP_fc.weight)


    def forward(self, x):
        '''
        x: gene expression
        '''
        h = self.SVGP_encoder_net(x)
        # Compute mean and log variance for the latent space
        SVGP_z = self.SVGP_fc(h)
        SVGP_enc_mu, SVGP_enc_logvar = torch.chunk(SVGP_z, 2, dim=1)
        return SVGP_enc_mu, torch.exp(SVGP_enc_logvar)  # Return mean and variance

# Define the GATEncoder class, which uses Graph Attention Networks (GAT) for encoding graph data
class GATEncoder(nn.Module):
    def __init__(self, input_dim, GAT_z_dim, hidden_dim=512, num_heads=4):
        super(GATEncoder, self).__init__()
        # Define multiple GAT layers with specified dimensions and heads
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        nn.init.xavier_uniform(self.gat1.lin.weight)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        nn.init.xavier_uniform(self.gat2.lin.weight)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=False)
        nn.init.xavier_uniform(self.gat3.lin.weight)
        self.GAT_fc = nn.Linear(hidden_dim, GAT_z_dim*2)
        nn.init.xavier_uniform_(self.GAT_fc.weight)

    def forward(self, x, edge_index):
        '''
        x: gene expression
        edge_index: graph structure
        '''
        # Pass input through each GAT layer with LeakyReLU activation
        h = F.leaky_relu(self.gat1(x, edge_index))
        h = F.leaky_relu(self.gat2(h, edge_index))
        h = self.gat3(h, edge_index)
        GAT_z = self.GAT_fc(h)
        GAT_enc_mu, GAT_enc_logvar = torch.chunk(GAT_z, 2, dim=1)
        return GAT_enc_mu, torch.exp(GAT_enc_logvar)

