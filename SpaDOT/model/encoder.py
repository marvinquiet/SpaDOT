import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SVGPEncoder(nn.Module):
    def __init__(self, input_dim, SVGP_z_dim, hidden_dims):
        super(SVGPEncoder, self).__init__()
        # Create a sequential network with specified hidden dimensions
        layers = [input_dim] + hidden_dims
        self.SVGP_encoder_net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layers[i - 1], layers[i]),  # Linear layer
                nn.BatchNorm1d(layers[i]),           # Batch normalization
                nn.LeakyReLU()                       # Activation function
            ) for i in range(1, len(layers))
        ])
        self.SVGP_fc = nn.Linear(hidden_dims[-1], SVGP_z_dim * 2)
        self._initialize_weights() # Initialize weights

    def _initialize_weights(self):
        for module in self.SVGP_encoder_net:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
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
        self.gat_layers = nn.ModuleList([
            GATConv(input_dim, hidden_dim, heads=num_heads, concat=True),
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True),
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=False)
        ])
        # Fully connected layer to output mean and log variance
        self.GAT_fc = nn.Linear(hidden_dim, GAT_z_dim * 2)
        nn.init.xavier_uniform_(self.GAT_fc.weight)

    def forward(self, x, edge_index):
        '''
        x: gene expression
        edge_index: graph structure
        '''
        # Pass input through each GAT layer with LeakyReLU activation
        for gat_layer in self.gat_layers[:-1]:
            x = F.leaky_relu(gat_layer(x, edge_index))
        x = self.gat_layers[-1](x, edge_index) # Final GAT layer without activation
        GAT_z = self.GAT_fc(x)
        GAT_enc_mu, GAT_enc_logvar = torch.chunk(GAT_z, 2, dim=1)
        return GAT_enc_mu, torch.exp(GAT_enc_logvar)  # Return mean and variance

