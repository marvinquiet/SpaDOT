import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, decoder_layers):
        super(Decoder, self).__init__()
        layers = [z_dim] + decoder_layers + [input_dim]
        self.decoder_net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layers[i - 1], layers[i]),
                nn.LayerNorm(layers[i]),
                nn.LeakyReLU()
            ) for i in range(1, len(layers) - 1)
        ] + [nn.Linear(layers[-2], layers[-1])])

    def _initialize_weights(self):
        for module in self.decoder_net:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)

    def forward(self, latent_sample):
        return self.decoder_net(latent_sample)
