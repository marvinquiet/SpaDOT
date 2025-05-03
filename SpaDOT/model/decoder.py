import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, decoder_layers):
        super(Decoder, self).__init__()
        layers = [z_dim] + decoder_layers + [input_dim]
        decoder_net = []
        for i in range(1, len(layers)-1):
            linear_layer = nn.Linear(layers[i-1], layers[i])
            nn.init.xavier_uniform_(linear_layer.weight)
            decoder_net.append(linear_layer)
            # normalization
            decoder_net.append(nn.LayerNorm(layers[i]))
            # activation
            decoder_net.append(nn.LeakyReLU())
        decoder_net.append(nn.Linear(layers[-2], layers[-1]))
        self.decoder_net = nn.Sequential(*decoder_net)

    def forward(self, latent_sample):
        return self.decoder_net(latent_sample)
