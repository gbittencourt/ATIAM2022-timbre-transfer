import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, encoder, decoder, encoding_dim):
        super(AE, self).__init__()
        self.encoding_dims = encoding_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SpectralVAE(AE):
    
    def __init__(self, encoder, decoder,freqs_dim, len_dim, encoding_dim, latent_dim):
        super(SpectralVAE, self).__init__(encoder, decoder, encoding_dim)
        self.dummy_param = nn.Parameter(torch.empty(0)) # to find device

        self.latent_dims = latent_dim
        self.freqs_dim = freqs_dim
        self.len_dim = len_dim

        self.mu = nn.Linear(self.encoding_dims, self.latent_dims)
        self.sigma = nn.Sequential(nn.Linear(self.encoding_dims, self.latent_dims), nn.Softplus())
        
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        sigma = self.sigma(encoded)
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde.reshape(-1, 1, self.freqs_dim, self.len_dim), kl_div
    
    def latent(self, x, z_params):
        device = self.dummy_param.device
        z = z_params[0] + torch.randn(self.latent_dims).to(device)*torch.square(z_params[1])
        kl_div = (1 + torch.log(torch.square(z_params[1])) - torch.square(z_params[0]) - torch.square(z_params[1]))/2
        return z, kl_div