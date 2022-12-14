import torch
import torch.nn as nn
import numpy as np

class AE(nn.Module):
    """Autoencoder class
    args :
        - encoder [torch.nn Module]: encoder of the autoencoder
        - decoder [torch.nn Module]: decoder of the autoencoder
        - encoding_dim [int] : dimension of the space to which the inputs are encoded"""
    def __init__(self, encoder : nn.Module, decoder : nn.Module, encoding_dim : int):
        super(AE, self).__init__()
        self.encoding_dims = encoding_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SpectralVAE_GAN(AE):
    """Spectral Variational Auto-Encoder class
    args :
        - encoder [torch.nn Module]: encoder of the autoencoder
        - decoder [torch.nn Module]: decoder of the autoencoder
        - encoding_dim [int] : dimension of the space to which the inputs are encoded"""

    def __init__(self, encoder, decoder, discriminator, freqs_dim, len_dim, encoding_dim, latent_dim):
        super(SpectralVAE_GAN, self).__init__(encoder, decoder, encoding_dim)

        self.latent_dims = latent_dim
        self.freqs_dim = freqs_dim
        self.len_dim = len_dim
        self.discriminator = discriminator

        self.mu = nn.Linear(self.encoding_dims, self.latent_dims)
        self.sigma = nn.Sequential(nn.Linear(self.encoding_dims, self.latent_dims), nn.Softplus())
        
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        sigma = self.sigma(encoded)
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)
    
    def discriminate(self, x_hat):
        return self.discriminator(x_hat)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        # Pass them through the discriminator
        result = self.discriminate(x_tilde)
        return result, x_tilde.reshape(-1, 1, self.freqs_dim, self.len_dim), kl_div
    
    def latent(self, z_params):
        z = z_params[0]+torch.randn_like(z_params[0])*z_params[1]
        kl_div = -(1 + torch.log(torch.square(z_params[1])) - torch.square(z_params[0]) - torch.square(z_params[1]))/2
        return z, kl_div