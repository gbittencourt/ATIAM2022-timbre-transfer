import torch
import torch.nn as nn
import numpy as np
from src.timbre_transfer.models.Spectral_VAE import SpectralVAE


class SpectralVAE_GAN(SpectralVAE):
    """Spectral Variational Auto-Encoder class
    args :
        - encoder [torch.nn Module]: encoder of the autoencoder
        - decoder [torch.nn Module]: decoder of the autoencoder
        - encoding_dim [int] : dimension of the space to which the inputs are encoded"""

    def __init__(self, encoder, decoder, discriminator, freqs_dim, len_dim, encoding_dim, latent_dim):
        super(SpectralVAE_GAN, self).__init__(encoder, decoder, freqs_dim, len_dim, encoding_dim, latent_dim)

        self.latent_dims = latent_dim
        self.freqs_dim = freqs_dim
        self.len_dim = len_dim
        self.discriminator = discriminator
    
    def discriminate(self, x_hat):
        return self.discriminator(x_hat)
    
    def forward(self, x):
        # Encode the inputs
        z_params = self.encode((x-.5)*1.6)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)/1.6+.5
        return x_tilde.reshape(-1, 1, self.freqs_dim, self.len_dim), kl_div