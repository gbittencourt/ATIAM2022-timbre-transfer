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

class SpectralVAE(AE):
    """Spectral Variational Auto-Encoder class
    args :
        - encoder [torch.nn Module]: encoder of the autoencoder
        - decoder [torch.nn Module]: decoder of the autoencoder
        - encoding_dim [int] : dimension of the space to which the inputs are encoded"""

    def __init__(self, encoder, decoder,freqs_dim, len_dim, encoding_dim, latent_dim):
        super(SpectralVAE, self).__init__(encoder, decoder, encoding_dim)

        self.latent_dims = latent_dim
        self.freqs_dim = freqs_dim
        self.len_dim = len_dim
        self.mu_sigma = nn.Linear(self.encoding_dims, 2*self.latent_dims)
        self.Softplus = nn.Softplus()
        
    def encode(self, x):
        encoded = self.encoder(x)

        mu_sigma = self.mu_sigma(encoded)
        mu = mu_sigma[:,:self.latent_dims]
        sigma = nn.Softplus(mu_sigma[:,self.latent_dims:])
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde.reshape(-1, 1, self.freqs_dim, self.len_dim), kl_div
    
    def latent(self, z_params):
        z = z_params[0]+torch.randn_like(z_params[0])*z_params[1]
        kl_div = -(1 + torch.log(torch.square(z_params[1])) - torch.square(z_params[0]) - torch.square(z_params[1]))/2
        return z, kl_div




def latentInterpolation(model, x1, x2, position : int, mode = 'linear'):
    """Finds a point between to samples in the latent space
    args :
        - model : VAE model
        - x1, x2 : samples between which you want to interpolate
        - position : The relative distance of the interpolated point between the two encoded points. 
            position = 0 => returns x1
            position = 1 => returns x2
        - mode : which interpolation mode you want to use. Interpolation modes :
            - 'linear'
            - 'spherical'"""
    
    with torch.no_grad():
        # Encoding to get both mu
        mu1,_ = model.encode(x1)
        mu2,_ = model.encode(x2)
        # Decoding
        if mode == 'linear':
            # Calculating the coordinates of the interpolated point
            mu_dec = mu2*position + mu1*(1-position)

            # Decoding
            y = model.decode(mu_dec)
            
        if mode == 'spherical':
            # Calculating the radius and angle of the two encoded points
            r1 = torch.sqrt(torch.sum(torch.square(mu1)))
            r2 = torch.sqrt(torch.sum(torch.square(mu2)))
            latent_space_dim = mu1.size()[1]
            phi1 = torch.zeros(1,latent_space_dim-1)
            phi2 = torch.zeros(1,latent_space_dim-1)
            for dim in range(latent_space_dim-1):
                phi1[:,dim] = torch.arccos(mu1[:,dim]/torch.sqrt(torch.sum(torch.square(mu1[:,dim:]))))
                phi2[:,dim] = torch.arccos(mu2[:,dim]/torch.sqrt(torch.sum(torch.square(mu2[:,dim:]))))
            
            # Calculating the radius and angle of interpolated point
            r_dec = r2*position + r1*(1-position)
            phi_dec = phi2*position+phi1*(1-position)

            #Calculating the cartesian coordinates of the interpolated point
            mu_dec = torch.zeros((1, latent_space_dim))
            mu_dec[0,0] = 1
            for dim in range(1,latent_space_dim):
                mu_dec[:,dim] = torch.prod(torch.sin(phi_dec[:,:dim-1]))
            for dim in range(latent_space_dim-1):
                mu_dec *= torch.cos(phi_dec[:,dim])
            mu_dec*=r_dec

            # Decoding
            y = model.decode(mu_dec)
    return y