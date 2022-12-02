import torch.nn as nn
import torch
import numpy as np

class Spectral_Decoder(nn.Module):
    def __init__(self, freqs_dim, len_dim, latent_dim = 16, hidden_dim = 512, base_depth = 8, n_convLayers=4, kernel_size = 25, max_depth = 128, stride = 2):
        super(Spectral_Decoder, self).__init__()
        self.freqs_dim = freqs_dim
        self.len_dim = len_dim
        self.n_convLayers = n_convLayers
        self.kernel_size = kernel_size
        self.LastSize = int(freqs_dim//np.power(stride, n_convLayers))
        self.lastLayerChannels = min(base_depth*np.power(2,n_convLayers-1), max_depth)


        #Definition of layers
        self.Lin1 = nn.Linear(latent_dim, hidden_dim)
        self.Lin2 = nn.Linear(hidden_dim, self.LastSize*self.LastSize*self.lastLayerChannels)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        #Definition of the multiple convolution layers
        self.ConvT_layers = nn.ModuleList()
        for i in range(0,n_convLayers-1):
            self.ConvT_layers.append(nn.ConvTranspose2d(
                    min(base_depth*np.power(2,n_convLayers-i-1),self.lastLayerChannels),
                    min(base_depth*np.power(2,n_convLayers-i-2),self.lastLayerChannels),
                    kernel_size = kernel_size,
                    padding = kernel_size//2,
                    stride = stride,
                    output_padding=stride-1
                    ))
            self.ConvT_layers.append(nn.ReLU())
        self.ConvT_layers.append(nn.ConvTranspose2d(base_depth, 1, kernel_size = kernel_size, padding = kernel_size//2, stride = stride, output_padding=stride-1))
    def forward(self, x):
        h = self.ReLU(self.Lin1(x))
        h = self.ReLU(self.Lin2(h))

        h = h.reshape(
            -1,
            self.lastLayerChannels,
            self.LastSize,
            self.LastSize)

        for ConvT_layer in self.ConvT_layers:
            h = ConvT_layer(h)
        x_hat = self.Sigmoid(h)
        #x_hat = h
        return x_hat