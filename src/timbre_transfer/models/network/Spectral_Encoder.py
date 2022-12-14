import torch.nn as nn
import torch
import numpy as np

class Spectral_Encoder(nn.Module):
    def __init__(self, freqs_dim, len_dim, latent_dim = 16, hidden_dim = 512, base_depth = 8, n_convLayers=4, kernel_size = 25, max_depth = 128, stride = 2):
        super(Spectral_Encoder, self).__init__()
        self.freqs_dim = freqs_dim
        self.len_dim = len_dim
        self.n_convLayers = n_convLayers
        self.kernel_size = kernel_size
        self.LastSize = int(freqs_dim//np.power(stride, n_convLayers))

        self.lastLayerChannels = freqs_dim
        self.lastLayerChannels = min(base_depth*np.power(2,n_convLayers-1), max_depth)
        

        #Definition of layers
        self.Lin1 = nn.Linear(self.LastSize*self.LastSize*self.lastLayerChannels, hidden_dim)
        self.Lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Flatten = nn.Flatten()

        #Definition of the multiple convolution layers
        self.Conv_layers = nn.Sequential()

        self.Conv_layers.append(
            nn.Conv2d(
                in_channels = 1,
                out_channels = base_depth,
                kernel_size = kernel_size,
                padding = kernel_size//2,
                stride = stride))
        self.Conv_layers.append(nn.ReLU())
        for i in range(1,n_convLayers):
            self.Conv_layers.append(
                nn.Conv2d(
                    in_channels = min(base_depth*np.power(2,i-1),self.lastLayerChannels),
                    out_channels = min(base_depth*np.power(2,i),self.lastLayerChannels),
                    kernel_size = kernel_size,
                    padding = kernel_size//2,
                    stride = stride))
            self.Conv_layers.append(nn.ReLU())
            
        
        
    def forward(self, x):
        h=x
        h = self.Conv_layers(h)
        
        h = self.Flatten(h)
        h = self.ReLU(self.Lin1(h))
        h = self.ReLU(self.Lin2(h))
        return h


