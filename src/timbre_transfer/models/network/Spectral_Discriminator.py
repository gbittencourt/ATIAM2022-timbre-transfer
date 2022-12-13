import torch.nn as nn
import numpy as np

class Spectral_Discriminator(nn.Module):
    def __init__(self, freqs_dim, len_dim, latent_dim = 16, hidden_dim = 512, base_depth = 8, n_convLayers=4, kernel_size = 25, max_depth = 128, stride = 2):
            super(Spectral_Discriminator, self).__init__()
            self.freqs_dim = freqs_dim
            self.len_dim = len_dim
            self.n_convLayers = n_convLayers
            self.kernel_size = kernel_size
            self.hidden_dim = hidden_dim
            self.LastSize = int(freqs_dim//np.power(stride, n_convLayers))
            
            self.lastLayerChannels = min(base_depth*np.power(2,n_convLayers-1), max_depth)
            
            # Definition of layers
            self.Lin1 = nn.Linear(self.LastSize*self.LastSize*self.lastLayerChannels, hidden_dim)
            self.Lin2 = nn.Linear(hidden_dim, hidden_dim)
            self.ReLU = nn.ReLU()
            self.Sigmoid = nn.Sigmoid()
            self.Flatten = nn.Flatten()

            #Definition of the multiple convolution layers
            self.Conv_layers = nn.ModuleList()
            
            self.batch_norm = nn.ModuleList()

            self.Conv_layers.append(
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = base_depth,
                    kernel_size = kernel_size,
                    padding = kernel_size//2,
                    stride = stride))
            
            self.batch_norm.append(
                nn.BatchNorm2d(base_depth)
            )
            
            for i in range(1,n_convLayers):
                self.Conv_layers.append(
                    nn.Conv2d(
                        in_channels = min(base_depth*np.power(2,i-1),self.lastLayerChannels),
                        out_channels = min(base_depth*np.power(2,i),self.lastLayerChannels),
                        kernel_size = kernel_size,
                        padding = kernel_size//2,
                        stride = stride))
                self.batch_norm(
                    nn.BatchNorm2d(min(base_depth*np.power(2,i),self.lastLayerChannels))
                )

                
    def forward(self,x):
        h = x
        
        for i,layer in enumerate(self.Conv_layers) :
            h = layer(h)
            h = self.batch_norm[i](h)
            h = self.ReLU(h)
        
        h = self.Flatten(h)
        h = self.Lin1(h)
        h = nn.BatchNorm1d(self.hidden_dim)
        h = self.ReLU(h)
        h = self.Lin2(h)
        h = nn.Sigmoid(h)
        
        return h 
        