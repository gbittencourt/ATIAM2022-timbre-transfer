import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, nin, n_latent = 16, n_hidden = 512, n_classes = 1,conv_channels = 1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride = 1), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride = 1), nn.ReLU(),
            #nn.Conv2d(conv_channels,conv_channels,3,padding = 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(((28-8)**2)*32, n_hidden), nn.ReLU(),
            #nn.Linear(n_hidden, n_hidden), nn.ReLU(),
        )
        
    def forward(self, x):
        return self.encoder(x)