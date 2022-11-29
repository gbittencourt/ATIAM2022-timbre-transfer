import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, nin, n_latent = 16, n_hidden = 512, n_classes = 1,conv_channels = 1):
        super(Decoder, self).__init__()
        self.conv_channels = conv_channels
        self.Lin1 = nn.Linear(n_latent, n_hidden)
        self.Lin2 = nn.Linear(n_hidden, ((28-8)**2)*32)
        self.ConvT1 = nn.ConvTranspose2d(32, 16, 5, stride = 1)
        self.ConvT2 = nn.ConvTranspose2d(16, 1, 5, stride = 1)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h = self.ReLU(self.Lin1(x))
        h = self.ReLU(self.Lin2(h))
        h = h.reshape(-1, 32, 28-8, 28-8)
        #h = self.ReLU(self.ConvT1(h))
        h = self.ReLU(self.ConvT1(h))
        x_hat = self.Sigmoid(self.ConvT2(h))
        return x_hat
