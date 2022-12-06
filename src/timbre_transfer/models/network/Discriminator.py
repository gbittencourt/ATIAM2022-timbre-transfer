
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        num_channels = 1
        num_classes = 10
        in_channels = num_channels 

        self.model = nn.Sequential(
            # Convolutional layer, from 1x28x28 into 16x14x14 tensor
            nn.Conv2d(num_channels, 16, kernel_size=4, stride=2, padding=1),
            # Batch normalization
            nn.BatchNorm2d(16),
            # Leaky ReLU activation
            nn.LeakyReLU(0.01),
            #nn.Dropout(0.3),
            # Convolutional layer, from 16x14x14 into 32x7x7 tensor
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            # Batch normalization
            nn.BatchNorm2d(32),     
            # Leaky ReLU activation
            nn.LeakyReLU(0.01),
            #nn.Dropout(0.3),
            #nn.Conv2d(32,1,kernel_size=7,stride=1),
            # 32x7x7 -> 1568
            nn.Flatten(start_dim=1),
            nn.Linear(32*7*7, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 1),

            #nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            # Batch normalization
            #nn.BatchNorm2d(256),     
            # Leaky ReLU activation
            #nn.LeakyReLU(0.01),
  
            # Output layer with sigmoid activation
            #nn.Flatten(),
            #nn.Sigmoid()
        )


    def forward(self, x):
        # convertit le tenseur (batch, 784) en (batch, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)
        #x = x.Reshape(1,28,28)
        #c = self.label_embedding(labels)
        #c = c.view(x.size(0), 10, 28, 28)
        #c = c.Reshape(10,28,28)
        #x = torch.cat([x, c], dim=1)
        output = self.model(x)
        #output = output.view(len(output),1)
        # applatit toutes les dimensions à partir de la n°1 en une seule dimension
        # convertit un tenseur (batch, 1, 1, 1) en un tenseur (batch, 1)
        #output = output.flatten(start_dim=1)
        return output