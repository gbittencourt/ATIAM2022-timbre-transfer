import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        latent_dim = 100
        num_classes = 10
        input_dim = latent_dim 

        self.model = nn.Sequential(
            # Reshape input into 7x7x256 tensor via a fully connected layer
            #nn.Linear(z_dim,256*7*7),
            #Reshape((256,7,7,)),
            #nn.BatchNorm1d(7*7*256),
            #nn.ReLU(),
            #  100x1x1 -> 128x7x7
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7,stride=1),#,padding=1,output_padding=1)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
            # 128x7x7 -> 64x14x14
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),  #,padding=1,output_padding=1),
            # Batch normalization
            nn.BatchNorm2d(64),
            # Leaky ReLU activation
            nn.ReLU(),
            # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
            # 64x14x14 -> 32x28x28
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),   #,padding=1),
            # Batch normalization
            nn.BatchNorm2d(32),
            # Leaky ReLU activation
            nn.ReLU(),
            # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
            #nn.ConvTranspose2d(64,1,kernel_size=4,stride=2),   #,padding=1,output_padding=1),
            nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2),
            # Output layer with tanh activation
            nn.Tanh()
        )
        

    def forward(self, x):
        #c = self.label_embedding(labels)
        #x = torch.cat([x,c], 1)
        #output = self.fc(x)
        #output = output.view(-1, 256, 7, 7)
        x=x.view(x.size(0),z_dim,1,1)
        output = self.model(x)
        #output = output.view(-1, 1, 28, 28)
        return output