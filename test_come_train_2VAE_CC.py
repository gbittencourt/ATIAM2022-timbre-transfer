import os

import torch
import torch.nn

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.transforms import ToTensor 

from src.timbre_transfer.datasets.NSynthDataset import NSynthDoubleDataset
from src.timbre_transfer.helpers.audiotransform import AudioTransform

from src.timbre_transfer.models.network.Spectral_Decoder import Spectral_Decoder
from src.timbre_transfer.models.network.Spectral_Encoder import Spectral_Encoder
from src.timbre_transfer.models.Spectral_VAE import SpectralVAE
from src.timbre_transfer.train_vae import trainStep_VAE, computeLoss_VAE
from src.timbre_transfer.train_cycleConsistency import trainStep_CC, computeLoss_CC

from torch.utils.tensorboard import SummaryWriter

dataset_folder = "data"


preTrained_loadNames = ["pretrained/exp_1_VAE_CC/vocal", "pretrained/exp_1_VAE_CC/string"]
preTrained_saveName = ["pretrained/exp_1_VAE_CC/vocal", "pretrained/exp_1_VAE_CC/string"]
writer = SummaryWriter(os.path.join('runs','test_2VAEs_CC'))


## Name of the saved trained network

## Training parameters

# Number of Epochs
epochs = 20
# Learning rate
lr = 1e-5
# Reconstruction Loss (always use reduction='none')
recons_criterion = torch.nn.MSELoss(reduction = 'none')

# Beta-VAE Beta coefficient and warm up length
beta_end = .1
warm_up_length = 10 #epochs

# Dataloaders parameters
train_batch_size = 8
valid_batch_size = 1024
num_threads = 0


## Model Parameters
# Dimension of the linear layer
hidden_dim = 256
# Dimension of the latent space
latent_dim = 8
# Number of filters of the first convolutionnal layer
base_depth = 64
# Max number of channels of te convolutionnal layers
max_depth = 512
# Number of convolutionnal layers
n_convLayers = 3
# Kernel size of convolutionnal layers (recommended : stride*2+3)
kernel_size = 11
# Stride of convolutionnal layers (recommended : 2 or 4)
stride = 4
# Models returns images of size freqs_dim*len_dim
freqs_dim = 128
len_dim = 128


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

AT = AudioTransform(input_freq = 16000, n_fft = 1024, n_mel = freqs_dim, stretch_factor=.8)

## Loading the NSynth dataset
train_dataset = NSynthDoubleDataset(
    dataset_folder,
    usage = 'train',
    filter_keys = ('vocal_acoustic', 'string_acoustic'),
    transform = AT,
    normalization = 11.57,
    length_style = 'min'
)

valid_dataset = NSynthDoubleDataset(
    dataset_folder,
    usage = 'valid',
    filter_keys = ('vocal_acoustic', 'string_acoustic'),
    transform = AT,
    normalization = 11.57,
    length_style = 'max'
)

nb_train = len(train_dataset)
nb_valid = len(valid_dataset)

print(f"Number of training examples : {nb_train}")
print(f"Number of validation examples : {nb_valid}")


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_threads, shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, num_workers=num_threads, shuffle=True)

## Model definition
encoder = Spectral_Encoder(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

decoder1 = Spectral_Decoder(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

decoder2 = Spectral_Decoder(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

model1 = SpectralVAE(encoder, decoder1, freqs_dim = freqs_dim, len_dim = len_dim, encoding_dim = hidden_dim, latent_dim = latent_dim)
model2 = SpectralVAE(encoder, decoder2, freqs_dim = freqs_dim, len_dim = len_dim, encoding_dim = hidden_dim, latent_dim = latent_dim)


## Loading pre-trained model
if os.path.isfile(preTrained_loadNames[0]+'.pt') and os.path.isfile(preTrained_loadNames[1]+'.pt'):
    model1.load_state_dict(torch.load('./'+preTrained_loadNames[0]+'.pt'))
    model2.load_state_dict(torch.load('./'+preTrained_loadNames[1]+'.pt'))

# Optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)


model1 = model1.to(device)
model2 = model2.to(device)

beta = 0
## Training
for epoch in range(epochs):
    train_losses = {}
    valid_losses = {}

    train_loss_VAE = 0
    valid_loss_VAE = 0

    train_loss_CC = 0
    valid_loss_CC = 0
    
    beta = min(beta_end, epoch/warm_up_length*beta_end)
    iter_loaders = []

    for i, (x1, x2) in enumerate(iter(train_loader)):
        
        x1 = x1.to(device)
        x2 = x2.to(device)

        # VAEs train steps
        l = trainStep_VAE(model1, x1, optimizer1, beta)[0]
        train_loss_VAE += l.cpu().detach().numpy()*x1.size()[0]/nb_train

        l = trainStep_VAE(model2, x2, optimizer2, beta)[0]
        train_loss_VAE += l.cpu().detach().numpy()*x2.size()[0]/nb_train

    # Saving the trained model
    torch.save(model1.state_dict(), preTrained_saveName[0]+'.pt')
    torch.save(model2.state_dict(), preTrained_saveName[1]+'.pt')

    with torch.no_grad():
        for i, (x1, x2) in enumerate(iter(train_loader)):
            x1 = x1.to(device)
            x2 = x2.to(device)

            l = computeLoss_VAE(model1, x1, beta)[0]
            valid_loss_VAE += l.cpu().detach().numpy()*x1.size()[0]/nb_valid

            l = computeLoss_VAE(model2, x2, beta)[0]
            valid_loss_VAE += l.cpu().detach().numpy()*x2.size()[0]/nb_valid

    #torch.save(model1.state_dict(), preTrained_saveName+'.pt')
    loss_dict = {}
    loss_dict['Training, VAE'] = train_loss_VAE
    loss_dict['Validation, VAE'] = valid_loss_VAE
    writer.add_scalars("Full Loss",loss_dict, epoch)

    print(f'epoch : {epoch}, beta  : {round(beta,2)}')
    
    with torch.no_grad():
        x1_test, x2_test = next(iter(valid_loader))
        x1_test = x1_test[:16].to(device)
        x2_test = x2_test[:16].to(device)

        y1_test = model1(x1_test)[0]
        y2_test = model2(x2_test)[0]

        x1_grid = torchvision.utils.make_grid(x1_test)
        x2_grid = torchvision.utils.make_grid(x2_test)

        y1_grid = torchvision.utils.make_grid(y1_test)
        y2_grid = torchvision.utils.make_grid(y2_test)

        writer.add_image("model 0, input image", x1_grid, epoch)
        writer.add_image("model 0, output image", y1_grid, epoch)

        writer.add_image("model 1, input image", x2_grid, epoch)
        writer.add_image("model 1, output image", y2_grid, epoch)
    
#    
#        x_test, x2_test = next(iter(valid_loader))
#
#        x_test = x_test[0]
#        x_test = x_test[0:16]
#        x_test = x_test.to(device)
#        model = models[modelIdx]
#        
#        y_test = model(x_test)[0]
#        
#        x_grid = torchvision.utils.make_grid(x_test)
#        y_grid = torchvision.utils.make_grid(y_test/torch.max(y_test))
#
#        writer.add_image("model " + str(modelIdx) + ", input image", x_grid, epoch)
#        writer.add_image("model " + str(modelIdx) + ", output image", y_grid, epoch)
#
#    
#
#        if (epoch+1)%20==0:
#            print('Exporting sound')
#            AT = AT.to('cpu')
#            x_test = x_test.to('cpu')
#            y_test = y_test.to('cpu').detach()
#            x_test_sound = AT.inverse(mel = x_test[0])
#            y_test_sound = AT.inverse(mel = y_test[0])
#
#            x_test_sound = x_test_sound/torch.max(torch.abs(x_test_sound))
#            y_test_sound = y_test_sound/torch.max(torch.abs(y_test_sound))
#
#            writer.add_audio("model " + str(modelIdx) + ", input audio", x_test_sound, sample_rate=16000, global_step=epoch)
#            writer.add_audio("model " + str(modelIdx) + ", output audio", y_test_sound, sample_rate=16000, global_step=epoch)
#            print('Exported !\n')
writer.flush()
writer.close()