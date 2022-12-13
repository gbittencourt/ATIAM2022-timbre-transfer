import os

import torch
import torch.nn

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.transforms import ToTensor 

from src.timbre_transfer.datasets.NSynthDataset import NSynthDataset
from src.timbre_transfer.helpers.audiotransform import AudioTransform

from src.timbre_transfer.models.network.Spectral_Decoder import Spectral_Decoder
from src.timbre_transfer.models.network.Spectral_Encoder import Spectral_Encoder
from src.timbre_transfer.models.network.Spectral_Discriminator import Spectral_Discriminator
from src.timbre_transfer.models.VAE_GAN import SpectralVAE_GAN
from src.timbre_transfer.train_VAE_GAN import trainStep_VAE_GAN ,  computeLoss
from torch.utils.tensorboard import SummaryWriter

dataset_folder = "data"

preTrained_loadName = "exp_VAEGAN/exp1"
preTrained_saveName = "exp_VAEGAN/exp1"
writer = SummaryWriter(os.path.join('runs','test_VAEGAN'))

## Name of the saved trained network

## Training parameters
# Proportion of the train dataset used for training
train_ratio = 1

# Number of Epochs
epochs = 400
# Learning rate
lr = 1e-4
# Reconstruction Loss (always use reduction='none')
recons_criterion = torch.nn.MSELoss(reduction = 'none')

# Beta-VAE Beta coefficient and warm up length
beta_end = 1
warm_up_length = 200 #epochs

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
train_dataset = NSynthDataset('data/', usage = 'train', filter_key='vocal_acoustic', transform=AT)
valid_dataset = NSynthDataset('data/', usage = 'valid', filter_key='vocal_acoustic', transform=AT)

nb_train = int(train_ratio * len(train_dataset))
nb_valid = len(valid_dataset)
print(f"Number of training examples : {nb_train}")
print(f"Number of validation examples : {nb_valid}")

train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset, [nb_train, len(train_dataset)-nb_train])

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

decoder = Spectral_Decoder(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

discriminator = Spectral_Discriminator(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

model = SpectralVAE_GAN(encoder, decoder, discriminator, freqs_dim = freqs_dim, len_dim = len_dim, encoding_dim = hidden_dim, latent_dim = latent_dim)

## Loading pre-trained model
if os.path.isfile(preTrained_loadName+'.pt'):
    model.load_state_dict(torch.load('./'+preTrained_loadName+'.pt'))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

beta = 0
## Training
for epoch in range(epochs):
    model = model.to(device)

    train_losses_vect = np.zeros(4)
    valid_losses_vect = np.zeros(4)

    beta = min(beta_end, epoch/warm_up_length*beta_end)
    for i, (x,_) in enumerate(iter(train_loader)):
        losses = trainStep_VAE_GAN(model, discriminator, x, optimizer, beta)
        for j,l in enumerate(losses):
            train_losses_vect[j]+=l.cpu().detach().numpy()*x.size()[0]/nb_train
    
    with torch.no_grad():
        for i , (x,_) in enumerate(iter(valid_loader)):
            x = x.to(device)
            losses = computeLoss(model, discriminator, x, beta)
            for j,l in enumerate(losses):
                valid_losses_vect[j]+=l.cpu().detach().numpy()*x.size()[0]/nb_valid
    # Saving trained model
    torch.save(model.state_dict(), preTrained_saveName+'.pt')

    writer.add_scalars("Full Loss",
        {'Training': train_losses_vect[0],
        'Validation': valid_losses_vect[0]}, epoch)
    writer.add_scalars("Reconstruction Loss",
        {'Training': train_losses_vect[1],
        'Validation': valid_losses_vect[1]}, epoch)
    writer.add_scalars("KL Divergence",
        {'Training': train_losses_vect[2],
        'Validation': valid_losses_vect[2]}, epoch)
    writer.add_scalars("Discriminator Loss",
        {'Training': train_losses_vect[3],
        'Validation': valid_losses_vect[3]}, epoch)
    print(f'epoch : {epoch}')

    x_test = next(iter(valid_loader))
    x_test = x_test[0]
    x_test = x_test[0:16]
    x_test = x_test.to(device)
    model = model.to(device)
    
    y_test = model(x_test)[0]
    
    x_grid = torchvision.utils.make_grid(x_test)
    y_grid = torchvision.utils.make_grid(y_test/torch.max(y_test))

    writer.add_image("input_image",x_grid, epoch)
    writer.add_image("output_image",y_grid, epoch)
    
    if (epoch+1)%20==0:
        print('Exporting sound')
        AT = AT.to('cpu')
        x_test = x_test.to('cpu')
        y_test = y_test.to('cpu').detach()
        x_test_sound = AT.inverse(mel = x_test[0])
        y_test_sound = AT.inverse(mel = y_test[0])
        
        x_test_sound = x_test_sound/torch.max(torch.abs(x_test_sound))
        y_test_sound = y_test_sound/torch.max(torch.abs(y_test_sound))

        writer.add_audio("input", x_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("output", y_test_sound, sample_rate=16000, global_step=epoch)
        print('Exported !\n')

writer.flush()
writer.close()
