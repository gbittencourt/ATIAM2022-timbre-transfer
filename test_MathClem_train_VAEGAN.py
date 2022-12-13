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
from src.timbre_transfer.models.Spectral_VAE import SpectralVAE
from src.timbre_transfer.train_vae import trainStep_betaVAE, computeLoss_VAE
from torch.utils.tensorboard import SummaryWriter

dataset_folder = "data"

preTrained_loadNames = ["exp_VAEGAN/exp1", "exp_VAEGAN/exp1"]
preTrained_saveName = ["exp_VAEGAN/exp1", "exp_VAEGAN/exp1"]
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


