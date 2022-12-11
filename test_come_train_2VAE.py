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

from torch.utils.tensorboard import SummaryWriter

dataset_folder = "data"

preTrained_loadNames = ["exp_2VAs/exp1_vocal_MSE", "exp_2VAs/exp1_string_MSE"]
preTrained_saveName = ["exp_2VAs/exp1_vocal_MSE", "exp_2VAs/exp1_string_MSE"]
writer = SummaryWriter(os.path.join('runs','test_2VAEs_MSE'))


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
freqs_dim = 64
len_dim = 64


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

AT = AudioTransform(input_freq = 16000, n_fft = 1024, n_mel = freqs_dim, stretch_factor=.8)

## Loading the NSynth dataset
train_dataset1 = NSynthDataset(dataset_folder, usage = 'train', filter_key='vocal_acoustic', transform=AT, normalization=16.2965)
valid_dataset1 = NSynthDataset(dataset_folder, usage = 'valid', filter_key='vocal_acoustic', transform=AT, normalization=16.2965)
train_dataset2 = NSynthDataset(dataset_folder, usage = 'train', filter_key='string_acoustic', transform=AT, normalization=16.2965)
valid_dataset2 = NSynthDataset(dataset_folder, usage = 'valid', filter_key='string_acoustic', transform=AT, normalization=16.2965)

nb_train = min(int(train_ratio * len(train_dataset1)), int(train_ratio * len(train_dataset2)))
nb_valid1, nb_valid2 = len(valid_dataset1), len(valid_dataset2)

nb_valids = [nb_valid1,nb_valid2]
print(f"Number of training examples : {nb_train}")
print(f"Number of validation examples : {nb_valid1, nb_valid2}")

train_dataset1, _ = torch.utils.data.dataset.random_split(train_dataset1, [nb_train, len(train_dataset1)-nb_train])
train_dataset2, _ = torch.utils.data.dataset.random_split(train_dataset2, [nb_train, len(train_dataset2)-nb_train])

train_datasets = [train_dataset1, train_dataset2]

train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=train_batch_size, num_workers=num_threads, shuffle=True)
valid_loader1 = torch.utils.data.DataLoader(dataset=valid_dataset1, batch_size=valid_batch_size, num_workers=num_threads, shuffle=True)

train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=train_batch_size, num_workers=num_threads, shuffle=True)
valid_loader2 = torch.utils.data.DataLoader(dataset=valid_dataset2, batch_size=valid_batch_size, num_workers=num_threads, shuffle=True)

train_loaders = [train_loader1, train_loader2]
valid_loaders = [valid_loader1, valid_loader2]

## Loss Function
def compute_loss_beta(model, x, beta):
    x_hat, kl_div = model(x)
    recons_loss = recons_criterion(x_hat,x).mean(0).sum()
    
    kl_loss = kl_div.mean(0).sum()
    if beta==0:
        full_loss = recons_loss
    else:
        full_loss = recons_loss - beta*kl_loss
    
    return full_loss, recons_loss, -kl_loss

## Train step
def train_step(model, x, optimizer, beta):
    # Compute the loss.
    full_loss, recons_loss, kl_loss = compute_loss_beta(model, x, beta)
    # Before the backward pass, zero all of the network gradients
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    full_loss.backward()
    # Calling the step function to update the parameters
    optimizer.step()
    return full_loss, recons_loss, kl_loss


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

models = [model1, model2]

## Loading pre-trained model
if os.path.isfile(preTrained_loadNames[0]+'.pt') and os.path.isfile(preTrained_loadNames[1]+'.pt'):
    model1.load_state_dict(torch.load('./'+preTrained_loadNames[0]+'.pt'))
    model2.load_state_dict(torch.load('./'+preTrained_loadNames[1]+'.pt'))

# Optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

optimizers = [optimizer1, optimizer2]

beta = 0
## Training
for epoch in range(epochs):
    train_losses = []
    valid_losses = []
    for model in models:
        model = model.to(device)
        train_losses.append(np.zeros(3))
        valid_losses.append(np.zeros(3))
    
    beta = min(beta_end, epoch/warm_up_length*beta_end)
    iter_loaders = []

    for train_loader in train_loaders:
        iter_loaders.append(iter(train_loader))
    
#    for modelIdx in range(len(models)):
#        model = models[modelIdx]
#        optimizer = optimizers[modelIdx]
#        for i, (x,_) in enumerate(iter(train_loaders[modelIdx])):
#            x = x.to(device)
#            losses = train_step_beta(model, x, optimizer, beta)
#            for j,l in enumerate(losses):
#                train_losses[modelIdx][j]+=l.cpu().detach().numpy()*x.size()[0]/nb_train
#

    len_loader = len(iter_loaders[0])
    for i in range(len_loader):
        for modelIdx in range(len(models)):
            x = next(iter_loaders[modelIdx])[0].to(device)
            model = models[modelIdx]
            optimizer = optimizers[modelIdx]
            losses = train_step(model, x, optimizer, beta)
            for j,l in enumerate(losses):
                train_losses[modelIdx][j]+=l.cpu().detach().numpy()*x.size()[0]/nb_train
    
    # Svaing the trained model
    for modelIdx in range(len(models)):
        torch.save(models[modelIdx].state_dict(), preTrained_saveName[modelIdx]+'.pt')


    with torch.no_grad():
        for modelIdx in range(len(models)):
            valid_loader = valid_loaders[modelIdx]
            b_valid = nb_valids[modelIdx]

            for i , (x,_) in enumerate(iter(valid_loader)):
                x = x.to(device)
                losses = compute_loss_beta(model1, x, beta)
                for j,l in enumerate(losses):
                    valid_losses[modelIdx][j]+=l.cpu().detach().numpy()*x.size()[0]/nb_valids[modelIdx]
    #torch.save(model1.state_dict(), preTrained_saveName+'.pt')
    loss_dict = {}
    for modelIdx in range(len(models)):
        loss_dict['Training, model ' + str(modelIdx)] = train_losses[modelIdx][0]
        loss_dict['Validation, model ' + str(modelIdx)] = valid_losses[modelIdx][0]
    writer.add_scalars("Full Loss",loss_dict, epoch)

    loss_dict = {}
    for modelIdx in range(len(models)):
        loss_dict['Training, model ' + str(modelIdx)] = train_losses[modelIdx][1]
        loss_dict['Validation, model ' + str(modelIdx)] = valid_losses[modelIdx][1]
    writer.add_scalars("Reconstruction Loss",loss_dict, epoch)

    loss_dict = {}
    for modelIdx in range(len(models)):
        loss_dict['Training, model ' + str(modelIdx)] = train_losses[modelIdx][2]
        loss_dict['Validation, model ' + str(modelIdx)] = valid_losses[modelIdx][2]
    writer.add_scalars("KL Divergence",loss_dict, epoch)

    print(f'epoch : {epoch}, beta  : {round(beta,2)}')
    
    for modelIdx in range(len(models)):
    
        x_test = next(iter(valid_loaders[modelIdx]))
        x_test = x_test[0]
        x_test = x_test[0:16]
        x_test = x_test.to(device)
        model = models[modelIdx]
        
        y_test = model(x_test)[0]
        
        x_grid = torchvision.utils.make_grid(x_test)
        y_grid = torchvision.utils.make_grid(y_test/torch.max(y_test))

        writer.add_image("model " + str(modelIdx) + ", input image", x_grid, epoch)
        writer.add_image("model " + str(modelIdx) + ", output image", y_grid, epoch)

    

        if (epoch+1)%20==0:
            print('Exporting sound')
            AT = AT.to('cpu')
            x_test = x_test.to('cpu')
            y_test = y_test.to('cpu').detach()
            x_test_sound = AT.inverse(mel = x_test[0])
            y_test_sound = AT.inverse(mel = y_test[0])

            x_test_sound = x_test_sound/torch.max(torch.abs(x_test_sound))
            y_test_sound = y_test_sound/torch.max(torch.abs(y_test_sound))

            writer.add_audio("model " + str(modelIdx) + ", input audio", x_test_sound, sample_rate=16000, global_step=epoch)
            writer.add_audio("model " + str(modelIdx) + ", output audio", y_test_sound, sample_rate=16000, global_step=epoch)
            print('Exported !\n')
writer.flush()
writer.close()