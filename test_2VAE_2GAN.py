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
from src.timbre_transfer.models.network.Spectral_Discriminator import Spectral_Discriminator
from src.timbre_transfer.models.VAE_GAN import SpectralVAE_GAN
from src.timbre_transfer.train_VAE_GAN_CC import trainStep_VAE_GAN_CC ,  computeLoss_VAE_GAN_CC
from src.timbre_transfer.train_VAE_GAN import trainStep_VAE_GAN ,  computeLoss_VAE_GAN
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

dataset_folder = "data"


preTrained_loadNames = ["pretrained/exp_2_VAE_GAN/vocal_2", "pretrained/exp_2_VAE_GAN/string_2"]
preTrained_saveName = ["pretrained/exp_2_VAE_GAN/vocal_2", "pretrained/exp_2_VAE_GAN/string_2"]
writer = SummaryWriter(os.path.join('runs','test_2VAEs_CC_2'))


## Name of the saved trained network

## Training parameters

# Number of Epochs
epochs = 30
# Learning rate
lr = 1e-4
# Reconstruction Loss (always use reduction='none')
recons_criterion = torch.nn.MSELoss(reduction = 'none')

# Beta-VAE Beta coefficient and warm up length
beta_end = 1
warm_up_length = 15 #epochs

#Lambdas [VAE & CC, Gan, Latent]
lambdas = [1,0,0]

# Dataloaders parameters
train_batch_size = 128
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
    length_style = 'min'
)

valid_dataset = NSynthDoubleDataset(
    dataset_folder,
    usage = 'valid',
    filter_keys = ('vocal_acoustic', 'string_acoustic'),
    transform = AT,
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

discriminator1 = Spectral_Discriminator(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

discriminator2 = Spectral_Discriminator(
    freqs_dim = freqs_dim,
    len_dim = len_dim,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    base_depth = base_depth,
    n_convLayers = n_convLayers,
    kernel_size = kernel_size,
    max_depth = max_depth,
    stride = stride)

model1 = SpectralVAE_GAN(encoder, decoder1, discriminator1, freqs_dim = freqs_dim, len_dim = len_dim, encoding_dim = hidden_dim, latent_dim = latent_dim)
model2 = SpectralVAE_GAN(encoder, decoder2, discriminator2, freqs_dim = freqs_dim, len_dim = len_dim, encoding_dim = hidden_dim, latent_dim = latent_dim)


## Loading pre-trained model
if os.path.isfile(preTrained_loadNames[0]+'.pt') and os.path.isfile(preTrained_loadNames[1]+'.pt'):
    model1.load_state_dict(torch.load('./'+preTrained_loadNames[0]+'.pt'))
    model2.load_state_dict(torch.load('./'+preTrained_loadNames[1]+'.pt'))

# Optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)


print('Model 1')
summary(model1, input_size=(train_batch_size, 1, 128, 128))
print('Model 2')
summary(model2, input_size=(train_batch_size, 1, 128, 128))

model1 = model1.to(device)
model2 = model2.to(device)


beta = 0

batchIdx = 0

## Training
for epoch in range(epochs):
    train_losses = {}
    valid_losses = {}

    train_losses_VAE_GAN = np.zeros(3)
    valid_losses_VAE_GAN = np.zeros(3)

    train_losses_CC = np.zeros(3)
    valid_losses_CC = np.zeros(3)
    
    if warm_up_length !=0:
        beta = min(beta_end, epoch/warm_up_length*beta_end)
    else:
        beta = beta_end

    AT = AT.to('cpu')

    for i, (x1, x2) in enumerate(iter(train_loader)):
        full_loss = 0
        x1 = x1.to(device)
        x2 = x2.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # VAEs train steps
        l = computeLoss_VAE_GAN(model1, x1, beta)
        full_loss+=l[0]
        for j in range(3):
            train_losses_VAE_GAN[j] += l[j].cpu().detach().numpy()*x1.size()[0]/nb_train

        l = computeLoss_VAE_GAN(model2, x2, beta)
        full_loss+=l[0]
        for j in range(3):
            train_losses_VAE_GAN[j] += l[j].cpu().detach().numpy()*x1.size()[0]/nb_train

        l = computeLoss_VAE_GAN_CC(model1, model2, x1, x2, beta)
        full_loss+=l[0]
        for j in range(3):
            train_losses_CC[j] += l[j].cpu().detach().numpy()*x1.size()[0]/nb_train
        
        full_loss.backward()

        optimizer1.step()
        optimizer2.step()
        
        batchIdx+=1


    # Saving the trained model
    torch.save(model1.state_dict(), preTrained_saveName[0]+'.pt')
    torch.save(model2.state_dict(), preTrained_saveName[1]+'.pt')

    with torch.no_grad():
        for i, (x1, x2) in enumerate(iter(train_loader)):
            x1 = x1.to(device)
            x2 = x2.to(device)

            l = computeLoss_VAE_GAN(model1, x1, beta)
            for j in range(3):
                valid_losses_VAE_GAN[j] += l[j].cpu().numpy()*x1.size()[0]/nb_valid

            l = computeLoss_VAE_GAN(model2, x2, beta)
            for j in range(3):
                valid_losses_VAE_GAN[j] += l[j].cpu().numpy()*x1.size()[0]/nb_valid
            
            l = computeLoss_VAE_GAN_CC(model1, model2, x1, x2, beta)
            for j in range(3):
                valid_losses_CC[j] += l[j].cpu().numpy()*x1.size()[0]/nb_valid



    #torch.save(model1.state_dict(), preTrained_saveName+'.pt')
    writer.add_scalars("Overview",{
        'Training, VAE' : train_losses_VAE_GAN[0],
        'Validation, VAE' : valid_losses_VAE_GAN[0],
        'Training, CC' : train_losses_CC[0],
        'Validation, CC' : valid_losses_CC[0]
    }, epoch)

    writer.add_scalars("VAE",{
        'Training, Reconstruction' : train_losses_VAE_GAN[1],
        'Validation, Reconstruction' : valid_losses_VAE_GAN[1],
        'Training, Kullback-Liebler Divergence' : train_losses_VAE_GAN[2],
        'Validation, Kullback-Liebler Divergence' : valid_losses_VAE_GAN[2]
    }, epoch)

    writer.add_scalars("Cycle Consistency",{
        'Training, Reconstruction' : train_losses_CC[1],
        'Validation, Reconstruction' : valid_losses_CC[1],
        'Training, Kullback-Liebler Divergence' : train_losses_CC[2],
        'Validation, Kullback-Liebler Divergence' : valid_losses_CC[2]
    }, epoch)

    print(f'epoch : {epoch}, beta  : {round(beta,2)}')
    
    x1_test, x2_test = next(iter(valid_loader))
    x1_test = x1_test[:16].to(device)
    x2_test = x2_test[:16].to(device)

    y11_test = model1(x1_test)[0].detach()
    y12_test = model2(x1_test)[0].detach()
    y22_test = model2(x2_test)[0].detach()
    y21_test = model1(x2_test)[0].detach()

    x1_grid = torchvision.utils.make_grid(x1_test)
    x2_grid = torchvision.utils.make_grid(x2_test)

    y11_grid = torchvision.utils.make_grid(y11_test)
    y12_grid = torchvision.utils.make_grid(y12_test)
    
    y22_grid = torchvision.utils.make_grid(y22_test)
    y21_grid = torchvision.utils.make_grid(y21_test)

    writer.add_image("Set 1, input image", x1_grid, epoch)
    writer.add_image("Set 1, model 1 output image", y11_grid, epoch)
    writer.add_image("Set 1, model 2 output image", y12_grid, epoch)

    writer.add_image("Set 2, input image", x2_grid, epoch)
    writer.add_image("Set 2, model 2 output image", y22_grid, epoch)
    writer.add_image("Set 2, model 1 output image", y21_grid, epoch)

    if (epoch+1)%5==0:
        print('Exporting sound')
        x1_test = x1_test.to('cpu')
        x2_test = x2_test.to('cpu')

        y11_test = y11_test.to('cpu')
        y12_test = y12_test.to('cpu')
        y22_test = y22_test.to('cpu')
        y21_test = y21_test.to('cpu')

        x1_test_sound = AT.inverse(mel = x1_test[0]*11.57)
        x2_test_sound = AT.inverse(mel = x2_test[0]*11.57)
        y11_test_sound = AT.inverse(mel = y11_test[0]*11.57)
        y12_test_sound = AT.inverse(mel = y12_test[0]*11.57)
        y22_test_sound = AT.inverse(mel = y22_test[0]*11.57)
        y21_test_sound = AT.inverse(mel = y21_test[0]*11.57)

        writer.add_audio("Set 1, input audio", x1_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 1, model1, output audio", y11_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 1, model2, output audio", y12_test_sound, sample_rate=16000, global_step=epoch)

        writer.add_audio("Set 2, input audio", x2_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 2, model1, output audio", y21_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 2, model2, output audio", y22_test_sound, sample_rate=16000, global_step=epoch)
        print('Exported !\n')
writer.flush()
writer.close()