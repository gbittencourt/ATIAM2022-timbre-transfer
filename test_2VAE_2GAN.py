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
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

dataset_folder = "data"


preTrained_loadNames = ["pretrained/exp_2_VAE_GAN/vocal_1", "pretrained/exp_2_VAE_GAN/string_1"]
preTrained_saveName = ["pretrained/exp_2_VAE_GAN/vocal_1", "pretrained/exp_2_VAE_GAN/string_1"]
writer = SummaryWriter(os.path.join('runs','test_2VAE_CC_GAN_1'))


## Name of the saved trained network

## Training parameters

# Number of Epochs
epochs = 30
# Learning rate
lr = 1e-4
# Reconstruction Loss (always use reduction='none')
recons_criterion = torch.nn.MSELoss(reduction = 'none')

# Beta-VAE Beta coefficient and warm up length
beta_end = .5
warm_up_length = 0 #epochs

#Lambdas [VAE & CC, Gan, Latent]
lambdas = [1,50]

# Dataloaders parameters
train_batch_size = 256
valid_batch_size = 1024
num_threads = 0


## Model Parameters
# Dimension of the linear layer
hidden_dim = 256
# Dimension of the latent space
latent_dim = 16
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
param_gen = list(encoder.parameters())+list(decoder1.parameters())+list(decoder2.parameters())
optimizer_gen = torch.optim.Adam(param_gen, lr=lr)

optimizer_dis_1 = torch.optim.Adam(discriminator1.parameters(), lr = lr)
optimizer_dis_2 = torch.optim.Adam(discriminator2.parameters(), lr = lr)

print('VAE')
summary(model1, input_size=(train_batch_size, 1, 128, 128))
#print('Model 2')
#summary(model2, input_size=(train_batch_size, 1, 128, 128))

print('Encoder')
summary(encoder)
print('Decoder')
print(decoder1)
print('Discriminator')
summary(discriminator1)

print('\n')

model1 = model1.to(device)
model2 = model2.to(device)

MSE = torch.nn.MSELoss(reduction = 'none')

def computeLoss_discriminator(model, real_samples, fake_samples):

    estimate_real = model.discriminate(real_samples)
    desired_real = torch.ones_like(estimate_real)

    estimate_fake = model.discriminate(fake_samples)
    desired_fake = torch.zeros_like(estimate_fake)

    loss = MSE(estimate_real, desired_real) + MSE(estimate_fake, desired_fake)
    loss = loss.mean()

    return loss

def computeLoss_generator(model, fake_samples):

    estimate_fake = model.discriminate(fake_samples)
    desired_fake = torch.ones_like(estimate_fake)

    loss = MSE(estimate_fake, desired_fake)
    loss = loss.mean()
    return loss

def trainStep(model1, model2, optimizer_gen, optimizer_dis_1, optimizer_dis_2, x1, x2, beta, lambdas, device):

    ## Computing the discriminator loss
    loss_discriminator = 0

    y11, kldiv11 = model1(x1)
    if lambdas[1] !=0:
        loss_discriminator += computeLoss_discriminator(model1, x1, y11)

    y22, kldiv22 = model2(x2)
    if lambdas[1] !=0:
        loss_discriminator += computeLoss_discriminator(model2, x2, y22)

    y12, kldiv12 = model2(x1)
    if lambdas[1] !=0:
        loss_discriminator += computeLoss_discriminator(model2, x2, y12)

    y21, kldiv21 = model1(x2)
    if lambdas[1] !=0:
        loss_discriminator += computeLoss_discriminator(model1, x1, y21)

    y121, kldiv121 = model1(y12)
    if lambdas[1] !=0:
        loss_discriminator += computeLoss_discriminator(model1, x1, y121)
    
    y212, kldiv212 = model1(y21)
    if lambdas[1] !=0:
        loss_discriminator += computeLoss_discriminator(model2, x2, y212)

    lossFull_discriminator = lambdas[1]*loss_discriminator

    optimizer_dis_1.zero_grad()
    optimizer_dis_2.zero_grad()
    lossFull_discriminator.backward()
    optimizer_dis_1.step()
    optimizer_dis_2.step()

    ## Computing the generator loss
    #loss_generator  = [Recons, KLDiv, Adversarial] 
    loss_generator = torch.zeros(3, device=device)

    y11, kldiv11 = model1(x1)
    loss_generator[0] += MSE(y11, x1).mean(0).sum()
    loss_generator[1] += kldiv11.mean(0).sum()
    if lambdas[1] !=0:
        loss_generator[2] += computeLoss_generator(model1, y11)

    y22, kldiv22 = model2(x2)
    loss_generator[0] += MSE(y22, x2).mean(0).sum()
    loss_generator[1] += kldiv22.mean(0).sum()
    if lambdas[1] !=0:
        loss_generator[2] += computeLoss_generator(model2, y22)

    y12, kldiv12 = model2(x1)
    loss_generator[1] += kldiv12.mean(0).sum()
    if lambdas[1] !=0:
        loss_generator[2] += computeLoss_generator(model2, y12)

    y21, kldiv21 = model1(x2)
    loss_generator[1] += kldiv21.mean(0).sum()
    if lambdas[1] !=0:
        loss_generator[2] += computeLoss_generator(model1, y21)

    y121, kldiv121 = model1(y12)
    loss_generator[0] += MSE(y121, x1).mean(0).sum()
    loss_generator[1] += kldiv121.mean(0).sum()
    if lambdas[1] !=0:
        loss_generator[2] += computeLoss_generator(model1, y121)
    
    y212, kldiv212 = model1(y21)
    loss_generator[0] += MSE(y212, x2).mean(0).sum()
    loss_generator[1] += kldiv212.mean(0).sum()
    if lambdas[1] !=0:
        loss_generator[2] += computeLoss_generator(model2, y212)
    
    lossFull_generator = lambdas[0]*loss_generator[0] + beta*loss_generator[1] + lambdas[1]*loss_generator[2]
    optimizer_gen.zero_grad()
    lossFull_generator.backward()
    optimizer_gen.step()

    return loss_generator, loss_discriminator

def computeLoss(model1, model2, x1, x2, device):
    ## Computing the discriminator loss
    loss_discriminator = 0
    #loss_generator  = [Recons, KLDiv, Adversarial] 
    loss_generator = torch.zeros(3, device = device)

    y11, kldiv11 = model1(x1)
    loss_discriminator += computeLoss_discriminator(model1, x1, y11)
    loss_generator[0] += MSE(y11, x1).mean(0).sum()
    loss_generator[1] += kldiv11.mean(0).sum()
    loss_generator[2] += computeLoss_generator(model1, y11)

    y22, kldiv22 = model2(x2)
    loss_discriminator += computeLoss_discriminator(model2, x2, y22)
    loss_generator[0] += MSE(y22, x2).mean(0).sum()
    loss_generator[1] += kldiv22.mean(0).sum()
    loss_generator[2] += computeLoss_generator(model2, y22)

    y12, kldiv12 = model2(x1)
    loss_discriminator += computeLoss_discriminator(model2, x2, y12)
    loss_generator[1] += kldiv12.mean(0).sum()
    loss_generator[2] += computeLoss_generator(model2, y12)

    y21, kldiv21 = model1(x2)
    loss_discriminator += computeLoss_discriminator(model1, x1, y21)
    loss_generator[1] += kldiv21.mean(0).sum()
    loss_generator[2] += computeLoss_generator(model1, y21)

    y121, kldiv121 = model1(y12)
    loss_discriminator += computeLoss_discriminator(model1, x1, y121)
    loss_generator[0] += MSE(y121, x1).mean(0).sum()
    loss_generator[1] += kldiv121.mean(0).sum()
    loss_generator[2] += computeLoss_generator(model1, y121)
    
    y212, kldiv212 = model1(y21)
    loss_discriminator += computeLoss_discriminator(model2, x2, y212)
    loss_generator[0] += MSE(y212, x2).mean(0).sum()
    loss_generator[1] += kldiv212.mean(0).sum()
    loss_generator[2] += computeLoss_generator(model2, y212)

    return loss_generator, loss_discriminator


beta = 0

batchIdx = 0
## Training
for epoch in range(epochs):
    
    train_losses = {}
    valid_losses = {}

    train_runningLosses = np.zeros(4)
    valid_runningLosses = np.zeros(4)
    
    if warm_up_length !=0:
        beta = min(beta_end, epoch/warm_up_length*beta_end)
    else:
        beta = beta_end

    AT = AT.to('cpu')

    for i, (x1, x2) in enumerate(iter(train_loader)):
        losses_plot = torch.zeros(4).to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)

        
        loss_gen, loss_dis = trainStep(
            model1 = model1, model2 = model2,
            optimizer_gen = optimizer_gen,
            optimizer_dis_1=optimizer_dis_1,
            optimizer_dis_2=optimizer_dis_2,
            x1=x1,
            x2=x2,
            beta=beta,
            lambdas=lambdas,
            device=device)
        
        train_runningLosses[0]+=loss_gen[0]*lambdas[0]*x1.size()[0]/nb_train
        train_runningLosses[1]+=loss_gen[1]*beta*x1.size()[0]/nb_train
        train_runningLosses[2]+=loss_gen[2]*lambdas[1]*x1.size()[0]/nb_train
        train_runningLosses[3]+=loss_dis*lambdas[1]*x1.size()[0]/nb_train
        

        #writer.add_scalars("Batch losses",
        #{
        #    'Reconstruction' : loss_gen[0]*lambdas[0],
        #    'KLDiv' : loss_gen[1]*beta,
        #    "Generator" : loss_gen[2]*lambdas[1],
        #    "Discriminator" : loss_dis*lambdas[1]
        #}, batchIdx)
        batchIdx+=1


    # Saving the trained model
    torch.save(model1.state_dict(), preTrained_saveName[0]+'.pt')
    torch.save(model2.state_dict(), preTrained_saveName[1]+'.pt')

    with torch.no_grad():
        for i, (x1, x2) in enumerate(iter(valid_loader)):
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            loss_gen, loss_dis = computeLoss(model1, model2, x1, x2, device=device)

            valid_runningLosses[0]+=loss_gen[0]*lambdas[0]*x1.size()[0]/nb_valid
            valid_runningLosses[1]+=loss_gen[1]*beta*x1.size()[0]/nb_valid
            valid_runningLosses[2]+=loss_gen[2]*lambdas[1]*x1.size()[0]/nb_valid
            valid_runningLosses[3]+=loss_dis*lambdas[1]*x1.size()[0]/nb_valid


    writer.add_scalars("VAE",{
        'Training_Reconstruction' : train_runningLosses[0],
        'Validation_Reconstruction' : valid_runningLosses[0],
        'Training_KLDiv' : train_runningLosses[1],
        'Validation_KLDiv' : valid_runningLosses[1]
    }, epoch)

    writer.add_scalars("Adversarial", {
        "Training_Generator" : train_runningLosses[2],
        "Training_Discriminator" : train_runningLosses[3],
        "Validation_Generator" : valid_runningLosses[2],
        "Validation_Discriminator" : valid_runningLosses[3]
    }, epoch)


    print(f'epoch : {epoch}, beta  : {round(beta,2)}')
    
    x1_test, x2_test = next(iter(valid_loader))
    x1_test = x1_test[:8].to(device)
    x2_test = x2_test[:8].to(device)

    zeros = torch.zeros_like(x1_test)

    y11_test = model1(x1_test)[0].detach()
    y12_test = model2(x1_test)[0].detach()
    y22_test = model2(x2_test)[0].detach()
    y21_test = model1(x2_test)[0].detach()

    y11_test = torch.where(y11_test>0, y11_test, zeros)        
    y12_test = torch.where(y12_test>0, y12_test, zeros)    
    y22_test = torch.where(y22_test>0, y22_test, zeros)    
    y21_test = torch.where(y21_test>0, y21_test, zeros)        

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

        x1_test_sound = AT.inverse(mel = x1_test[0])
        x2_test_sound = AT.inverse(mel = x2_test[0])
        y12_test_sound = AT.inverse(mel = y12_test[0])
        y22_test_sound = AT.inverse(mel = y22_test[0])
        y11_test_sound = AT.inverse(mel = y11_test[0])
        y21_test_sound = AT.inverse(mel = y21_test[0])

        writer.add_audio("Set 1, input audio", x1_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 1, model1, output audio", y11_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 1, model2, output audio", y12_test_sound, sample_rate=16000, global_step=epoch)

        writer.add_audio("Set 2, input audio", x2_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 2, model1, output audio", y21_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("Set 2, model2, output audio", y22_test_sound, sample_rate=16000, global_step=epoch)
        print('Exported !\n')
writer.flush()
writer.close()