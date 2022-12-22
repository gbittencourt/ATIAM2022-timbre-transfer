import os

import torch
import torch.nn

import torchvision

from src.timbre_transfer.datasets.NSynthDataset import NSynthDoubleDataset
from src.timbre_transfer.helpers.audiotransform import AudioTransform

from src.timbre_transfer.models.network.Spectral_Decoder import Spectral_Decoder
from src.timbre_transfer.models.network.Spectral_Encoder import Spectral_Encoder
from src.timbre_transfer.models.network.Spectral_Discriminator import Spectral_Discriminator
from src.timbre_transfer.models.VAE_GAN import SpectralVAE_GAN

from torch.utils.tensorboard import SummaryWriter

dataset_folder = os.path.join("/fast-1","atiam22-23")


preTrained_loadNames = ["pretrained/exp_2/exp_2_VAE_GAN/vocal_2", "pretrained/exp_2/exp_2_VAE_GAN/string_2"]
writer = SummaryWriter(os.path.join('runs', 'exports', '2VAEs_CC'))


## Name of the saved trained network

## Training parameters

# Learning rate
lr = 1e-4
# Dataloaders parameters
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
kernel_size = 15
# Stride of convolutionnal layers (recommended : 2 or 4)
stride = 4
# Models returns images of size freqs_dim*len_dim
freqs_dim = 128
len_dim = 128


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

AT = AudioTransform(input_freq = 16000, n_fft = 1024, n_mel = freqs_dim, stretch_factor=.8, device = device).to(device)
## Loading the NSynth dataset
valid_dataset = NSynthDoubleDataset(
    dataset_folder,
    usage = 'valid',
    filter_keys = ('vocal_acoustic', 'string_acoustic'),
    transform = AT,
    length_style = 'max',
    device = device
)

nb_valid = len(valid_dataset)

print(f"Number of validation examples : {nb_valid}")

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, num_workers=num_threads, shuffle=False)

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
print('\n')

model1 = model1.to(device)
model2 = model2.to(device)

MSE = torch.nn.MSELoss(reduction = 'none')

def norm(x):
    return x/torch.max(torch.abs(x))

def computeLoss(model1, model2, x1, x2, device):
    ## Computing the discriminator loss
    #loss_generator  = [Recons, KLDiv, Adversarial] 
    loss_generator = torch.zeros(2, device = device)

    y11, _ = model1(x1)
    loss_generator[0] += MSE(y11, x1).mean(0).sum()

    y22, _ = model2(x2)
    loss_generator[0] += MSE(y22, x2).mean(0).sum()

    y12, _ = model2(x1)

    y21, _ = model1(x2)

    y121, _ = model1(y12)
    loss_generator[1] += MSE(y121, x1).mean(0).sum()
    
    y212, _ = model1(y21)
    loss_generator[1] += MSE(y212, x2).mean(0).sum()

    return loss_generator


x1_test, x2_test = next(iter(valid_loader))
x1_test = x1_test[:800:100].to(device)
x2_test = x2_test[:800:100].to(device)

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

writer.add_image("Set 1, input image", x1_grid)
writer.add_image("Set 1, model 1 output image", y11_grid)
writer.add_image("Set 1, model 2 output image", y12_grid)

writer.add_image("Set 2, input image", x2_grid)
writer.add_image("Set 2, model 2 output image", y22_grid)
writer.add_image("Set 2, model 1 output image", y21_grid)

x1_test_sound = AT.inverse(mel = x1_test)
x2_test_sound = AT.inverse(mel = x2_test)
y12_test_sound = AT.inverse(mel = y12_test)
y22_test_sound = AT.inverse(mel = y22_test)
y11_test_sound = AT.inverse(mel = y11_test)
y21_test_sound = AT.inverse(mel = y21_test)

for i in range(8):
    writer.add_audio("Set 1, input audio", norm(x1_test_sound[i]), sample_rate=16000, global_step=i)
    writer.add_audio("Set 1, model2, output audio", norm(y12_test_sound[i]), sample_rate=16000, global_step=i)
    writer.add_audio("Set 1, model1, output audio", norm(y11_test_sound[i]), sample_rate=16000, global_step=i)

    writer.add_audio("Set 2, input audio", norm(x2_test_sound[i]), sample_rate=16000, global_step=i)
    writer.add_audio("Set 2, model1, output audio", norm(y21_test_sound[i]), sample_rate=16000, global_step=i)
    writer.add_audio("Set 2, model2, output audio", norm(y22_test_sound[i]), sample_rate=16000, global_step=i)

writer.flush()
writer.close()


MSE = torch.nn.L1Loss(reduction = 'none')

def norm(x):
    return x/torch.max(torch.abs(x))

def computeLoss(model1, model2, x1, x2, device):
    ## Computing the discriminator loss
    #loss_generator  = [Recons, KLDiv, Adversarial] 
    loss_generator = torch.zeros(4, device = device)

    y11, _ = model1(x1)
    loss_generator[0] += MSE(y11, x1).mean(0).sum()

    y22, _ = model2(x2)
    loss_generator[1] += MSE(y22, x2).mean(0).sum()

    y12, _ = model2(x1)

    y21, _ = model1(x2)

    y121, _ = model1(y12)
    loss_generator[2] += MSE(y121, x1).mean(0).sum()
    
    y212, _ = model1(y21)
    loss_generator[3] += MSE(y212, x2).mean(0).sum()

    return loss_generator

running_losses=torch.zeros(4, device = device)
for i, (x1, x2) in enumerate(iter(valid_loader)):
    
    loss = computeLoss(model1, model2, x1, x2, device)
    running_losses+=loss*x1.size()[0]/nb_valid

writer.add_scalars("Losses",{
    'Reconstruction Loss 1' : running_losses[0],
    'Reconstruction Loss 2' : running_losses[1],
    'Cycle Consistency Loss 1-2-1' : running_losses[2],
    'Cycle Consistency Loss 2-1-2' : running_losses[3],
    'Zero' : 0},
    global_step = 0)

print({
    'Reconstruction Loss 1' : running_losses[0],
    'Reconstruction Loss 2' : running_losses[1],
    'Cycle Consistency Loss 1-2-1' : running_losses[2],
    'Cycle Consistency Loss 2-1-2' : running_losses[3]
})