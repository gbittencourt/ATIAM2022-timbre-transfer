
import torch
import torch.nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.transforms import ToTensor 
from torch.utils.tensorboard import SummaryWriter
import os
writer = SummaryWriter('runs/spectralVAE_experiment_1')


from src.timbre_transfer.datasets.NSynthDataset import NSynthDataset
from src.timbre_transfer.helpers.audiotransform import AudioTransform
from src.timbre_transfer.models.Spectral_VAE import SpectralVAE


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ratio = .2

epochs = 2000

hidden_dim = 256
latent_dim = 128
base_depth = 32
n_convLayers = 5
kernel_size = 5
max_depth = 512

stride = 2
beta = .1


batch_size = 4
num_threads = 0

freqs_dim = 128
len_dim = 128

preTrained_saveName = "spectral_VAE.pt"

AT = AudioTransform(input_freq = 16000, n_fft = 1024, n_mel = 128, stretch_factor=.8)

train_dataset = NSynthDataset('data/', usage = 'train', select_class='vocal_acoustic', transform=AT)
valid_dataset = NSynthDataset('data/', usage = 'valid', select_class='vocal_acoustic', transform=AT)
nb_train = int(train_ratio * len(train_dataset))
print(nb_train)
train_dataset = torch.utils.data.dataset.random_split(train_dataset, [nb_train, len(train_dataset)-nb_train])[0]



#train_dataset = torchvision.datasets.MNIST(root='data', train = True, transform=torchvision.transforms.ToTensor())
#valid_dataset = torchvision.datasets.MNIST(root='data', train = False, transform=torchvision.transforms.ToTensor())
#freqs_dim = 28
#len_dim = 28



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_threads, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=num_threads, shuffle=False)

recons_criterion = torch.nn.MSELoss(reduction = 'none')
def compute_loss_beta(model, x, beta):
    x_hat, kl_div = model(x)
    recons_loss = recons_criterion(x_hat,x).mean(0).sum()
    
    kl_loss = kl_div.mean(0).sum()
    if beta==0:
        full_loss = recons_loss
    else:
        full_loss = recons_loss - beta*kl_loss
    
    return full_loss, recons_loss, -kl_loss

def train_step_beta(model, x, optimizer, beta):
    # Compute the loss.
    model = model.to(device)
    x = x.to(device)
    full_loss, recons_loss, kl_loss = compute_loss_beta(model, x, beta)
    # Before the backward pass, zero all of the network gradients
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    full_loss.backward()
    # Calling the step function to update the parameters
    optimizer.step()
    return full_loss, recons_loss, kl_loss


from src.timbre_transfer.models.network.Spectral_Decoder import Spectral_Decoder
from src.timbre_transfer.models.network.Spectral_Encoder import Spectral_Encoder
from src.timbre_transfer.models.Spectral_VAE import SpectralVAE




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

model = SpectralVAE(encoder, decoder, freqs_dim = freqs_dim, len_dim = len_dim, encoding_dim = hidden_dim, latent_dim = latent_dim)

if os.path.isfile(preTrained_saveName):
    model.load_state_dict(torch.load('./'+preTrained_saveName))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

x_test = next(iter(valid_loader))
x_test = x_test[0]
x_test_sound = AT.inverse(mel = x_test[0])
x_test_sound = x_test_sound/torch.max(torch.abs(x_test_sound))
x_test = x_test.to(device)




for epoch in range(epochs):
    losses_vect = [[],[],[]]
    for i, (x,_) in enumerate(iter(train_loader)):
        losses = train_step_beta(model, x, optimizer, beta)
        for j,l in enumerate(losses):
            losses_vect[j].append(l.cpu().detach())
            #print(l)
        #print('\n')
        #print(f"full loss : {full_loss}, recons_loss : {recons_loss}, kl_loss : {kl_loss*beta}")
    
    torch.save(model.state_dict(), preTrained_saveName)

    writer.add_scalar("Full_Loss/train", np.mean(losses_vect[0]), epoch)
    writer.add_scalar("Recons_Loss/train", np.mean(losses_vect[1]), epoch)
    writer.add_scalar("KL_Loss/train", np.mean(losses_vect[2]), epoch)
    print(epoch)
    model = model.to(device)
    x_test = x_test.to(device)
    

    y_test = model(x_test)[0]
    
    x_grid = torchvision.utils.make_grid(x_test)
    y_grid = torchvision.utils.make_grid(y_test/torch.max(y_test))

    writer.add_image("input_image",x_grid, epoch)
    writer.add_image("output_image",y_grid, epoch)
    
    if epoch%20==0:
        print('Exporting sound')
        AT = AT.to('cpu')
        y_test = y_test.to('cpu').detach()
        y_test_sound = AT.inverse(mel = y_test[0])

        y_test_sound = y_test_sound/torch.max(torch.abs(y_test_sound))
        writer.add_audio("input", x_test_sound, sample_rate=16000, global_step=epoch)
        writer.add_audio("output", y_test_sound, sample_rate=16000, global_step=epoch)
        print('Exported !\n')

writer.flush()
writer.close()
torch.save(model.state_dict(), 'Spectral_VAE.pt')