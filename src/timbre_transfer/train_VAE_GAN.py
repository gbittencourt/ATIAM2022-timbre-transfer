import torch.nn as nn 
import torch
from src.timbre_transfer.train_gan import computeLoss_discriminator, computeLoss_generator

recons_criterion = nn.MSELoss(reduction='none')
MSELoss = nn.MSELoss(reduction='none')

def computeLoss_VAE_GAN(VAE_GAN, x):
    x_hat, kl_div = VAE_GAN(x)
    recons_loss = recons_criterion(x_hat,x).mean(0).sum()
    
    kl_loss = kl_div.mean(0).sum()
    
    discriminator_loss = computeLoss_discriminator(VAE_GAN, x, x_hat)
    generator_loss = computeLoss_generator(VAE_GAN, x_hat)
    
    return recons_loss, kl_loss, generator_loss, discriminator_loss