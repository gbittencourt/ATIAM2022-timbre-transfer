import torch.nn as nn 
import torch

recons_criterion = nn.MSELoss(reduction='none')
loss_disc = nn.MSELoss(reduction='none')

def computeLoss(VAE_GAN, Discriminator,x, beta):
    result, x_hat, kl_div = VAE_GAN(x)
    recons_loss = recons_criterion(x_hat,x).mean(0).sum()
    
    kl_loss = kl_div.mean(0).sum()
    
    disc_real_output = Discriminator(x)
    disc_fake_output = result
    
    loss_discriminator = 1/2 * (loss_disc(disc_real_output, torch.ones_like(disc_real_output)) + loss_disc(disc_fake_output, torch.zeros_like(disc_fake_output)))

    if beta==0:
        full_loss = recons_loss + loss_discriminator
    else:
        full_loss = recons_loss +beta*kl_loss + loss_discriminator
    
    return full_loss, recons_loss, kl_loss, loss_discriminator

def trainStep_VAE_GAN(VAE_GAN, Discriminator, x, optimizer, beta):
    # Compute the loss
    full_loss, recons_loss, kl_loss, loss_discriminator = computeLoss(VAE_GAN, Discriminator, x, beta)
    # Before the backward pass, zero all of the network gradients
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    full_loss.backward()
    # Calling the step function to update the parameters
    optimizer.step()
    
    return full_loss, recons_loss, kl_loss, loss_discriminator