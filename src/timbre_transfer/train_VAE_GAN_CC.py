# Compute reconstruction loss, kl_loss, discriminator loss + CC loss
# Compute train step

import torch.nn
from src.timbre_transfer.train_gan import computeLoss_discriminator, computeLoss_generator

recons_criterion = torch.nn.MSELoss(reduction = 'none')

def computeLoss_CC_GAN(VAE_GAN1, VAE_GAN2, x1, x2):
    y2, kl_div11 = VAE_GAN1(x2)
    x2_hat, kl_div12 = VAE_GAN2(y2)

    y1, kl_div21 = VAE_GAN2(x1)
    x1_hat, kl_div22 = VAE_GAN1(y1)

    #Reconstruction Loss
    recons_loss = recons_criterion(x1_hat,x1).mean(0).sum() + recons_criterion(x2_hat,x2).mean(0).sum()
    
    #Kullback-Liebler divergence
    kl_loss = kl_div11.mean(0).sum() + kl_div12.mean(0).sum() + kl_div21.mean(0).sum() + kl_div22.mean(0).sum()
    
    #Discriminator Loss
    discriminator_loss = 0
    generator_loss = 0
    discriminator_loss += computeLoss_discriminator(VAE_GAN1, x1, y2)+computeLoss_discriminator(VAE_GAN2, x2, y1)
    discriminator_loss += computeLoss_discriminator(VAE_GAN1, x1, x1_hat)+computeLoss_discriminator(VAE_GAN2, x2, x2_hat)

    generator_loss += computeLoss_generator(VAE_GAN1, y2) + computeLoss_generator(VAE_GAN2, y1)
    generator_loss += computeLoss_generator(VAE_GAN1, x1_hat) + computeLoss_generator(VAE_GAN2, x2_hat)
    
    return recons_loss, kl_loss, generator_loss, discriminator_loss