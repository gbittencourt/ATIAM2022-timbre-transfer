# Compute reconstruction loss, kl_loss, discriminator loss + CC loss
# Compute train step

import torch.nn

recons_criterion = torch.nn.MSELoss(reduction = 'none')
loss_disc = torch.nn.MSELoss(reduction='none')

def computeLoss_VAE_GAN_CC(VAE_GAN1, VAE_GAN2, x1, x2, beta):
    result_y2, y2, kl_div11 = VAE_GAN1(x2)
    result_x2_hat, x2_hat, kl_div12 = VAE_GAN2(y2)
    #del(y2)

    result_y1, y1, kl_div21 = VAE_GAN2(x1)
    result_x1_hat, x1_hat, kl_div22 = VAE_GAN1(y1)
    #del(y1)

    #Reconstruction Loss
    recons_loss = (recons_criterion(x1_hat,x1).mean(0).sum() + recons_criterion(x2_hat,x2).mean(0).sum())/2
    
    #Kullback-Liebler divergence
    kl_loss = (kl_div11.mean(0).sum() + kl_div12.mean(0).sum() + kl_div21.mean(0).sum() + kl_div22.mean(0).sum())/4
    
    #Discriminator Loss
    
    real_x2 = VAE_GAN1.discriminate(x2)
    real_x1 = VAE_GAN2.discriminate(x1)
    real_y2 = VAE_GAN1.discriminate(y2)   #utiles ?
    real_y1 = VAE_GAN2.discriminate(y1)   #utiles ?
    fake_x2 = result_y2
    fake_x1 = result_y1
    fake_y2 = result_x2_hat
    fake_y1 = result_x1_hat
    
    loss_discriminator = (loss_disc(real_x1, torch.ones_like(real_x1)) + loss_disc(fake_x1, torch.zeros_like(fake_x1))).sum()
    loss_discriminator += (loss_disc(real_x2, torch.ones_like(real_x2)) + loss_disc(fake_x2, torch.zeros_like(fake_x2))).sum()
    loss_discriminator += (loss_disc(real_y1, torch.ones_like(real_y1)) + loss_disc(fake_y1, torch.zeros_like(fake_y1))).sum()
    loss_discriminator += (loss_disc(real_y2, torch.ones_like(real_y2)) + loss_disc(fake_y2, torch.zeros_like(fake_y2))).sum()
    loss_discriminator /= 8
    
    if beta==0:
        full_loss = recons_loss
    else:
        full_loss = recons_loss + beta*kl_loss
    
    return full_loss, recons_loss, kl_loss, loss_discriminator

## Train step
def trainStep_VAE_GAN_CC(VAE_GAN1, VAE_GAN2, x1, x2,optimizer1, optimizer2, beta):
    # Compute the loss.
    full_loss, recons_loss, kl_loss, loss_discriminator = computeLoss_VAE_GAN_CC(VAE_GAN1, VAE_GAN2, x1, x2, beta)
    # Before the backward pass, zero all of the network gradients
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    full_loss.backward()
    # Calling the step function to update the parameters
    optimizer1.step()
    optimizer2.step()
    return full_loss, recons_loss, kl_loss, loss_discriminator