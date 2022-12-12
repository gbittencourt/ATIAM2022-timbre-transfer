## Cycle Consistency Loss Function and train_step
import torch.nn

recons_criterion = torch.nn.MSELoss(reduction = 'none')

def computeLoss_CC(VAE1, VAE2, x1, x2, beta):
    y2, kl_div11 = VAE1(x2)
    x2_hat, kl_div12 = VAE2(y2)
    del(y2)

    y1, kl_div21 = VAE2(x1)
    x1_hat, kl_div22 = VAE1(y1)
    del(y1)

    #Reconstruction Loss
    recons_loss = (recons_criterion(x1_hat,x1).mean(0).sum() + recons_criterion(x2_hat,x2).mean(0).sum())/2
    
    #Kullback-Liebler divergence
    kl_loss = (kl_div11.mean(0).sum() + kl_div12.mean(0).sum() + kl_div21.mean(0).sum() + kl_div22.mean(0).sum())/4
    if beta==0:
        full_loss = recons_loss
    else:
        full_loss = recons_loss + beta*kl_loss
    
    return full_loss, recons_loss, kl_loss

## Train step
def trainStep_CC(VAE1, VAE2, x1, x2,optimizer1, optimizer2, beta):
    # Compute the loss.
    full_loss, recons_loss, kl_loss = computeLoss_CC(VAE1, VAE2, x1, x2, beta)
    # Before the backward pass, zero all of the network gradients
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    full_loss.backward()
    # Calling the step function to update the parameters
    optimizer1.step()
    optimizer2.step()
    return full_loss, recons_loss, kl_loss
