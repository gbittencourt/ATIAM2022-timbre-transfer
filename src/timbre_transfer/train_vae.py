## Loss Function
import torch.nn

recons_criterion = torch.nn.MSELoss(reduction = 'none')

def computeLoss_VAE(model, x, beta):
    x_hat, kl_div = model(x)
    recons_loss = recons_criterion(x_hat,x).mean(0).sum()
    
    kl_loss = kl_div.mean(0).sum()

    if beta==0:
        full_loss = recons_loss
    else:
        full_loss = recons_loss +beta*kl_loss
    
    return full_loss, recons_loss, kl_loss

## Train step
def trainStep_VAE(model, x, optimizer, beta):
    # Compute the loss.
    full_loss, recons_loss, kl_loss = computeLoss_VAE(model, x, beta)
    # Before the backward pass, zero all of the network gradients
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    full_loss.backward()
    # Calling the step function to update the parameters
    optimizer.step()
    return full_loss, recons_loss, kl_loss