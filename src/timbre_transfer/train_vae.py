## Loss Function
import torch.nn

recons_criterion = torch.nn.BCELoss(reducion = 'none')

def compute_loss_beta(model, x, beta):
    x_hat, kl_div = model(x)
    recons_loss = recons_criterion(x_hat,x).mean(0).sum()
    
    kl_loss = kl_div.mean(0).sum()
    if beta==0:
        full_loss = recons_loss
    else:
        full_loss = recons_loss - beta*kl_loss
    
    return full_loss, recons_loss, -kl_loss

## Train step
def train_step_beta(model, x, optimizer, beta, device):
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

def train_step_both(model1, model2, x1, x2, optimizer1, optimizer2, beta1, beta2, device):
    losses1 = train_step_beta(model1, x1, optimizer1, beta1)
    losses2 = train_step_beta(model2, x2, optimizer2, beta2)
    return losses1, losses2