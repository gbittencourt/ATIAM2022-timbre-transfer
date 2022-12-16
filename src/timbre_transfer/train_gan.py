import torch
import torch.nn

BCELoss = torch.nn.BCELoss(reduction='none')

def computeLoss_discriminator(model, real_samples, fake_samples):

    estimate_real = model.discriminate(real_samples)
    desired_real = torch.ones_like(estimate_real)

    estimate_fake = model.discriminate(fake_samples)
    desired_fake = torch.zeros_like(estimate_fake)

    loss = BCELoss(estimate_real, desired_real) + BCELoss(estimate_fake, desired_fake)
    loss = loss.mean()

    return loss

def computeLoss_generator(model, fake_samples):

    estimate_fake = model.discriminate(fake_samples)
    desired_fake = torch.ones_like(estimate_fake)

    loss = BCELoss(estimate_fake, desired_fake)
    loss = loss.mean()
    return loss
