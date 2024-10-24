import torch
import torch.nn.functional as F


def loss_bce_kld(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
