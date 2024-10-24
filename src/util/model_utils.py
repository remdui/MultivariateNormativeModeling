import os
from datetime import datetime

import torch


def save_model(model, epoch):
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(
        model, f"models/vae_model_epoch_{epoch}_{datetime.now().strftime('%Y%m%d')}.pt"
    )
