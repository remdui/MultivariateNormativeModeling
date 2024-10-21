import os
import torch
from datetime import datetime

def save_model(model, epoch):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model, f"models/vae_model_epoch_{epoch}_{datetime.now().strftime('%Y%m%d')}.pt")
