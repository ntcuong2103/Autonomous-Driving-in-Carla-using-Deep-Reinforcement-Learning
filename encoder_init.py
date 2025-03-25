import sys
import torch
from autoencoder.encoder import VariationalEncoder
import pytorch_lightning as pl
import os
import torchvision.transforms as transforms


class EncodeState(pl.LightningModule):
    def __init__(self, latent_dim):
        super(EncodeState, self).__init__()
        self.latent_dim = latent_dim
        self.model_file = os.path.join('autoencoder/model', 'var_encoder_model.pth')
        
        self.encoder = VariationalEncoder(self.latent_dim)
        self.encoder.load()
        self.encoder.eval()
        
        for params in self.encoder.parameters():
            params.requires_grad = False
    
    def forward(self, observation):
        image_obs = transforms.ToTensor()(observation[0])
        image_obs = image_obs.unsqueeze(0)
        image_obs = self.encoder(image_obs.to(self.device)).cpu()
        navigation_obs = torch.tensor(observation[1], dtype=torch.float)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        return observation
    