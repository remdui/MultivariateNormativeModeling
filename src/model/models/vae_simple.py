import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, covariate_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + covariate_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim + covariate_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, covariates):
        x = torch.cat([x, covariates], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, covariates):
        z = torch.cat([z, covariates], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, covariates):
        mu, logvar = self.encode(x, covariates)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, covariates), mu, logvar