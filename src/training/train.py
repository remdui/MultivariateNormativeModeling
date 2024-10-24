from torch import optim

from entities.properties import Properties
from model.models.vae_simple import VAE
from model.loss_functions.bce_kl_loss import loss_bce_kld
from preprocessing.dataloader import FreeSurferDataloader
from util.log_utils import log_message
from util.model_utils import save_model


def train_vae():

    properties = Properties.get_instance()

    # call train_vae with the necessary parameters
    epochs = properties.train.epochs
    latent_dim = properties.model.latent_dim
    hidden_dim = properties.model.hidden_dim
    lr = properties.train.learning_rate
    device = properties.system.device

    dataloader = FreeSurferDataloader.init_dataloader()
    input_dim = dataloader.dataset[0][0].shape[0]
    covariate_dim = dataloader.dataset[0][1].shape[0]
    model = VAE(input_dim, hidden_dim[0], latent_dim, covariate_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for _, (data, covariates) in enumerate(dataloader):
            data = data.float().to(device)  # Ensure data is of type float
            covariates = covariates.float().to(
                device
            )  # Ensure covariates are of type float
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, covariates)
            loss = loss_bce_kld(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}")
        log_message(f"Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}")
        save_model(model, epoch)


def main():
    print("Training model")


if __name__ == "__main__":
    main()
