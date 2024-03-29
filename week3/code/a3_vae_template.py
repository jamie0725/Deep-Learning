import argparse
import os

import torch
import numpy as np
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from datasets.bmnist import bmnist
from scipy.stats import norm


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cuda:0'):
        super().__init__()
        self.i2h = nn.Linear(784, hidden_dim)
        self.h2z1 = nn.Linear(hidden_dim, z_dim)
        self.h2z2 = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        out = self.i2h(input)
        out =  self.relu(out)
        mean = self.h2z1(out)
        logvar = self.h2z2(out)
        
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cuda:0'):
        super().__init__()
        self.z2h = nn.Linear(z_dim, hidden_dim)
        self.h2i = nn.Linear(hidden_dim, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        out = self.z2h(input)
        out = self.relu(out)
        mean = self.h2i(out)
        mean = self.sigmoid(mean)
        
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cuda:0'):
        super().__init__()

        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(hidden_dim, z_dim, device).to(device)
        self.decoder = Decoder(hidden_dim, z_dim, device).to(device)
        self.to(device)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, logvar = self.encoder(input)
        epsilon = torch.randn(mean.shape).to(self.device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        rc_input = self.decoder(z) # Here we follow the original paper to use the means as the recon input without sampling from the Bernoulli distribution
        rc_loss = torch.nn.functional.binary_cross_entropy(rc_input, input, reduction='sum')
        r_loss = 0.5 * torch.sum(logvar.exp() + mean.pow(2) - 1 - logvar)
        average_negative_elbo = (rc_loss + r_loss) / input.shape[0]
        
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_zs = torch.randn(n_samples, self.z_dim).to(self.device)
        means = self.decoder(sampled_zs)
        sampled_ims = torch.bernoulli(means)
        im_means = means

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0.
    if model.training:
        for i, imgs in enumerate(data):
            imgs = imgs.view(-1, 784).to(device)
            optimizer.zero_grad()
            loss = model(imgs)
            loss.backward()
            optimizer.step()
            average_epoch_elbo += loss.item()
    else:
        for i, imgs in enumerate(data):
            imgs = imgs.view(-1, 784).to(device)
            loss = model(imgs)
            average_epoch_elbo += loss.item()
    average_epoch_elbo /= len(data)
    
    return average_epoch_elbo


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Print all configs to confirm parameter settings
    print_flags()
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Create output directories
    os.makedirs('./images/vae', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer, device)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if ARGS.zdim != 2:
            im_samples, means_samples = model.sample(25)
            save_image(im_samples.view(im_samples.shape[0], 1, 28, 28),
                       './images/vae/bernoulli_{}.png'.format(epoch),
                       nrow=5, normalize=True)
            save_image(means_samples.view(means_samples.shape[0], 1, 28, 28),
                       './images/vae/mean_{}.png'.format(epoch),
                       nrow=5, normalize=True)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        model.eval()
        x_values = norm.ppf(np.linspace(0.0001, 0.9999, 15))
        y_values = norm.ppf(np.linspace(0.0001, 0.9999, 15))
        manifold = torch.FloatTensor(np.array(np.meshgrid(x_values, y_values)).T).view(-1, 2).to(device)
        with torch.no_grad():
            imgs = model.decoder(manifold)
        save_image(imgs.view(imgs.shape[0], 1, 28, 28),
                   './images/vae/manifold.png'.format(epoch),
                   nrow=15, normalize=True)

    save_elbo_plot(train_curve, val_curve, './images/vae/elbo_zdim-{}.png'.format(ARGS.zdim))
    save_elbo_plot(train_curve, val_curve, './images/vae/elbo_zdim-{}.eps'.format(ARGS.zdim))

def print_flags():
  """
  Prints all entries in args variable.
  """
  for key, value in vars(ARGS).items():
    print(key + ' : ' + str(value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
