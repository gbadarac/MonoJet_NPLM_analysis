import torch
import torch.nn as nn


def map_to_latent_VAE(model, loader, input_dim, mask_label=None, device='cpu'):
    z1_latent = []
    z2_latent = []
    y_latent = []
    with torch.no_grad():
        for batch_idx, (x, l) in enumerate(loader):
            x = x.view(-1, input_dim)
            if mask_label!=None:
                mask = (l!=mask_label)
                x = x[mask]
                l = l[mask]
            x = x.to(device)
            z1,z2 = model.encode(x)
            z1,z2 =z1.detach().cpu(),z2.detach().cpu()
            z1_latent.append(z1)
            z2_latent.append(z2)
            y_latent.append(l)
    y_latent = torch.cat(y_latent)
    z1_latent = torch.cat(z1_latent)
    z2_latent = torch.cat(z2_latent)
    z_latent = torch.column_stack([z1_latent, z2_latent])
    return z_latent.numpy(), y_latent.numpy()

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256, device='cpu'):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD