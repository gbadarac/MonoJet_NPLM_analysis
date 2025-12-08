import torch
from torch.utils.data import Dataset

# Define a custom dataset
class MNIST_DatasetCorrupted_VAE(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data/255.).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define a custom dataset
class MNIST_DatasetCorrupted_CL(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data/255.).float()
        self.data = torch.moveaxis(self.data, 3, 1)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def map_MNIST_to_latent_cl(model, loader,  mask_label=None, device='cpu'):
    z1_latent = []
    y_latent = []
    with torch.no_grad():
        for batch_idx, (x, l) in enumerate(loader):
            if mask_label!=None:
                mask = (l!=mask_label)
                x = x[mask]
                l = l[mask]
            x = x.to(device)
            z1 = model(x)
            z1 =z1.detach().cpu()
            z1_latent.append(z1)
            y_latent.append(l)
    y_latent = torch.cat(y_latent)
    z1_latent = torch.cat(z1_latent)
    return z1_latent.numpy(), y_latent.numpy()
