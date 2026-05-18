import torch
from torch.utils.data import Dataset

# Define a custom dataset
class CIFAR_Dataset_CL(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CIFAR_5m_Dataset_CL(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.data = torch.moveaxis(self.data, 3, 1)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]