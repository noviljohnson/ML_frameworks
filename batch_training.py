# data sets and dataloaders
"""
epoch = 1 forward and backward pass of ALL training samples
batch_size = number of training samples in one forward & backward pass
number of iterations = number of passes, each pass using [batch_size] number of samples
e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self):
        # data loading
        xy = np.loadtxt('./datasets/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
dataiter = iter(dataloader)
data = dataiter._next_data()
# features, labels = data
# print(features, labels)
print(data)