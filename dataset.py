import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
class PrepareDataset():
    # def __init__(self) -> None:
    
    def GetFashianMIST(self):
        training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
        test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(training_data, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
        self.train_loader = train_loader
        self.test_loader = test_loader

