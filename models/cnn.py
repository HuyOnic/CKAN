import torch 
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNNBlock(Module):
    def __init__(self, in_channels, out_channels, **kwargs): #kernel_size, stride, padding
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        return self.max_pool(self.ReLU(self.conv(x)))


class CNN(Module):
    def __init__(self, num_classes, in_channels = 3, **kwargs):
        super(CNN, self).__init__()
        self.conv_layer1 = CNNBlock(in_channels=in_channels, out_channels=32, **kwargs)
        self.conv_layer2 = CNNBlock(in_channels=32, out_channels=32, **kwargs)
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.ReLU()
        self.logits = nn.Softmax(dim=1)
        self.falatten = nn.Flatten()
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.falatten(x)
        x = self.activation(self.fc1(x))
        return self.logits(self.fc2(x))
    




