import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import models
import time
import numpy as np

# This refers to Vgg16. If required, add additional layers to implement Vgg19
class vgg_loss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg_loss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(X), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parametrs():
                param.requires_grad = False
                
    def forward(self, X):
        h = self.slice(X)
        h_relu1_2 = h   # where's ReLU?
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3