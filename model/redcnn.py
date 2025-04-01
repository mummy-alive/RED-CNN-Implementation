import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from importlib import import_module
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
# import helper
# model 돌리기 전에 patch size 55 * 55로 조정해줘야 함.
class REDCNN(nn.Module):
    def __init__(self, out_ch=96): 
        super(REDCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True), 
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True), 
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True), 
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True), 
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True), 
                                   nn.ReLU(inplace=True))
        self.dconv1 = nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.dconv2 = nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.dconv3 = nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True)    # Residual Connection 필요
        self.dconv4 = nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.dconv5 = nn.ConvTranspose2d(in_channels=out_ch, out_channels=1, kernel_size=5, stride=1, padding=0, bias=True)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        #x = self.flatten(x)
        out = self.conv1(x)
        out = self.conv2(out)
        par1 = out
        out = self.conv3(out)
        out = self.conv4(out)
        par2 = out
        out = self.conv5(out)
        out = self.dconv1(out)
        out += par2
        out = self.ReLU(out)
        out = self.dconv2(out)
        out = self.ReLU(out) 
        out = self.dconv3(out)
        out += par1
        out = self.ReLU(out)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out += x
        #out = self.ReLU(out)
        return out