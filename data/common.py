import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

def get_patch(ldct, ndct, patch_size=55): #input: 임의의 사진 하나 / output: patch들이 되도록 코드 짜기
    ## np. rand() size 맞게 random 수 뽑아서 거기서부터 patch 뽑으면 됨.
    start_row = random.randint(0, 512-patch_size)
    start_col = random.randint(0, 512-patch_size);
    
    ldct = ldct[start_row : start_row+patch_size, start_col : start_col+patch_size]
    ndct = ndct[start_row : start_row+patch_size, start_col : start_col+patch_size]
    return ldct, ndct

def augment(ldct, ndct, hflip=True, vflip=True, rot=True):  
    ldct = torch.tensor(ldct)
    ndct = torch.tensor(ndct)  
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot * random.randint(0,4)
    # # write your code
    # 논문에는 rotate와 scaling이 있음.
    if hflip:
        ldct = torch.flip(ldct, [0])
        ndct = torch.flip(ndct, [0])
        
    if vflip:
        ldct = torch.flip(ldct, [1])
        ndct = torch.flip(ndct, [1])
    if rot90:
        ldct = torch.rot90(ldct, rot90, [0, 1])
        ndct = torch.rot90(ndct, rot90, [0, 1])   
    ldct = ldct.numpy()
    ndct = ndct.numpy()
    return ldct, ndct
