import os
import glob
from data import common
import numpy as np
import torch.utils.data as data
import torch
import math
import random
import pydicom
import mat73
from pathlib import Path
import time

from matplotlib import pyplot as plt

class SRData(data.Dataset):
    def __init__(self, config, mode='train', augment=False):
        self.dataset_spec = config['dataset']
        self.mode = mode
        self.augment = augment
        self.device = torch.device('cpu' if config['cpu'] else 'cuda')
        self.ldct, self.ndct = self._scan()

    def __getitem__(self, idx):
        ldct, ndct = self._load_file(idx)
        ldct, ndct = self.preparation(ldct, ndct)
        return ldct.copy(), ndct.copy()

    def __len__(self):
        if self.mode == 'train':
            return self.ldct.shape[2] * 16
        else:
            return self.ldct.shape[2]
        
    def _scan(self):
        ldct = np.array([]).reshape(512,512,0)
        ndct = np.array([]).reshape(512,512,0)
        data = np.load(os.path.join(self.dataset_spec['data_dir'], self.mode + 'set.npy'), allow_pickle=True).item()
        ldct = data['f_qd']
        ndct = data['f_nd']
        u_water = 0.0192
        # mm-1 to HU
        ldct = (ldct - u_water) * 1000 / u_water
        ndct = (ndct - u_water) * 1000 / u_water
        
        return ldct, ndct

    def _load_file(self, idx):
        idx = idx % self.ldct.shape[2]   
        ldct = self.ldct[:, :, idx].astype(np.float32)
        ndct = self.ndct[:, :, idx].astype(np.float32)
        return ldct, ndct

    def preparation(self, ldct, ndct):
        if self.mode == 'train' and self.augment:
            ldct, ndct = common.augment(ldct, ndct)
        ldct, ndct = common.get_patch(ldct, ndct)
        ldct, ndct = np.expand_dims(ldct, 0), np.expand_dims(ndct, 0)
        ## code for patch extraction & data augmentation
        # 57 line 제외하고 모든 부분은 srdata-checkpoint.py를 참고하여 추가하였음.
        # 여기서 return 해야하는것: 3차원 tensor
        return ldct, ndct
