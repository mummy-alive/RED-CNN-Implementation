import os
import time
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import shutil
import cv2

class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    
def compute_num_params(model, text=False):
    num_params = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if num_params >= 1e6:
            return '{:.1f}M'.format(num_params / 1e6)
        else:
            return '{:.1f}K'.format(num_params / 1e3)
    else:
        return num_params

class checkpoint(): #생성된 model과 result를 저장하는 checkpoint
    def __init__(self, load, save=''):
        self.ok = True
        self.train_log = torch.Tensor()
        self.val_log = torch.Tensor()

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        if not save:
            save = now
        self.dir = os.path.join('experiment', save)
    
        print('experiment directory is {}'.format(self.dir))
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        os.makedirs(self.get_path('results'), exist_ok=True)

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def add_train_log(self, log):
        self.val_log = torch.cat([self.train_log, log])

    def add_val_log(self, log):
        self.val_log = torch.cat([self.val_log, log])

    def save(self, model, is_best=False):  
        if is_best:
            torch.save(model.state_dict(), self.get_path('model', 'model_best.pt'))
            torch.save(model.state_dict(), self.get_path('model', 'model_latest.pt'))
        else:
            torch.save(model.state_dict(), self.get_path('model', 'model_latest.pt'))

    def save_results(self, recon, imgname, recon_size, scale):
        save_dir = os.path.join(self.dir, 'recon'+str(max(recon_size))+'_x'+str(max(scale)))
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, imgname[0])
        np.save(filename, recon)

def calc_rmse(img1, img2):
    mse = ((img1 - img2) ** 2).mean([-2, -1])
    return torch.sqrt(mse).mean().cpu()

def calc_psnr(test, ref):
    mse = ((test - ref) ** 2).mean([-2, -1])
    return 20 * torch.log10(ref.max() / torch.sqrt(mse)).cpu()

def make_optimizer(optim_spec, target):
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': optim_spec['lr'], 'weight_decay': optim_spec['weight_decay']}

    if optim_spec['name'] == 'SGD':
        optimizer_class = optim.SGD
    elif optim_spec['name'] == 'ADAM':
        optimizer_class = optim.Adam
    elif optim_spec['name'] == 'RMSprop':
        optimizer_class = optim.RMSprop
    elif optim_spec['name'] == 'RADAM':
        optimizer_class = optim.RAdam

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')
        
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    return optimizer
