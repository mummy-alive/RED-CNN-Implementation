import os
import utility as utility
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import vessl

import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from model import redcnn
#from vgg_loss import vgg_loss   #not an existing library!! Gotta create on by myself
class Trainer():
    def __init__(self, config, loader, ckp):
        self.config = config
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        
        ## Customize your loss function
        #self.vgg_loss = vgg_loss.fooConstructor()
        #self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()
        self.model = redcnn.REDCNN().to('cuda')
        self.optimizer = utility.make_optimizer(config['optimizer'], self.model)
        
        self.scheduler = MultiStepLR(self.optimizer, 
                                    milestones=config['optimizer']['milestones'], # Call Adam optimizer
                                    gamma=config['optimizer']['gamma'])
        print('total number of parameter is {}'.format(sum(p.numel() for p in self.model.parameters())))

    def train(self):
        epoch = self.scheduler.last_epoch
        self.ckp.add_train_log(torch.zeros(1))  # 
        learning_rate = self.scheduler.get_last_lr()[0]
        self.model.train()
        train_loss = utility.Averager()
        timer = utility.Timer()
        # itnum = 1
        for batch, (ldct, ndct) in enumerate(self.loader_train):
            #### write training code. 
            ldct, ndct = self.prepare(ldct, ndct) 
            #이 for문안의 코드- 여기서부터는 내가 작성함.
            self.optimizer.zero_grad()  # batch마다 optimizer init
            outputs = self.model(ldct)
            loss = self.criterion(outputs, ndct) # operate loss function(L1Loss)
            loss.backward()   # backpropagation. Loss function의 gradient 값을 .grad에 저장
            self.optimizer.step()
            #self.model.zero_grad() # do we really need this?
            train_loss.add(loss.item(), ldct.size(0)) # loss 결과 저장해줌. 
        vessl.log(step=epoch, payload={'train_loss': train_loss.item(), 'train_time': timer.t(), 'learning_rate': learning_rate})
        self.scheduler.step()

    def eval(self):     # Evaluate the accuracy of model
        epoch = self.scheduler.last_epoch
        mn = 2e18
        apath = self.config['model']['apath']
        if epoch % self.config['test_every'] == 0:
            self.ckp.add_val_log(torch.zeros(1))
            self.model.eval()
            timer = utility.Timer()
            with torch.no_grad():
                loss = 0
                for i, (ldct, ndct) in enumerate(self.loader_test):
                    # evaluate validation image quality
                    ldct, ndct = self.prepare(ldct, ndct)
                    outputs = self.model(ldct)
                    mse_loss = self.criterion(outputs, ndct)  
                    loss += mse_loss.item()
                # print(f'shape: {len(self.loader_test[0])}, len: {len(self.loader_test)}, loss:{loss}')                
                loss /= len(self.loader_test)
                print(f'Validation MSE Loss(CT num):{loss} | MSE(mu):{loss*0.0192/1000} || RMSE(mu):{np.sqrt(loss*0.0192/1000)}') 
                save_dirs = [os.path.join(apath, 'model_latest.pt')]
                if loss < mn:
                    mn = loss
                    save_dirs.append(os.path.join(apath, 'model_best.pt'))
                #save_dirs.append(oss.path.join(apath, f'model_{epoch}.pt'))
                for i in save_dirs:
                    torch.save(self.model.state_dict(), i)
		### if trained model was the best so far: update both model_best.pt & model_latest.pt 
		### if it is not, just update model_latest.pt

    def prepare(self, *args):
        device = torch.device('cpu' if self.config['cpu'] else 'cuda')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        epoch = self.scheduler.last_epoch
        return epoch >= self.config['epochs']
