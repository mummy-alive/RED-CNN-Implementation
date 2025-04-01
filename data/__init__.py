from importlib import import_module
from torch.utils.data import DataLoader
import os

class Data:
      def __init__(self, config, test_only=False):
            self.loader_train = None
            if not test_only:
                  # main에서 data.Data(config)로 call함.
                  self.loader_train = None
                  module_train = import_module('data.srdata')     # 기존 모듈을 재시작 없이 재임포트
                  trainset = getattr(module_train, 'SRData')(config, mode='train', 
                                                            augment=config['dataset']['augment'])
                  self.loader_train = DataLoader(
                        trainset,
                        batch_size=config['dataset']['batch_size'],
                        shuffle=True,    # test 시 data는 고정해야함
                        num_workers=config['n_threads'] #what is this for?
                  # write your code
                  )
                  module_test = import_module('data.srdata')
                  testset = getattr(module_test, 'SRData')(config, mode='valid')
                  self.loader_test = DataLoader(
                        testset,
                        batch_size=config['dataset']['batch_size'],
                        shuffle=False,
                        num_workers=config['n_threads']
                  # write your code
                  )
            else:
                  module_test = import_module('data.srdata')
                  testset = getattr(module_test, 'SRData')(config, mode='test')
                  self.loader_test = DataLoader(
                        testset,
                        batch_size=config['dataset']['batch_size'],
                        shuffle=True,
                        num_workers=config['n_threads']
                  # write your code
                  )
      # def __len__(self, test_only=False):
      #       if not test_only:
      #             return len(self.loader_train)
      #       else:
      #             return len(self.loader_test)