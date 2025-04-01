import torch
import utility as utility
import data
import model
from trainer import Trainer
import warnings
import vessl
import yaml
import os
import argparse

def main():
    parser = argparse.ArgumentParser()  # 스크립트 실행 명령 parse
    #parser.add_argument('--config', type=str, default='train/redcnn')
    #parser.add_argument('--save', type=str, default='redcnn')
    parser.add_argument('--config', type=str, default='train/train_example')
    parser.add_argument('--save', type=str, default='train/train_example')
    args = parser.parse_args()

    with open(os.path.join('REDCNN_Codes/configs', args.config+'.yaml').replace('\\', '/'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded, config_path: {}'.format(os.path.join('configs', args.config+'.yaml')))

    torch.manual_seed(config['seed'])       # random number 값을 실험 내내 고정시킨다.
    checkpoint = utility.checkpoint(args.save)

    vessl.configure(organization_name='mummyee', project_name='2023-12-Medisys') # Can I change the vessl?
    vessl.init(message=args.save)   #vessl에 실험 로그 저장

    if checkpoint.ok:
        loader = data.Data(config)
        t = Trainer(config, loader, checkpoint)   # trainer.p에서 Trainer 갖다 씀. 
        while not t.terminate():
            t.train()  
            t.eval()    # 여기서 train한 model_best.pt 저장되었어야 함.

if __name__ == '__main__':
    main()
