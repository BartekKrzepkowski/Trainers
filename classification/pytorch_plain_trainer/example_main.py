import argparse
import os
import configparser

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from trainer import Trainer, IteratorParams
from models import AllCNN
from loaders import loaders_example


dir_name = os.path.dirname(__file__)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", help="config file", default='config.ini')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(os.path.join(dir_name, "config.ini"))


def main(config):
    params_trainer = {
        'model': AllCNN,
        'loaders': loaders_example,
        'criterion': torch.nn.CrossEntropyLoss,
        'optim': torch.optim.SGD,
        'scheduler': torch.optim.lr_scheduler.ExponentialLR,
        #     'params_clearml': params_clearml,
        'is_tensorboard': config['default']['is_tensorboard']
    }
    trainer = Trainer(**params_trainer)

    DATASET_NAME = 'cifar10'
    model_ls = [{}]
    loaders_ls = [{'batch_size': 128, 'dataset_name': DATASET_NAME}]
    criterion_ls = [{}]
    optim_ls = [{'lr': 0.05, 'weight_decay': 0.001}]
    scheduler_ls = [{'gamma': 0.97}]
    iter_params = IteratorParams(model_ls, loaders_ls, criterion_ls, optim_ls, scheduler_ls)

    params_runs = {
        'iter_params': iter_params,
        'epochs': int(config['default']['epochs']),
        'exp_name': config['default']['exp_name'],
        'val_step': int(config['default']['val_step']),
        'verbose': config['default']['verbose'] == 'True',
        'checkpoint_save_step': int(config['default']['checkpoint_save_step']),
        'device': device
    }
    trainer.run_trainer(**params_runs)


if __name__ == "__main__":
    main(config)

