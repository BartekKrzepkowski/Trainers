get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
get_ipython().run_line_magic("load_ext", " tensorboard")


get_ipython().run_line_magic("matplotlib", " inline")

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from trainer import Trainer, IteratorParams
from models import AllCNN
from loaders import loaders_example

params_clearml = {

}

DATASET_NAME = 'cifar10'
params_trainer = {
    'model': AllCNN,
    'loaders': loaders_example,
    'criterion': torch.nn.CrossEntropyLoss,
    'optim': torch.optim.SGD,
    'scheduler': torch.optim.lr_scheduler.ExponentialLR,
    'params_clearml': params_clearml,
    'is_tensorboard': True
}

trainer = Trainer(**params_trainer)


get_ipython().run_line_magic("tensorboard", " --logdir=data")


model_ls = [{}]
loaders_ls = [{'batch_size':128, 'dataset_name': 'cifar10'}]
criterion_ls = [{}]
optim_ls = [{'lr': 0.05, 'weight_decay': 0.001}]
scheduler_ls = [{'gamma':0.97}]

iter_params = IteratorParams(model_ls, loaders_ls, criterion_ls, optim_ls, scheduler_ls)

params_runs = {
    'iter_params': iter_params,
    'epochs': 3,
    'exp_name': 'cifar_without_deficit',
    'val_step': 35,
    'verbose': False,
    'checkpoint_save_step': 2, 
    'device': device
}

trainer.run_trainer(**params_runs)


device



