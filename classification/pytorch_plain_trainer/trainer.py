import copy
import datetime
import gc
import os
from collections import ChainMap
from itertools import product

import torch
import pandas as pd
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_model


class Trainer:
    def __init__(self, model, loaders, criterion, optim, scheduler=None, params_clearml=None, is_tensorboard=False):
        """
        Trainer initializer. Every argument is a shell of class or method that is initialized in init_run.

        Args:
            model (Net): Custom neural network
            loaders (dict[str, torch.utils.data.DataLoader]): Dict of train, test loaders
            criterion (torch.nn.NLLLoss): The negative log likelihood loss
            optim (torch.optim.Optimizer): Chosen optimizer
        """
        self.model_ = model
        self.loaders_ = loaders
        self.criterion_ = criterion
        self.optim_ = optim
        self.scheduler_ = scheduler
        self.params_clearml = params_clearml
        self.is_tensorboard = is_tensorboard
        if params_clearml:
            Task.set_credentials(**params_clearml)

    def run_trainer(self, iter_params, epochs, exp_name, val_step, checkpoint_save_step, verbose=False, device=None):
        """
        Main method of trainer.
        Init df -> [Pick run  -> Init Run -> [Run Epoch]_{IL} -> Update df]_{IL} -> Save df -> Plot Results
        {IL - In Loop}

        iter_params (IteratorParams): Iterator which yield run parameters
        epochs (int): Number of epochs
        exp_name (str): Name of experiment
        key_params (list(str)): List of parameters whose values differ between iterations
        device (torch.device): Specify if use gpu or cpu
        """
        self.device = device if device else self.device
        self.manual_seed(42)
        # assign
        self.checkpoint_save_step = checkpoint_save_step
        self.verbose = verbose
        self.val_step = val_step
        self.base_path = os.path.join(os.getcwd(), f'data/{exp_name}/'
                                              f'_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        os.makedirs(self.base_path)
        df_runs = pd.DataFrame()
        for run, params_run in enumerate(iter_params):
            self.init_run(params_run)
            params_pooled = self.params_adjust(copy.deepcopy(params_run))
            self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(f'{self.base_path}/checkpoints')
            if self.params_clearml:
                task = Task.init(project_name=f'{exp_name}', task_name=f'run_{exp_name}_{run}')
            if self.is_tensorboard :
                self.writer = SummaryWriter(f'{self.base_path}/tensorboard/{self.date}')
                inp = next(iter(self.loaders['train']))[0]
                self.writer.add_graph(self.model, inp.to(self.device))

            self.loop(0, epochs)

            if self.is_tensorboard:
                self.writer.close()
            df_runs = pd.concat([df_runs, pd.DataFrame({'log loss': self.logs['train_log_loss'],
                                                        'accuracy': self.logs['train_accuracy'],
                                                        'val_log loss': self.logs['test_log_loss'],
                                                        'val_accuracy': self.logs['test_accuracy'],
                                                        **params_pooled}, index=[run])], axis=0)
            if self.params_clearml:
                task.close()
        df_runs.to_csv(f'{self.base_path}/{exp_name}.csv')

    def loop(self, epochs_start, epochs_end):
        for epoch in tqdm(range(epochs_start, epochs_end)):
            self.logs = {}

            self.model.train()
            self.run_epoch('train', epoch)

            self.model.eval()
            if (epoch + 1) % self.val_step == 0:
                phase = 'val' if 'val' in self.loaders else 'test'
                with torch.no_grad():
                    self.run_epoch(phase, epoch)

            if (epoch + 1) % self.checkpoint_save_step == 0:
                self.save_net(f'{self.base_path}/checkpoints/{self.date}_epoch_{epoch}.')

            if self.scheduler_:
                self.scheduler.step()
        with torch.no_grad():
            self.run_epoch('test', epoch)
        gc.collect()

    def init_run(self, params):
        """Initiate run."""
        self.model = self.model_(**params['model']).to(self.device)
        self.model.apply(self.init_weights)
        self.criterion = self.criterion_(**params['criterion']).to(self.device)
        self.optim = self.optim_(self.model.parameters(), **params['optim'])
        if self.scheduler_:
            self.scheduler = self.scheduler_(self.optim, **params['scheduler'])
        self.loaders = self.loaders_(**params['loaders'])


    def params_adjust(self, params):
        """Group run parameters from different dicts into one dict."""
        return dict(ChainMap(*params.values()))

    def run_epoch(self, phase, epoch):
        """Run whole epoch."""
        running_acc = 0.0
        running_loss = 0.0
        for x_true, y_true in self.loaders[phase]:
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            loss = self.criterion(y_pred, y_true)
            if phase == 'train':
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            running_acc += (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
            running_loss += loss.item() * x_true.size(0)

        epoch_acc = running_acc / len(self.loaders[phase].dataset)
        epoch_loss = running_loss / len(self.loaders[phase].dataset)

        self.logs[f'{phase}_accuracy'] = round(epoch_acc, 4)
        self.logs[f'{phase}_log_loss'] = round(epoch_loss, 4)
        self.writer.add_scalar(f'Acc/{phase}', self.logs[f'{phase}_accuracy'], epoch + 1)
        self.writer.add_scalar(f'Loss/{phase}', self.logs[f'{phase}_log_loss'], epoch + 1)
        if self.verbose:
            print(self.logs)

    def save_net(self, path):
        """ Saves policy_net parameters as given checkpoint.

        state_dict of current policy_net is stored.

        Args:
            path: path were to store model's parameters.
        """
        save_model(self.model, path)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def manual_seed(self, random_seed):
        import numpy as np
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)


class IteratorParams(object):
    """Iterate over all given values of parameters."""
    def __init__(self, model_ls, loaders_ls, criterion_ls, optim_ls, scheduler_ls):
        self.product = list(product(model_ls, loaders_ls, criterion_ls, optim_ls, scheduler_ls,))
        self.no_run = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.no_run += 1
        if self.no_run < len(self.product):
            tuple_run = self.product[self.no_run]
            return {
                'model': tuple_run[0],
                'loaders': tuple_run[1],
                'criterion': tuple_run[2],
                'optim': tuple_run[3],
                'scheduler': tuple_run[4],
            }
        raise StopIteration

