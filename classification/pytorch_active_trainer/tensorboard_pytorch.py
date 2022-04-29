import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy

from tensorboard_utils import weight_histograms_linear, weight_histograms_rnn, weight_histograms_conv2d, get_val_data

functions = {
    'entropy': lambda prob: -(prob * torch.log(prob)).sum(axis=-1),
    'max_prob': lambda prob: prob.max(axis=-1)[0],
    'max_label': lambda prob: prob.max(axis=-1)[1],
    'max_prev_prob': lambda prob: prob.sort(dim=-1,
                                            descending=True)[0][:,1]
}

internal_params = {
    'func1d': ['entropy', 'max_prob', 'max_label'],
    'epoch_threshold': 10,
    'prob_threshold': 0.9,
    'amount': 5
}


class TensorboardPyTorch:
    def __init__(self, log_name, device):
        self.writer = SummaryWriter(log_dir=log_name, flush_secs=60)
        self.device = device
        self.tensors = {}

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def update_tensors(self, name, value):
        if name in self.tensors:
            self.tensors[name] = torch.cat((self.tensors[name], value))
        else:
            self.tensors[name] = value

    def release_tensors(self):
        self.tensors = {}

    def log_graph(self, model, inp):
        self.writer.add_graph(model, inp)

    def log_histogram(self, tag, tensor, global_step=0):
        self.writer.add_histogram(tag, tensor, global_step)
        self.flush()

    def log_scalar(self, tag, scalar, global_step):
        self.writer.add_scalar(tag, scalar, global_step)
        self.flush()

    def log_weight_histogram(self, model, epoch):
        # Iterate over all model layers
        for name, layer in model.named_modules():
            # Compute weight histograms for appropriate layer
            if isinstance(layer, nn.Conv2d):
                weight_histograms_conv2d(self.writer, epoch, layer.weight, name)
            elif isinstance(layer, nn.Linear):
                weight_histograms_linear(self.writer, epoch, layer.weight, name)
            elif isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU):
                weight_histograms_rnn(self.writer, epoch, layer, name)
        self.flush()

    def log_epsilon_typical(self, eps, epoch):
        tag = 'epsilon'
        self.writer.add_histogram(tag, eps.flatten(), global_step=epoch, bins='tensorflow')

    def log_histogram_values1d(self, epoch, phase):
        prob = torch.softmax(self.tensors['y_pred'], dim=-1)
        for name in internal_params['func1d']:
            self.writer.add_histogram(f'{name}/{phase}/',
                                      functions[name](prob),
                                      global_step=epoch, bins='auto')
        self.flush()

    # popraw tak by wszystkie labele z jednej fazy były na jednym plocie
    def log_pr_curve_per_label(self, epoch, phase):
        prob = torch.softmax(self.tensors['y_pred'], dim=-1)
        y_true = self.tensors['y_true']
        for label in y_true.unique():
            self.writer.add_pr_curve(f'pr_curve/{phase}/{label}',
                                     labels=y_true == label,
                                     predictions=prob[:, label],
                                     global_step=epoch)
        self.flush()

    def log_embeddings(self, model, dataset, epoch):
        self.writer.add_embedding(
            model.forward(get_val_data(dataset, 'data').to(self.device)),
            metadata=get_val_data(dataset, 'target'),
            label_img=get_val_data(dataset, 'data'),
            global_step=epoch)
        self.flush()

    def log_misc_image(self, dataset):
        pass
        # y_prob, y_pred = torch.max(y_prob, axis=1)
        # idxs = ((y_pred != self.y_true) & (y_prob > self.internal_params['prob_threshold']))
        # idxs = idxs.cpu().numpy().nonzero()[0];
        # if idxs.shape[0] == 0: return
        #
        # sample_size = min(self.internal_params['amount'], idxs.shape[0])
        # sample_idxs = np.random.choice(idxs, replace=False, size=sample_size)
        # imgs = get_val_data(dataset, 'data', external_ids=sample_idxs)
        # for i, idx in enumerate(sample_idxs):
        #     img_name = f'Val-Misclassified/Epoch-{self.epoch.count}/Label-{self.y_true[idx]}' \
        #                f'/Prob-{y_prob[idx]:.3f}_Prediction-{y_pred[idx]}/'
        #     self.writer.add_image(img_name, imgs[i], global_step=self.epoch.count)
        # self.flush()

    def log_at_epoch_end(self, epoch, phase, model=None, loaders=None):
        self.log_histogram_values1d(epoch, phase)
        # if self.flags['pr_curve_per_label']:
            # self.log_pr_curve_per_label(epoch, phase)
        # if self.flags['embeddings']:
        #     self.log_embeddings(model, loaders['train'].dataset)
        # if self.flags['image_misclassifications'] and self.epoch.count >= self.internal_params['epoch_threshold']:
        #     self.log_misc_image(self.loaders['val'].dataset)
        self.release_tensors()
