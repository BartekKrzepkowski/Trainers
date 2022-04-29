import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy

from tensorboard_utils import weight_histograms_linear, weight_histograms_rnn, weight_histograms_conv2d, get_val_data

functions = {
    'entropy': lambda prob: torch.tensor(entropy(prob.cpu().numpy().T, base=2)),
    'max_prob': lambda prob: prob.max(axis=1)[0],
    'max_label': lambda prob: prob.max(axis=1)[1],
    'max_prev_prob': lambda prob: prob.sort(dim=1,
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
        self.y_prob = torch.FloatTensor().to(self.device)
        self.y_true = torch.LongTensor().to(self.device)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def update_y(self, y_prob, y_true):
        self.y_prob = torch.cat((self.y_prob, y_prob))
        self.y_true = torch.cat((self.y_true, y_true))

    def release_y(self):
        del self.y_prob
        del self.y_true
        self.y_prob = torch.FloatTensor().to(self.device)
        self.y_true = torch.LongTensor().to(self.device)

    def log_graph(self, model, inp):
        self.writer.add_graph(model, inp)

    def log_scalar(self, tag, scalar, global_step):
        self.writer.add_scalar(tag, scalar, global_step)

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

    def log_histogram_values1d(self, epoch):
        for name in internal_params['func1d']:
            self.writer.add_histogram(f'Histogram/{name}/',
                                      functions[name](self.y_prob),
                                      global_step=epoch, bins='auto')
        self.flush()

    def log_pr_curve_per_label(self, epoch, labels):
        for label in labels:
            self.writer.add_pr_curve(f'pr_curve/{label}',
                                     labels=self.y_true == label,
                                     predictions=self.y_prob[:, label],
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

    def log_at_epoch_end_val(self, model=None, loaders=None):
        self.log_scalars(denominator=self.mm.dataset_size['val'],
                         global_step=self.epoch.count,
                         phase='val', prefix='')
        # if self.flags['histogram_values1d']:
        #     self.log_histogram_values1d()
        # if self.flags['pr_curve_per_label']:
        #     self.log_pr_curve_per_label()
        if self.flags['embeddings']:
            self.log_embeddings(model, loaders['train'].dataset)
        # if self.flags['image_misclassifications'] and self.epoch.count >= self.internal_params['epoch_threshold']:
        #     self.log_misc_image(self.loaders['val'].dataset)
        self.release_y()