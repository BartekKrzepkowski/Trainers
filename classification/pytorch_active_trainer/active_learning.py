from tensorboard_pytorch import TensorboardPyTorch

import numpy as np
import torch
import os
import torch.nn.functional as F

class Active_Learning:
    def __init__(self, model, loaders, unc_type):
        self.model = model
        self.loaders = loaders
        self.unc_type = unc_type
        self.device = next(model.parameters()).device
        self.help_func = {
            'sub': lambda x: x[0] - x[1],
            'div': lambda x: x[0] / x[1],
        }
        self.unc_sampling = {
            'entropy': lambda prob: -(prob * torch.log(prob)).sum(axis=-1),
            'least': lambda prob: 1 - torch.max(prob, dim=1)[0],
            'margin': lambda prob: 1 - self.help_func['sub'](torch.topk(prob, 2, axis=1)[0]),
            'ratio_margin': lambda prob: 1 / self.help_func['div'](torch.topk(prob, 2, axis=1)[0]),
        }

    def init_run(self, loaders, model, patience=5):
        self.patience = max(patience - 1, 1)
        self.N = len(loaders['unlabeled'].dataset)
        self.pred_props = torch.arange(self.N).float().unsqueeze(1)
        self.pred_labels = torch.cat([y_true for _, y_true in self.loaders['unlabeled']]).unsqueeze(1)

        self.unc_score = torch.arange(self.N).float().unsqueeze(1)
        self.true_labels = torch.cat([y_true for _, y_true in self.loaders['unlabeled']])
        self.n_class = self.true_labels.unique().shape[0]

    def tb_close(self):
        self.tb.close()

    def adjust_to_best_model(self):
        '''
        Korekta dla najlepszego modelu.
        dim(self.unlabeled_seq): [#unlabeled_data, #epochs_before_break]
        '''
        self.pred_props = self.pred_props[:, :-self.patience]
        self.pred_labels = self.pred_labels[:, :-self.patience]
        self.unc_score = self.unc_score[:, :-self.patience]


    def run_query(self, k):
        '''
        Po każdej epoce wskazuje indeks dla którego prob > 0.9, w p.p. -1.
        dim(self.unlabeled_seq): [#unlabeled_data, #epochs_before_break]
        '''
        pred = []
        labels = []
        unc = []
        unl_indices = self.loaders['unlabeled'].dataset.indices
        train_indices = self.loaders['train'].dataset.indices
        for x_data, _ in self.loaders['unlabeled']:
            y_pred = self.model(x_data.to(self.device))
            y_prob = F.softmax(y_pred, dim=-1).detach().cpu()
            # ssl
            pred_prob_max, labels_max = torch.max(y_prob, axis=-1)
            pred.append(pred_prob_max)
            labels.append(labels_max)
            # y_idx[y_prob_max < 0.9] = -1
            # al
            unc.append(self.unc_sampling[self.unc_type](y_prob))

        sorted_unc, indices_unc = torch.sort(torch.cat(unc, dim=0), descending=True)
        indices_unc = indices_unc[:k].numpy()
        self.loaders['unlabeled'].dataset.indices = np.delete(unl_indices, indices_unc, axis=0)
        self.loaders['train'].dataset.indices = np.concatenate([train_indices, indices_unc], axis=0)


    def update_labeled_idx0(self, root_dir, run_nb):
        mask = self.create_mask()
        x_unlabeled, y_unlabeled = torch.load(f'{root_dir}/unlabeled_idx_{run_nb}.pt'),\
                                   torch.load(f'{root_dir}/y_unlabeled_{run_nb}.pt')
        # save new unlabeled
        torch.save(x_unlabeled[~mask], f'{root_dir}/x_unlabeled_{run_nb+1}.pt')
        torch.save(y_unlabeled[~mask], f'{root_dir}/y_unlabeled_{run_nb+1}.pt')
        chosen_dir = f'{root_dir}/chosen'
        if not os.path.exists(chosen_dir): os.makedirs(chosen_dir)
        # save new labeled
        torch.save(y_unlabeled[mask], f'{chosen_dir}/y_true_{run_nb}.pt')
        torch.save(x_unlabeled[mask], f'{chosen_dir}/x_chosen_{run_nb}.pt')
        torch.save(self.unlabeled_seq[mask][:, -1], f'{chosen_dir}/y_chosen_{run_nb}.pt') # predicted labels for chosen
        # self.tb.classification_report(self.unlabeled_seq[mask][:, -1].numpy(), y_unlabeled[mask].numpy(), step=run_nb)

    def update_labeled_idx(self, data_dir, run_nb, K):
        mask = self.create_mask(run_nb, K)
        unlabeled_idx = self.loaders['unlabeled'].dataset.indices
        # save new unlabeled
        torch.save(unlabeled_idx[~mask], f'{data_dir}/idxs/unlabeled_idx_{run_nb+1}.pt')
        chosen_dir = f'{data_dir}/chosen'
        if not os.path.exists(chosen_dir): os.makedirs(chosen_dir)
        # save labeled diff
        torch.save(unlabeled_idx[mask], f'{chosen_dir}/train_idx_{run_nb}.pt')
        # save predicted labels for chosen
        torch.save(self.pred_labels[mask][:, -1], f'{chosen_dir}/y_chosen_{run_nb}.pt')
        # self.tb.classification_report(self.above_th[mask][:, -1].numpy(),
        #                               self.above_th[:, 0][mask].numpy(), step=run_nb)


    def create_mask_old(self, gamma=1/2):
        epochs = self.above_th.shape[1]
        cutoff = int(0.6 * epochs)
        # indicate examples to label
        mask_b = np.array([len(set(tens.numpy())) == 1 and tens[-1] != -1
                         for tens in self.above_th[:, cutoff:]])
        return mask_b


    def create_mask(self, run_nb, K, gamma=1/2):
        from collections import Counter
        mask = np.full(self.N, False)
        epochs = self.pred_labels.shape[1]
        cutoff = int(0.6 * epochs)
        gamma_tensor = torch.tensor([gamma**(i) for i in range(epochs-cutoff)[::-1]]).unsqueeze(1)

        if self.is_ssl:
            K_part = (K//10) * ((run_nb+1) if self.unc_scorer else 10)
            mask += self.get_ssl_mask(gamma_tensor, K_part, cutoff, epochs)

        if self.unc_scorer:
            K_part = (K//10) * ((10-run_nb-1) if self.is_ssl else 10)
            mask += self.get_al_mask(gamma_tensor, K_part, cutoff, epochs)

        return mask

    def get_ssl_mask(self, gamma_tensor, K_part, cutoff, epochs):
        # print(cutoff, epochs, self.pred_labels.shape, self.pred_props.shape)
        v0 = torch.zeros(self.N, self.n_class, epochs - cutoff)
        v0 = v0.scatter(1, self.pred_labels[:, cutoff:].unsqueeze(1),
                        self.pred_props[:, cutoff:].unsqueeze(1))
        v0 = (v0 @ gamma_tensor).squeeze()
        scores, labels = torch.max(v0, axis=1)
        topk_labels = self.topk_balanced(labels, scores, K_part)
        return np.isin(np.arange(self.N), topk_labels)

    def get_al_mask(self, gamma_tensor, K_part, cutoff, epochs):
        scores_u = (self.unc_score[:, cutoff:] @ gamma_tensor).squeeze()
        topk_labels_u = self.topk_balanced(self.true_labels, scores_u, K_part)
        return np.isin(np.arange(self.N), topk_labels_u)

    def topk_balanced(self, labels, scores, K):
        top_labels = []
        scores = torch.cat([scores.unsqueeze(1),
                            torch.arange(scores.shape[0]).unsqueeze(1).float()], axis=1)
        for i in range(self.n_class):
            scores_class = scores[labels == i]
            indices_sorted = scores_class[:, 0].argsort(descending=True)
            top_labels.append(scores_class[indices_sorted, 1][: K // self.n_class].long())
        return torch.cat(top_labels).numpy()
