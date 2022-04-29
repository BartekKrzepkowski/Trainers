import torch
from torchvision.transforms import Compose, ToTensor, Normalize,\
    RandomAffine, RandomHorizontalFlip, Pad
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision.datasets import CIFAR10, MNIST
import numpy as np


class Coreset:
    def __init__(self, loaders, pretrained_model):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loaders = loaders
        self.pretrained_model = pretrained_model.to(self.device) # spróbuj autoenkoder

    def get_features(self, unl_loader, pretrained_model):
        print('Getting features!!!')
        features = np.array([])
        for x_data, _ in unl_loader:
            x_data = x_data.to(self.device)
            x_feature = pretrained_model(x_data).detach().cpu().numpy()
            features = np.vstack([features, x_feature]) if features.size else x_feature
        return features

    def get_coreset(self, features, n, k):
        from sklearn.cluster import KMeans as KM
        print('Getting coreset indices!!!')
        coreset_dict = {}
        km_fit = KM(n_clusters=n).fit(features)
        pred = km_fit.fit_transform(features)
        for i, label in enumerate(np.unique(km_fit.labels_)):
            mask_label = km_fit.labels_ == label
            idxs = np.nonzero(mask_label)[0]
            min_k = np.argsort(pred[:, i][mask_label], axis=-1)[k:]
            coreset_dict[label] = idxs[min_k]
        return coreset_dict

    def coreset(self, n=10, k=10):
        # słownik jest nadpisywany
        unl_loader = self.loaders['train']
        unl_idxs = np.arange(len(unl_loader.dataset))
        features = self.get_features(unl_loader, self.pretrained_model)
        coreset_dict = self.get_coreset(features, n, k)
        core_idxs = np.concatenate(list(coreset_dict.values()))
        unl_idxs = np.delete(unl_idxs, core_idxs, axis=0)
        train_dataset = Subset(unl_loader.dataset, core_idxs)
        unl_dataset = Subset(unl_loader.dataset, unl_idxs)
        self.loaders['train'] = DataLoader(train_dataset, batch_size=unl_loader.batch_size,
                                           shuffle=True, pin_memory=True, num_workers=4)
        self.loaders['unlabeled'] = DataLoader(unl_dataset, batch_size=unl_loader.batch_size,
                                         shuffle=False, pin_memory=True, num_workers=4)


def loaders_from_dataset(train_dataset, test_dataset=None, batch_size=32, val_perc_size=0):
    loaders = {}
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        loaders['test'] = test_loader
    if val_perc_size > 0:
        train_size = len(train_dataset.dataset)
        val_size = int(train_size * val_perc_size)
        train_dataset, val_dataset = random_split(train_dataset, [train_size - val_size, val_size])
        val_dataset.transform = test_dataset.transform
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        loaders['val'] = val_loader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    loaders['train'] = train_loader
    return loaders


def loaders_from_dataset_active_learning(train_dataset, test_dataset, transform_test, batch_size, init_idxs=None):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    loaders = {
        'test': test_loader
    }
    if init_idxs is not None:
        train_size = len(train_dataset.dataset)
        val_size = int(train_size * val_perc_size)
        train_dataset, val_dataset = random_split(train_dataset, [train_size - val_size, val_size])
        val_dataset.transform = transform_test
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        loaders['val'] = val_loader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    loaders['train'] = train_loader
    return loaders


def loaders_example(batch_size, dataset_name, val_perc_size=0, is_coreset=False, is_shuffled=True):
    ########
    # datasets & transforms
    if dataset_name == 'cifar10':
#         transform_test = 
        transform_train = Compose([ToTensor(), RandomAffine(degrees=0, translate=(1/8, 1/8)), RandomHorizontalFlip(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10
    elif dataset_name == 'mnist':
        transform_train = Compose([ToTensor(), Pad(2), Normalize((0.5,), (0.5,))])
        dataset = MNIST
    ########

    train_dataset = dataset(root='data', 
                            train=True,
                            download=True,
                            transform=transform_train)

    test_dataset = dataset(root='data', 
                           train=False,
                           download=True,
                           transform=transform_train)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    loaders = {
        'test': test_loader
    }
    if val_perc_size > 0:
        train_size = len(train_dataset.dataset)
        val_size = int(train_size * val_perc_size)
        train_dataset, val_dataset = random_split(train_dataset, [train_size - val_size, val_size])
        # val_dataset.transform = transform_test
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        loaders['val'] = val_loader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffled, pin_memory=True, num_workers=4)
    loaders['train'] = train_loader

    if is_coreset:
        from torchvision.models import resnet18
        pretrained_model = resnet18(pretrained=True)
        coreset = Coreset(loaders, pretrained_model)
        coreset.coreset(n=10, k=100)
    else:
        indices = np.arange(len(train_dataset))
        train_indices = np.random.choice(indices, 1000, replace=False)
        unlabeled_dataset = Subset(train_dataset, np.delete(indices, train_indices))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        loaders['unlabeled'] = unlabeled_loader
        train_dataset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=4)
        loaders['train'] = train_loader
    return loaders

