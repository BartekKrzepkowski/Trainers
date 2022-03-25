from torchvision.transforms import Compose, ToTensor, Normalize,\
    RandomAffine, RandomHorizontalFlip, Pad
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST


def loaders_from_dataset(train_dataset, test_dataset, transform_test, batch_size, val_perc_size=0):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    loaders = {
        'test': test_loader
    }
    if val_perc_size > 0:
        train_size = len(train_dataset.dataset)
        val_size = int(train_size * val_perc_size)
        train_dataset, val_dataset = random_split(train_dataset, [train_size - val_size, val_size])
        val_dataset.transform = transform_test
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        loaders['val'] = val_loader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    loaders['train'] = train_loader
    return loaders


def loaders_example(batch_size, dataset_name, val_perc_size=0):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    loaders['train'] = train_loader
    
        
    return loaders

