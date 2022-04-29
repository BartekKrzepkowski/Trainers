import datetime
from collections import ChainMap

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


def params_adjust(params):
    """Group run parameters from different dicts into one dict."""
    return dict(ChainMap(*params.values()))


def loaders_from_dataset(train_dataset, test_dataset=None, batch_size=16, val_perc_size=0):
    loaders = {}
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=to_input_tensor,
                                 pin_memory=True, num_workers=4)
        loaders['test'] = test_loader
    if val_perc_size > 0:
        train_size = len(train_dataset)
        val_size = int(train_size * val_perc_size)
        train_dataset, val_dataset = random_split(train_dataset, [train_size - val_size, val_size])
        val_dataset.transform = test_dataset.transform
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=to_input_tensor,
                                pin_memory=True, num_workers=4)
        loaders['val'] = val_loader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=to_input_tensor,
                              pin_memory=True, num_workers=4)
    loaders['train'] = train_loader
    return loaders


def to_input_tensor(source):
    source = sorted(source, key=lambda e: len(e), reverse=True)
    x_true = [sent[:-1] for sent in source]
    y_true = [sent[1:] for sent in source]
    lengths = [len(s) for s in x_true]
    x_true = torch.tensor(pad_sents(x_true, 0), dtype=torch.long)
    y_true = torch.tensor(pad_sents(y_true, 0), dtype=torch.long)
    return torch.t(x_true), torch.t(y_true), lengths


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()


def visualize_embeddings(embs, chosen_idxs, Ind2word):
    if len(chosen_idxs) != 0:
        embs = embs[chosen_idxs]
    temp = (embs - np.mean(embs, axis=0))
    covariance = 1.0 / len(temp) * (temp.T @ temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp @ U[:, :2]

    # svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
    # coord = svd.fit_transform(covariance)
    print('SVD done!')

    plt.figure(figsize=(15, 15))
    for i in range(len(chosen_idxs)):
        plt.text(coord[i, 0], coord[i, 1], Ind2word[chosen_idxs[i]],
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'word_vectors_{stamp}.png')


def set_seeds(random_seed=42) -> None:
    """
    Set random seeds for reproduceability.
    See: https://pytorch.org/docs/stable/notes/randomness.html
         https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    """
    #########################
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    #########################
    
    
def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
     
    Args:
        sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
        pad_token (str): padding token
    
    Returns:
        sents_padded (list[list[str]]): 
            list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, 
            such that each sentences in the batch now has equal length.
    """
    sents_padded = []
    max_length = max([len(sent) for sent in sents])
    for i, sent in enumerate(sents):
        padded_sent = sent + [pad_token] * (max_length - len(sent))
        sents_padded.append(padded_sent)

    return sents_padded


def save_model(model, PATH):
    """ Saves model's state_dict.
    Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    torch.save(model.state_dict(), PATH)


def load_model(model, PATH):
    """ Loads model's parameters from state_dict """
    model.load_state_dict(torch.load(PATH))
