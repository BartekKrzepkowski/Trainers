import torch


def save_model(model, PATH):
    """ Saves model's state_dict.

    Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    torch.save(model.state_dict(), PATH)


def load_model(model, PATH):
    """ Loads model's parameters from state_dict """
    model.load_state_dict(torch.load(PATH))