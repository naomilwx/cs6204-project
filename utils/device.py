import torch

def get_device():
    dev = 'cpu'
    if torch.cuda.is_available():
        dev = 'cuda:0'
    elif torch.backends.mps.is_available():
        dev = 'mps'
    return torch.device(dev)