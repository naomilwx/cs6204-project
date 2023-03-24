import torch
import random
import numpy as np

def get_device():
    dev = 'cpu'
    if torch.cuda.is_available():
        dev = 'cuda:0'
    elif torch.backends.mps.is_available():
        dev = 'mps'
    return torch.device(dev)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)