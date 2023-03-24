import torch
from torchvision import transforms
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import numpy as np


class CenterRatioCrop(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        dims = []
        if isinstance(img, torch.Tensor):
            dims = img.shape
        else:
            dims = img.size
        num_dims = len(dims)
        if num_dims == 2:
            size = (int(self.ratio*dims[0]), int(self.ratio*dims[1]))
        else:
            size = (int(self.ratio*dims[1]), int(self.ratio*dims[2]))
        return F.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ratio={self.ratio})"
    

def display_preprocessed_image(img, title='', mean_std=None):
    plt.figure()
    plt.title(title)
    
    mean = np.array([0.5862785803043838])
    std = np.array([0.27950088968644304])
    if mean_std is not None:
        mean = np.array(mean_std['mean'])
        std = np.array(mean_std['std'])
    transform = transforms.Compose([
        transforms.Normalize(
            mean=-mean*(1/std),
            std=1/std
        ),
        transforms.ToPILImage(),
    ])
    if img.shape[0] == 1:
        plt.imshow(transform(img), cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(transform(img))