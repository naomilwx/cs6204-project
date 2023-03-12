import torch
import torchvision.transforms.functional as F

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