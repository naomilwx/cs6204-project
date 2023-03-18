import os
from torch.utils import data
import numpy as np
import pandas as pd

from skimage import io
import torchvision.transforms as transforms
from utils.image import CenterRatioCrop

class Dataset(data.Dataset):
    def __init__(self, img_root, img_info, img_ids, full_labels, classes, crop=0.9, mean_std=None):
        super().__init__()
        # Each dataframe should contain: image_id, file, class 1, ..., class n, meta_split
        self.img_root = img_root
        self.image_info = img_info
        self.image_ids = img_ids
        self.label_names = full_labels
        self.classes = classes
        self.classes.sort()
        tfms = [transforms.ToTensor()]
        if crop < 1:
            tfms.append(CenterRatioCrop(crop))
        normalize = transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])
        if mean_std is not None:
            normalize =  transforms.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        tfms += [
            transforms.Resize((224, 224), antialias=True),
            normalize
        ]
        self.transforms = transforms.Compose(tfms)

    def class_labels(self):
        return [self.label_names[c] for c in self.classes]
    
    def get_class_indicators(self):
        img_infos= pd.DataFrame(self.image_ids, columns=['image_id']).merge(self.image_info, on='image_id', how='inner')
        return img_infos[self.classes].to_numpy().astype(int)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.image_info[self.image_info['image_id'] == img_id].iloc[0]
        class_inds = info[self.classes].to_numpy().astype(int)
        img_path = os.path.join(self.img_root, info['file'])
        image = self.transforms(io.imread(img_path))
        return image, class_inds

    def __len__(self):
        return len(self.image_ids)