import os
from torch.utils import data as d

from skimage import io

class Dataset(d.Dataset):
    def __init__(self, img_root, img_info, img_ids, full_labels, classes):
        super().__init__()
        # Each dataframe should contain: image_id, file, class 1, ..., class n, meta_split
        self.image_info = img_info
        self.img_root = img_root
        self.image_ids = img_ids
        self.label_names = full_labels
        self.classes = classes
        self.classes.sort()

    def class_labels(self):
        return [self.label_names[c] for c in self.classes]
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.image_info[self.image_info['image_id'] == img_id].iloc[0]
        class_inds = info[self.classes]
        image = io.imread(os.path.join(self.img_root, info['file']))
        return image, class_inds

    def __len__(self):
        return len(self.image_ids)