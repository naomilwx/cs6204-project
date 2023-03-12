import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from utils.sampling import MultilabelBalancedRandomSampler, create_single_class_sampler

MEAN_STDS = {
  'pathmnist': { 'mean': [0.7405, 0.5330, 0.7058], 'std': [0.3920, 0.5636, 0.3959] },
  'chestmnist': { 'mean': [0.4936], 'std': [0.7392] },
  'dermamnist': { 'mean': [0.7631, 0.5381, 0.5614], 'std': [0.1354, 0.1530, 0.1679] },
  'octmnist': { 'mean': [0.1889],  'std': [0.6606] },
  'pneumoniamnist': { 'mean': [0.5719], 'std': [0.1651] },
  'retinamnist': { 'mean': [0.3984, 0.2447, 0.1558], 'std': [0.2952, 0.1970, 0.1470] },
  'breastmnist': { 'mean': [0.3276], 'std': [0.2027] },
  'bloodmnist': { 'mean': [0.7943, 0.6597, 0.6962], 'std': [0.2930, 0.3292, 0.1541] },
  'tissuemnist': { 'mean': [0.1020], 'std': [0.4443] },
  'organamnist': { 'mean': [0.4678], 'std': [0.6105] },
  'organcmnist': { 'mean': [0.4932], 'std': [0.3762] },
  'organsmnist': { 'mean': [0.4950], 'std': [0.3779] }
}

class DataSets:
  def __init__(self, dataname, mean_stds):
    self.info = INFO[dataname]
    self.dataname = dataname
    self.task = self.info['task']
    self.DataClass = getattr(medmnist, self.info['python_class'])
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # mean=[0.5862785803043838],std=[0.27950088968644304]
                transforms.Normalize(mean=mean_stds[dataname]['mean'], std=mean_stds[dataname]['std'])
            ])
    self.train_dataset = self.DataClass(split='train', transform=transform, download=True)
    self.test_dataset = self.DataClass(split='test', transform=transform, download=True)
    self.val_dataset = self.DataClass(split='val', transform=transform, download=True)
