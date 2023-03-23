import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import copy

from skimage import io
import torchvision.transforms as transforms

from utils.metrics import AverageMeter, calculate_auc, multilabel_accuracy
from utils.sampling import FewShotBatchSampler
from utils.image import CenterRatioCrop
from utils.data import get_query_and_support_ids

class DataloaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def next_batch(self):
        batch = next(self.iterator, None)
        if batch is None:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

class MDataset(Dataset):
    def __init__(self, img_root, img_info, img_ids, full_labels, classes, sampler=None, crop=0.9, mean_std=None):
        super().__init__()
        # Each dataframe should contain: image_id, file, class 1, ..., class n, meta_split
        self.img_root = img_root
        self.image_info = img_info
        self.image_ids = img_ids
        self.label_names = full_labels

        self.classes = classes
        self.classes.sort()
        self.classes = np.array(classes)

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
        self.sampler = sampler

    def class_labels(self, classes=None):
        if classes is None:
            classes = self.classes
        return [self.label_names[c] for c in classes]
    
    def get_class_indicators(self):
        img_infos= pd.DataFrame(self.image_ids, columns=['image_id']).merge(self.image_info, on='image_id', how='inner')
        return img_infos[self.classes].to_numpy().astype(int)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.image_info[self.image_info['image_id'] == img_id].iloc[0]

        if self.sampler is not None:
            classes = self.classes[self.sampler.curr_classes]
        else:
            classes = self.classes

        class_inds = info[classes].to_numpy().astype(int)
        img_path = os.path.join(self.img_root, info['file'])
        image = self.transforms(io.imread(img_path))
        return image, class_inds, self.class_labels(classes)

    def __len__(self):
        return len(self.image_ids)

class ClassCycler:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classes = []
        self._update_class_list()
    
    def _update_class_list(self):
        self.classes.extend(np.random.choice(self.num_classes, self.num_classes, replace=False))

    def next_n(self, n):
        if n > len(self.classes):
            self._update_class_list()
        out = self.classes[:n]
        self.classes = self.classes[n:]
        return out
    
class ControlledFewShotBatchSampler(FewShotBatchSampler):
    def __init__(self, labels, k_shot, n_ways=None, include_query=False):
        super(ControlledFewShotBatchSampler, self).__init__(labels, k_shot, n_ways=n_ways, include_query=include_query)
        self.sample_classes = None
        self.cycler = None

    def reset_sample_classes(self):
        self.sample_classes = None

    def set_sample_classes(self, classes):
        self.sample_classes = classes
    
    def get_iteration_classes(self, _):
        if self.sample_classes is not None:
            classes = self.sample_classes
        else:
            if self.cycler is None:
                self.cycler = ClassCycler(self.num_classes)
            classes = self.cycler.next_n(self.n_ways)
        
        if self.include_query:
            classes = sorted(classes, key=lambda c: len(self.class_indices[c]))
        
        return classes
    
class ControlledMetaTrainer:
    def __init__(self, model, shots, n_ways, dataset_config, train_n_ways=None, device='cpu', metric_funcs=None):
        self.model = model
        self.device = device

        self.shots = shots
        self.n_ways = n_ways
        self.train_n_ways = train_n_ways
        if self.train_n_ways is None:
            self.train_n_ways = n_ways

        self.dataset_config = dataset_config
        self.initialise_datasets(dataset_config)
        self.initialise_dataloaders()
        self.set_metric_funcs(metric_funcs)
    
    def set_metric_funcs(self, metric_funcs):
        self.accuracy_func = multilabel_accuracy
        self.auc_func = calculate_auc
        if metric_funcs is not None:
            if 'acc' in metric_funcs:
                self.accuracy_func = metric_funcs['acc']
            if 'auc' in metric_funcs:
                self.auc_func = metric_funcs['auc']

    def create_query_eval_dataloader(self, split_type='train'):
        img_info = self.dataset_config.img_info
        img_path = self.dataset_config.img_path
        classes_split_map = self.dataset_config.classes_split_map
        label_names_map = self.dataset_config.label_names_map
        mean_std = self.dataset_config.mean_std

        query_dataset = MDataset(img_path, img_info, self.query_image_ids, label_names_map, classes_split_map[split_type], mean_std=mean_std)
        return _create_dataloader(query_dataset, self.shots, self.n_ways, include_query=True)

    def initialise_datasets(self, ds_config):
        img_info = ds_config.img_info
        img_path = ds_config.img_path
        classes_split_map = ds_config.classes_split_map
        label_names_map = ds_config.label_names_map
        mean_std = ds_config.mean_std

        query_image_ids, support_image_ids = get_query_and_support_ids(img_info, ds_config.training_split_path)
        self.query_image_ids = query_image_ids
        self.train_query_dataset = MDataset(img_path, img_info, query_image_ids, label_names_map, classes_split_map['train'], mean_std=mean_std)
        self.train_support_dataset = MDataset(img_path, img_info, support_image_ids, label_names_map, classes_split_map['train'], mean_std=mean_std)
        
        self.val_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'val', mean_std)
        self.test_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'test', mean_std)

    def initialise_dataloaders(self):
        self.train_query_loader = _create_dataloader(self.train_query_dataset, self.shots, self.train_n_ways, include_query=False)
        self.train_support_loader = _create_dataloader(self.train_support_dataset, self.shots, self.train_n_ways, include_query=False)

        self.val_loader = _create_dataloader(self.val_dataset, self.shots, self.n_ways, include_query=True)
        self.test_loader = _create_dataloader(self.test_dataset, self.shots, self.n_ways, include_query=True)

    def _update_train_iteration_classes(self, classes):
        self.train_query_dataset.sampler.set_sample_classes(classes)
        self.train_support_dataset.sampler.set_sample_classes(classes)

    def _reset_train_iteration_classes(self):
        self.train_query_dataset.sampler.reset_sample_classes()
        self.train_support_dataset.sampler.reset_sample_classes()
        
    def run_train(self, epochs, lr=1e-5):
        model = self.model.to(self.device)
        best_epoch = None
        best_acc = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        cycler = ClassCycler(len(self.train_query_dataset.classes))
        support_iterator = DataloaderIterator(self.train_support_loader)
        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            query_iterator = DataloaderIterator(self.train_query_loader)
            for i in range(len(self.train_query_loader)):
                self._update_train_iteration_classes(cycler.next_n(self.train_n_ways))

                qimages, qclass_inds, train_class_labels = query_iterator.next_batch()
                qimages, qclass_inds = qimages.to(self.device), qclass_inds.to(self.device)
                simages, sclass_inds, support_class_labels = support_iterator.next_batch()
                assert train_class_labels == support_class_labels, f"Mismatch {train_class_labels} {support_class_labels}"
                simages, sclass_inds = simages.to(self.device), sclass_inds.to(self.device)

                optimizer.zero_grad()
                predictions = model.update_support_and_classify(train_class_labels, simages, sclass_inds, qimages)
                loss = model.loss(predictions, qclass_inds)

                acc = self.accuracy_func(predictions, qclass_inds)
                acc_meter.update(acc, qclass_inds.shape[0])

                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), len(qimages))
                print(f"Batch {i+1}: loss {loss_meter.average()} | Acc {acc}")
            
            print(f"Epoch {epoch+1}: Training loss {loss_meter.average()} | Acc: {acc_meter.average()}")

            val_acc, val_auc, val_loss = self.run_eval(model, self.val_loader)
            print(f"Epoch {epoch+1}: Validation loss {val_loss} | Accuracy {val_acc} | AUC {val_auc}")

            self._reset_train_iteration_classes()

            if best_acc is None or val_acc > best_acc:
                best_acc = val_acc
                self.best_model = copy.deepcopy(model)
                best_epoch = epoch
        self.model = model
        print('Best epoch: ', best_epoch+1)
    
    def run_eval(self, model, dataloader, verbose=False):
        model.eval()
        model = model.to(self.device)
        
        loss_meter = AverageMeter()
        auc_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            for images, class_inds, class_labels in dataloader:
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                shots = images.shape[0]//2
                qimages, simages = images[:shots,:,:], images[shots:,:,:]
                qclass_inds, sclass_inds = class_inds[:shots,:], class_inds[shots:,:]

                predictions = model.update_support_and_classify(class_labels, simages, sclass_inds, qimages)
                loss = model.loss(predictions, qclass_inds)

                loss_meter.update(loss.item(), shots)

                auc = self.auc_func(predictions, qclass_inds)
                auc_meter.update(auc, shots)
            
                acc = self.accuracy_func(predictions, qclass_inds)
                acc_meter.update(acc, shots)
                if verbose:
                    # print(class_labels)
                    # print(torch.nonzero(class_inds)[:,1].bincount())
                    print(f"Loss {loss} | Accuracy {acc} | AUC {auc}")
        return acc_meter.average(), auc_meter.average(), loss_meter.average()
    
def _collate_batch(batch):
    images, class_inds, labels = [], [], []
    for (_image, _inds, _labels) in batch:
        images.append(_image)
        class_inds.append(_inds)
        if len(labels) > 0:
            assert labels == _labels
        labels = _labels
    
    images = torch.stack(images)
    class_inds = torch.tensor(np.array(class_inds))
    return images, class_inds, labels
    
def _create_dataset(img_path, img_info, label_names_map, classes_split_map, split, mean_std=None):
    img_ids = img_info[img_info['meta_split'] == split]['image_id'].to_list()
    return MDataset(img_path, img_info, img_ids, label_names_map, classes_split_map[split], mean_std=mean_std)

def _create_dataloader(dataset, shots, n_ways, include_query):
    if dataset.sampler is None:
        sampler = ControlledFewShotBatchSampler(dataset.get_class_indicators(), shots, n_ways=n_ways, include_query=include_query)
        dataset.sampler = sampler
    return DataLoader(dataset, batch_sampler=dataset.sampler, collate_fn=_collate_batch)

    