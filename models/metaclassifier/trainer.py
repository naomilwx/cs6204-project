import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import copy

from skimage import io
import torchvision.transforms as transforms

from utils.metrics import AverageMeter
from utils.sampling import FewShotBatchSampler, ClassCycler
from utils.image import CenterRatioCrop
from utils.data import get_query_and_support_ids

from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity, MultilabelF1Score, MultilabelAccuracy, MultilabelPrecision, MultilabelAveragePrecision

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
    
class ControlledFewShotBatchSampler(FewShotBatchSampler):
    def __init__(self, labels, k_shot, n_ways=None, include_query=False, n_query=None):
        super(ControlledFewShotBatchSampler, self).__init__(labels, k_shot, n_ways=n_ways, include_query=include_query)
        self.sample_classes = None
        if n_query is not None:
            self.set_query_size(n_query)

    def reset_sample_classes(self):
        self.sample_classes = None

    def set_sample_classes(self, classes):
        self.sample_classes = classes
    
    def get_iteration_classes(self, it):
        if self.sample_classes is not None:
            return self.sample_classes
        else:
            return super().get_iteration_classes(it)
        
def create_baseline_eval_dataloaders(ds_config, n_ways, n_query):
    img_info = ds_config.img_info
    img_path = ds_config.img_path
    classes_split_map = ds_config.classes_split_map
    label_names_map = ds_config.label_names_map
    mean_std = ds_config.mean_std

    query_image_ids, _ = get_query_and_support_ids(img_info, ds_config.training_split_path)
    train_query_dataset = MDataset(img_path, img_info, query_image_ids, label_names_map, classes_split_map['train'], mean_std=mean_std)
    val_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'val', mean_std)
    test_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'test', mean_std)

    train_query_loader = _create_dataloader(train_query_dataset, n_query, n_ways, n_query=0)
    val_loader = _create_dataloader(val_dataset, n_query, n_ways, n_query=0)
    test_loader = _create_dataloader(test_dataset, n_query, n_ways, n_query=0)

    return train_query_loader, val_loader, test_loader
    
class ControlledMetaTrainer:
    def __init__(self, model, shots, n_ways, dataset_config, train_n_ways=None, n_query=None, device='cpu'):
        self.model = model
        self.device = device
        self.best_acc = None

        self.shots = shots
        self.n_ways = n_ways
        self.n_query = shots
        if n_query is not None:
            self.n_query = n_query
        self.train_n_ways = train_n_ways
        if self.train_n_ways is None:
            self.train_n_ways = n_ways

        self.dataset_config = dataset_config
        self.initialise_datasets(dataset_config)
        self.initialise_dataloaders()

    
    def create_query_eval_dataloader(self, split_type='train'):
        img_info = self.dataset_config.img_info
        img_path = self.dataset_config.img_path
        classes_split_map = self.dataset_config.classes_split_map
        label_names_map = self.dataset_config.label_names_map
        mean_std = self.dataset_config.mean_std

        query_dataset = MDataset(img_path, img_info, self.query_image_ids, label_names_map, classes_split_map[split_type], mean_std=mean_std)
        return _create_dataloader(query_dataset, self.shots, self.n_ways, n_query=self.n_query)

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
        self.train_query_loader = _create_dataloader(self.train_query_dataset, self.n_query, self.train_n_ways, n_query=0)
        self.train_support_loader = _create_dataloader(self.train_support_dataset, self.shots, self.train_n_ways, n_query=0)

        self.val_loader = _create_dataloader(self.val_dataset, self.shots, self.n_ways, n_query=self.n_query)
        self.test_loader = _create_dataloader(self.test_dataset, self.shots, self.n_ways, n_query=self.n_query)

    def _update_train_iteration_classes(self, classes):
        self.train_query_dataset.sampler.set_sample_classes(classes)
        self.train_support_dataset.sampler.set_sample_classes(classes)

    def _reset_train_iteration_classes(self):
        self.train_query_dataset.sampler.reset_sample_classes()
        self.train_support_dataset.sampler.reset_sample_classes()

    def update_best_model(self, model, num_eval_episodes=None):
        val_loss, val_raw_acc, val_f1, val_spec, val_rec, val_prec = self.run_eval(model, self.val_loader, n_ways=self.train_n_ways, num_episodes=num_eval_episodes)
        val_acc = 2 * (val_spec * val_rec) /(val_spec + val_rec)
        print(f"Validation loss {val_loss} | Raw Accuracy {val_raw_acc} | Micro F1 {val_f1} | Accuracy H-Mean {val_acc} | Spec {val_spec} | Recall {val_rec} | Precision {val_prec}")
        if self.best_acc is None or val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model = copy.deepcopy(model)
            return True

        return False

    def run_train(self, num_episodes, lr=1e-5, min_lr=5e-7, lr_change_step=5, update_interval=20):
        model = self.model.to(self.device)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, cycle_momentum=False, step_size_up=lr_change_step)

        loss_meter = AverageMeter()
        f1_meter = AverageMeter()
        spec_meter = AverageMeter()
        f1_func = MultilabelF1Score(num_labels=self.train_n_ways, average='micro').to(self.device)
        spec_func = MultilabelSpecificity(num_labels=self.train_n_ways).to(self.device)

        cycler = ClassCycler(len(self.train_query_dataset.classes))
        support_iterator = DataloaderIterator(self.train_support_loader)
        query_iterator = DataloaderIterator(self.train_query_loader)
        best_epi = None
        for i in range(num_episodes):
            model.train()
            self._update_train_iteration_classes(cycler.next_n(self.train_n_ways))

            qimages, qclass_inds, train_class_labels = query_iterator.next_batch()
            qimages, qclass_inds = qimages.to(self.device), qclass_inds.to(self.device)
            simages, sclass_inds, support_class_labels = support_iterator.next_batch()
            assert train_class_labels == support_class_labels, f"Mismatch {train_class_labels} {support_class_labels}"
            simages, sclass_inds = simages.to(self.device), sclass_inds.to(self.device)

            optimizer.zero_grad()
            predictions, query_proto = model.update_support_and_classify(train_class_labels, simages, sclass_inds, qimages)
            loss = model.loss(query_proto, predictions, qclass_inds)

            f1 = f1_func(predictions, qclass_inds)
            f1_meter.update(f1.item(), qclass_inds.shape[0])

            spec = spec_func(predictions, qclass_inds)
            spec_meter.update(spec.item(), qclass_inds.shape[0])

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), len(qimages))
            print(f"Episode {i+1}: Loss {loss.item()} | F1 {f1} | Spec {spec}")
            
            if (i % update_interval) == (update_interval - 1):      
                if self.update_best_model(model, 30):
                    best_epi = i
                
                print(f"Average Training loss {loss_meter.average()} | Micro F1: {f1_meter.average()} | Spec {spec_meter.average()}")
                loss_meter.reset()
                f1_meter.reset()
                spec_meter.reset()
            
        self.model = model
        print('Best episode: ', best_epi+1)
    
    def run_eval(self, model, dataloader, verbose=False, n_ways=None, num_episodes=None):
        if n_ways is None:
            n_ways = self.n_ways

        model.eval()
        model = model.to(self.device)
        
        loss_meter = AverageMeter()
        prec_meter = AverageMeter()
        f1_meter = AverageMeter()
        spec_meter = AverageMeter()
        rec_meter = AverageMeter()
        acc_meter = AverageMeter()

        acc_func = MultilabelAccuracy(num_labels=n_ways).to(self.device)
        f1_func = MultilabelF1Score(num_labels=n_ways, average='micro').to(self.device)
        specificity_func = MultilabelSpecificity(num_labels=n_ways).to(self.device)
        recall_func = MultilabelRecall(num_labels=n_ways).to(self.device)
        prec_func = MultilabelPrecision(num_labels=n_ways).to(self.device)

        if num_episodes is None:
            num_episodes = len(dataloader)

        if verbose:
            ap_meter = AverageMeter()
            ap_func = MultilabelAveragePrecision(num_labels=n_ways).to(self.device)

        with torch.no_grad():
            iterator = DataloaderIterator(dataloader)
            for i in range(num_episodes):
                images, class_inds, class_labels = iterator.next_batch()
                s_size = self.shots * n_ways
                simages, qimages = images[:s_size,:,:].to(self.device), images[s_size:,:,:].to(self.device)
                sclass_inds, qclass_inds = class_inds[:s_size,:].to(self.device), class_inds[s_size:,:].to(self.device)
                predictions, query_proto = model.update_support_and_classify(class_labels, simages, sclass_inds, qimages)
                loss = model.loss(query_proto, predictions, qclass_inds)
                loss_meter.update(loss.item(), qclass_inds.shape[0])

                prec = prec_func(predictions, qclass_inds)
                prec_meter.update(prec.item(), qclass_inds.shape[0])
            
                f1 = f1_func(predictions, qclass_inds)
                f1_meter.update(f1.item(), qclass_inds.shape[0])

                spec = specificity_func(predictions, qclass_inds)
                spec_meter.update(spec.item(), qclass_inds.shape[0])
                rec = recall_func(predictions, qclass_inds)
                rec_meter.update(rec.item(), qclass_inds.shape[0])

                acc = acc_func(predictions, qclass_inds)
                acc_meter.update(acc.item(), qclass_inds.shape[0])

                if verbose:
                    # print(class_labels)
                    # print(torch.nonzero(class_inds)[:,1].bincount())
                    # print(predictions)
                    ap = ap_func(predictions, qclass_inds)
                    ap_meter.update(ap.item(), qclass_inds.shape[0])
                    print(f"Episode {i+1} | Loss {loss} | F1 {f1} | Specificity {spec} | Recall {rec} |  Precision {prec} | Bal Acc {(spec+rec)/2} | Raw Acc {acc} | AP {ap}")
        if verbose:
            return loss_meter.average(), acc_meter.average(), f1_meter.average(), spec_meter.average(), rec_meter.average(), prec_meter.average(), ap_meter.average()
        
        return loss_meter.average(), acc_meter.average(), f1_meter.average(), spec_meter.average(), rec_meter.average(), prec_meter.average()

    
class DynamicMetaTrainer(ControlledMetaTrainer):
    def __init__(self, model, shots, n_ways, dataset_config, train_n_ways=None, n_query=None, device='cpu'):
        super().__init__(model, shots, n_ways, dataset_config, train_n_ways=train_n_ways, n_query=n_query, device=device)

    def initialise_datasets(self, ds_config):
        img_info = ds_config.img_info
        img_path = ds_config.img_path
        classes_split_map = ds_config.classes_split_map
        label_names_map = ds_config.label_names_map
        mean_std = ds_config.mean_std

        query_image_ids, _ = get_query_and_support_ids(img_info, ds_config.training_split_path)
        self.query_image_ids = query_image_ids

        self.train_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'train', mean_std)
        self.val_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'val', mean_std)
        self.test_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'test', mean_std)

    def initialise_dataloaders(self):
        self.train_loader = _create_dataloader(self.train_dataset, self.shots, self.train_n_ways, n_query=self.n_query)
        self.val_loader = _create_dataloader(self.val_dataset, self.shots, self.n_ways, n_query=self.n_query)
        self.test_loader = _create_dataloader(self.test_dataset, self.shots, self.n_ways, n_query=self.n_query)

    def _update_train_iteration_classes(self, classes):
        self.train_dataset.sampler.set_sample_classes(classes)

    def _reset_train_iteration_classes(self):
        self.train_dataset.sampler.reset_sample_classes()

    def run_train(self, episodes, lr=1e-5, min_lr=5e-7, lr_change_step=5, update_interval=50, train_dataloader=None):
        if train_dataloader is None:
            train_dataloader = self.train_loader
        model = self.model.to(self.device)
        best_epi = None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, cycle_momentum=False, step_size_up=lr_change_step)

        f1_func = MultilabelF1Score(num_labels=self.train_n_ways, average='micro').to(self.device)
        spec_func = MultilabelSpecificity(num_labels=self.train_n_ways).to(self.device)
        rec_func = MultilabelRecall(num_labels=self.train_n_ways).to(self.device)

        cycler = ClassCycler(len(self.train_dataset.classes))
        loss_meter = AverageMeter()
        f1_meter = AverageMeter()
        spec_meter = AverageMeter()
        rec_meter = AverageMeter()

        n_ways = self.train_n_ways
        s_size = self.shots * n_ways
        tr_iterator = DataloaderIterator(train_dataloader)

        for i in range(episodes):
            model.train()

            self._update_train_iteration_classes(cycler.next_n(n_ways))
            
            images, class_inds, class_labels = tr_iterator.next_batch()
            simages, qimages = images[:s_size, :, :].to(self.device), images[s_size:, :, :].to(self.device)
            sclass_inds, qclass_inds = class_inds[:s_size, :].to(self.device), class_inds[s_size:, :].to(self.device)

            optimizer.zero_grad()
            predictions, query_proto = model.update_support_and_classify(class_labels, simages, sclass_inds, qimages)
            loss = model.loss(query_proto, predictions, qclass_inds)

            f1 = f1_func(predictions, qclass_inds)
            f1_meter.update(f1.item(), qclass_inds.shape[0])

            spec = spec_func(predictions, qclass_inds)
            spec_meter.update(spec.item(), qclass_inds.shape[0])

            rec = rec_func(predictions, qclass_inds)
            rec_meter.update(rec.item(), qclass_inds.shape[0])

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), len(qimages))
            print(f"Episode {i+1}: Loss {loss.item()} | Micro F1 {f1} | Spec {spec} | Recall {rec}")

            if (i % update_interval) == (update_interval-1):
                if self.update_best_model(model, 40):           
                    best_epi = i

                print(f"Average Training loss {loss_meter.average()} | Micro F1: {f1_meter.average()} | Spec {spec_meter.average()} | Recall {rec_meter.average()}")
                loss_meter.reset()
                f1_meter.reset()
                spec_meter.reset()
                rec_meter.reset()
            
            self._reset_train_iteration_classes()
            
        self.model = model
        if best_epi is not None:
            print('Best episode: ', best_epi+1)

class FTDynamicMetaTrainer(DynamicMetaTrainer):
    def __init__(self, model, shots, n_ways, dataset_config, train_n_ways=None, n_query=None, device='cpu'):
        super().__init__(model, shots, n_ways, dataset_config, train_n_ways=train_n_ways, n_query=n_query, device=device)

    def initialise_datasets(self, ds_config):
        img_info = ds_config.img_info
        img_path = ds_config.img_path
        classes_split_map = ds_config.classes_split_map
        label_names_map = ds_config.label_names_map
        mean_std = ds_config.mean_std

        query_image_ids, support_image_ids = get_query_and_support_ids(img_info, ds_config.training_split_path)
        self.query_image_ids = query_image_ids

        test_ft_image_ids, test_img_ids = get_query_and_support_ids(img_info, ds_config.test_split_path, split='test')

        self.train_dataset = MDataset(img_path, img_info, support_image_ids, label_names_map, classes_split_map['train'], mean_std=mean_std)
        self.val_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'val', mean_std)
        self.test_dataset = MDataset(img_path, img_info, test_img_ids, label_names_map, classes_split_map['test'], mean_std=mean_std)
        self.test_ft_dataset = MDataset(img_path, img_info, test_ft_image_ids, label_names_map, classes_split_map['test'], mean_std=mean_std)

    def initialise_dataloaders(self):
        self.train_loader = _create_dataloader(self.train_dataset, self.shots, self.train_n_ways, n_query=self.n_query)
        self.val_loader = _create_dataloader(self.val_dataset, self.shots, self.n_ways, n_query=self.n_query)
        self.test_loader = _create_dataloader(self.test_dataset, self.shots, self.n_ways, n_query=self.n_query)
        self.test_ft_loader = _create_dataloader(self.test_ft_dataset, self.shots, self.n_ways, n_query=self.n_query)

class MDynamicMetaTrainer(DynamicMetaTrainer):
    def __init__(self, model, shots, n_ways, dataset_config, train_n_ways=None, n_query=None, device='cpu'):
        super().__init__(model, shots, n_ways, dataset_config, train_n_ways=train_n_ways, n_query=n_query, device=device)

    def initialise_datasets(self, ds_config):
        img_info = ds_config.img_info
        img_path = ds_config.img_path
        classes_split_map = ds_config.classes_split_map
        label_names_map = ds_config.label_names_map
        mean_std = ds_config.mean_std

        query_image_ids, support_image_ids = get_query_and_support_ids(img_info, ds_config.training_split_path)
        self.query_image_ids = query_image_ids

        self.train_dataset = MDataset(img_path, img_info, support_image_ids, label_names_map, classes_split_map['train'], mean_std=mean_std)
        self.val_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'val', mean_std)
        self.test_dataset = _create_dataset(img_path, img_info, label_names_map, classes_split_map, 'test', mean_std)
    
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

def _create_dataloader(dataset, shots, n_ways, n_query):
    if dataset.sampler is None:
        sampler = ControlledFewShotBatchSampler(dataset.get_class_indicators(), shots, n_ways=n_ways, include_query=n_query!=0, n_query=n_query)
        dataset.sampler = sampler
    return DataLoader(dataset, batch_sampler=dataset.sampler, collate_fn=_collate_batch)
