import torch
from torch import nn
import torch.utils.data as data
# from tqdm import tqdm
import copy

import medmnist
from medmnist import Evaluator
from medmnist.evaluator import getACC, getAUC

from utils.sampling import MultilabelBalancedRandomSampler, create_single_class_sampler
from utils.metrics import AverageMeter, calculate_auc, multilabel_logit_accuracy
from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity

class Trainer:
  def __init__(self, model, datasets, batch_size, device='cpu', balance=False):
      # TODO: check label distribution and do som rebalancing
      self.datasets = datasets
      self.device = device
      self.model = model

      sampler = None
      if balance == True:
        train_labels = Evaluator(datasets.dataname, 'train').labels
        if datasets.task == "multi-label, binary-class":
          sampler = MultilabelBalancedRandomSampler(train_labels)
        else:
          sampler = create_single_class_sampler(train_labels.flatten())
      
      self.train_loader = data.DataLoader(dataset=datasets.train_dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
      self.test_loader = data.DataLoader(dataset=datasets.test_dataset, batch_size=batch_size, shuffle=True)
      self.val_loader = data.DataLoader(dataset=datasets.val_dataset, batch_size=batch_size, shuffle=True)
      if datasets.task == "multi-label, binary-class":
        self.criterion = nn.BCEWithLogitsLoss()
      else:
        self.criterion = nn.CrossEntropyLoss()
  
  def run_train(self, epochs, lr=1e-4):
    model = self.model.to(self.device)
    best_acc = None
    best_epoch = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
      model.train()
      accuracy = 0
      total = 0
      total_loss = 0
      for i, (inputs, targets) in enumerate(self.train_loader):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if self.datasets.task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()
        
        loss = self.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = getACC(targets.cpu(), outputs.detach().cpu(), self.datasets.task)
        total += len(targets)
        accuracy += len(targets) * acc
        total_loss += len(targets) * loss.item()
        print(f"Batch {i+1}: loss {total_loss/total}")
        
      accuracy /= total
      print(f"Epoch {epoch+1}: Training loss {total_loss/total} | Accuracy {accuracy}")
      val_acc, val_auc, val_loss = self.run_eval(model, self.val_loader)
      print(f"Epoch {epoch+1}: Validation loss {val_loss} | Accuracy {val_acc} | AUC {val_auc}")

      if best_acc is None or val_acc > best_acc:
        best_acc = val_acc
        self.best_model = copy.deepcopy(model)
        best_epoch = epoch
    print('Best epoch: ', best_epoch+1)


  def run_eval(self, model, dataloader):
    model.eval()
    model = model.to(self.device)
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    avg_loss = 0
    total = 0
    with torch.no_grad():
      for inputs, targets in dataloader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = model(inputs)
        if self.datasets.task == 'multi-label, binary-class':
          targets = targets.to(torch.float32)
          outputs = outputs.softmax(dim=-1)
        else:
          targets = targets.squeeze().long()
          outputs = outputs.softmax(dim=-1)
        
        loss = self.criterion(outputs, targets)
        avg_loss += loss.item() * len(inputs)
        total += len(inputs)
        y_true = torch.cat((y_true, targets.cpu()), 0)
        y_score = torch.cat((y_score, outputs.cpu()), 0)
      
      y_true = y_true.numpy()
      y_score = y_score.cpu().numpy()
      acc = getACC(y_true, y_score, self.datasets.task)
      auc = getAUC(y_true, y_score, self.datasets.task)
      avg_loss /= total

      return acc, auc, avg_loss

class DSTrainer:
    def __init__(self, model, class_labels, device='cpu'):
        self.model = model
        self.device = device
        self.class_labels = class_labels
    
    def run_train(self, epochs, dataloader, val_dataloader, lr=1e-4, min_lr=1e-6, weight_decay=1e-5):
        model = self.model.to(self.device)
        best_epoch = None
        best_acc = None
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, cycle_momentum=False, step_size_up=10)
        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            for i, (images, class_inds) in enumerate(dataloader):
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                optimizer.zero_grad()

                predictions = model(images)
                loss = criterion(predictions, class_inds.float())
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.item(), len(class_inds))
                print(f"Batch {i+1}: loss {loss_meter.average()}")
            
            print(f"Epoch {epoch+1}: Training loss {loss_meter.average()}")
            val_acc, val_auc, val_loss = self.run_eval(model, val_dataloader)
            print(f"Epoch {epoch+1}: Validation loss {val_loss} | Accuracy {val_acc} | AUC {val_auc}")
            
            if best_acc is None or val_acc > best_acc:
                best_acc = val_acc
                self.best_model = copy.deepcopy(model)
                best_epoch = epoch
        self.model = model
        print('Best epoch: ', best_epoch+1)

    def run_eval(self, model, dataloader, additional_stats=False):
        model.eval()
        model = model.to(self.device)

        criterion = nn.BCEWithLogitsLoss()

        loss_meter = AverageMeter()
        auc_meter = AverageMeter()
        acc_meter = AverageMeter()

        if additional_stats:
            specificity = MultilabelSpecificity(num_labels=len(self.class_labels)).to(self.device)
            spec_meter = AverageMeter()
            recall = MultilabelRecall(num_labels=len(self.class_labels)).to(self.device)
            rec_meter = AverageMeter()
        with torch.no_grad():
            for images, class_inds in dataloader:
                images, class_inds = images.to(self.device), class_inds.to(self.device)

                predictions = model(images)
                loss = criterion(predictions, class_inds.float())

                loss_meter.update(loss.item(), len(class_inds))

                auc = calculate_auc(predictions, class_inds)
                auc_meter.update(auc, len(class_inds))
                
                acc = multilabel_logit_accuracy(predictions, class_inds)
                acc_meter.update(acc, len(class_inds))

                if additional_stats:
                    spec = specificity(predictions, class_inds)
                    spec_meter.update(spec.item(), len(class_inds))
                    rec = recall(predictions, class_inds)
                    rec_meter.update(rec.item(), len(class_inds))
                    print(f"Loss {loss} | Accuracy {acc} | AUC {auc} | Specificity {spec} | Recall {rec}")
        if additional_stats:
            return acc_meter.average(), auc_meter.average(), loss_meter.average(), spec_meter.average(), rec_meter.average()
        return acc_meter.average(), auc_meter.average(), loss_meter.average()