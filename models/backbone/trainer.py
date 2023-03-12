import torch
from torch import nn
import torch.utils.data as data
# from tqdm import tqdm
import copy

import medmnist
from medmnist import Evaluator
from medmnist.evaluator import getACC, getAUC

from utils.sampling import MultilabelBalancedRandomSampler, create_single_class_sampler

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