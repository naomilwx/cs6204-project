import copy
from utils.metrics import AverageMeter, calculate_auc, multilabel_accuracy
import torch

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

class MetaTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def run_train(self, epochs, query_dataloader, support_dataloader, val_dataloader, train_class_labels, val_class_labels, lr=1e-5):
        model = self.model.to(self.device)
        best_epoch = None
        best_acc = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        support_iterator = DataloaderIterator(support_dataloader)
        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            for i, (qimages, qclass_inds) in enumerate(query_dataloader):
                qimages, qclass_inds = qimages.to(self.device), qclass_inds.to(self.device)
                simages, sclass_inds = support_iterator.next_batch()
                simages, sclass_inds = simages.to(self.device), sclass_inds.to(self.device)

                optimizer.zero_grad()
                predictions = model.update_support_and_classify(train_class_labels, simages, sclass_inds, qimages)
                loss = model.loss(predictions, qclass_inds)

                acc = multilabel_accuracy(predictions, qclass_inds)
                acc_meter.update(acc, qclass_inds.shape[0])

                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), len(qimages))
                print(f"Batch {i+1}: loss {loss_meter.average()} | Acc {acc}")
            
            print(f"Epoch {epoch+1}: Training loss {loss_meter.average()} | Acc: {acc_meter.average()}")

            val_acc, val_auc, val_loss = self.run_eval(model, val_dataloader, val_class_labels)
            print(f"Epoch {epoch+1}: Validation loss {val_loss} | Accuracy {val_acc} | AUC {val_auc}")

            if best_acc is None or val_acc > best_acc:
                best_acc = val_acc
                self.best_model = copy.deepcopy(model)
                best_epoch = epoch
        self.model = model
        print('Best epoch: ', best_epoch+1)
    
    def run_eval(self, model, val_dataloader, val_class_labels):
        model.eval()
        model = model.to(self.device)
        
        loss_meter = AverageMeter()
        auc_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            for images, class_inds in val_dataloader:
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                shots = images.shape[0]//2
                qimages, simages = images[:shots,:,:], images[shots:,:,:]
                qclass_inds, sclass_inds = class_inds[:shots,:], class_inds[shots:,:]

                predictions = model.update_support_and_classify(val_class_labels, simages, sclass_inds, qimages)
                loss = model.loss(predictions, qclass_inds)

                loss_meter.update(loss.item(), shots)

                auc = calculate_auc(predictions, qclass_inds)
                auc_meter.update(auc, shots)
            
                acc = multilabel_accuracy(predictions, qclass_inds)
                acc_meter.update(acc, shots)
        return acc_meter.average(), auc_meter.average(), loss_meter.average()