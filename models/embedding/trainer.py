import torch
import copy
from utils.metrics import AverageMeter, calculate_auc, multilabel_logit_accuracy
from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity

class Trainer:
    def __init__(self, model, class_labels, device='cpu'):
        self.model = model
        self.device = device
        self.class_labels = class_labels
    
    def run_train(self, epochs, dataloader, val_dataloader, lr=1e-4, min_lr=1e-6):
        model = self.model.to(self.device)
        best_epoch = None
        best_acc = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, cycle_momentum=False, step_size_up=10)
        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            for i, (images, class_inds) in enumerate(dataloader):
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                optimizer.zero_grad()

                text_embeddings, image_embeddings = model(self.class_labels, images, pool=True)
                loss = model.loss(text_embeddings, image_embeddings, class_inds)
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
                text_embeddings, image_embeddings = model(self.class_labels, images, pool=True)

                logits_per_text, logits_per_image = model.compute_logits(text_embeddings, image_embeddings)
                loss = model.contrastive_logit_loss(logits_per_text, logits_per_image, class_inds)
                loss_meter.update(loss.item(), len(class_inds))

                auc = calculate_auc(logits_per_image, class_inds)
                auc_meter.update(auc, len(class_inds))
                
                acc = multilabel_logit_accuracy(logits_per_image, class_inds)
                acc_meter.update(acc, len(class_inds))

                if additional_stats:
                    spec = specificity(logits_per_image, class_inds)
                    spec_meter.update(spec.item(), len(class_inds))
                    rec = recall(logits_per_image, class_inds)
                    rec_meter.update(rec.item(), len(class_inds))
                    print(f"Loss {loss} | Accuracy {acc} | AUC {auc} | Specificity {spec} | Recall {rec}")
        if additional_stats:
            return acc_meter.average(), auc_meter.average(), loss_meter.average(), spec_meter.average(), rec_meter.average()
        return acc_meter.average(), auc_meter.average(), loss_meter.average()