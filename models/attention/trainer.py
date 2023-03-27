import torch
import copy
from utils.metrics import AverageMeter, calculate_auc, multilabel_logit_accuracy
from models.attention.model import image_text_logits
from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity

class Trainer:
    def __init__(self, model, class_labels, device='cpu'):
        self.model = model
        self.device = device
        self.class_labels = class_labels
    
    def run_train(self, epochs, dataloader, val_dataloader, lr=1e-4, full_training=False):
        model = self.model.to(self.device)
        best_epoch = None
        best_loss = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if full_training:
            model.unfreeze_encoder()
        else:
            model.freeze_encoder()
        
        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            for i, (images, class_inds) in enumerate(dataloader):
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                optimizer.zero_grad()

                text_embeddings, _, prototypes = model(self.class_labels, images, class_inds)
                loss = model.attention_loss(text_embeddings, prototypes, class_inds)
                if full_training:
                    logits_per_image = image_text_logits(text_embeddings, prototypes, model.encoder.get_logit_scale())
                    loss += 0.5*model.encoder.contrastive_logit_loss(logits_per_image.t(), logits_per_image, class_inds)
                
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), len(class_inds))
                print(f"Batch {i+1}: loss {loss_meter.average()}")
            
            print(f"Epoch {epoch+1}: Training loss {loss_meter.average()}")

            val_acc, val_auc, val_loss = self.run_eval(model, val_dataloader)
            print(f"Epoch {epoch+1}: Validation loss {val_loss} | Accuracy {val_acc} | AUC {val_auc}")

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                self.best_model = copy.deepcopy(model)
                best_epoch = epoch
        self.model = model
        print('Best epoch: ', best_epoch+1)

    def run_eval(self, model, dataloader, full_training=False, additional_stats=False):
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
                text_embeddings, _, prototypes = model(self.class_labels, images, class_inds)

                logits_per_image = image_text_logits(text_embeddings, prototypes, model.encoder.get_logit_scale())
                loss = model.attention.contrastive_loss(prototypes, class_inds).item()
                loss += model.attention.classification_loss(logits_per_image, class_inds).item()
                if full_training:
                    loss += 0.5*model.encoder.contrastive_logit_loss(logits_per_image.t(), logits_per_image, class_inds).item()
        
                loss_meter.update(loss, len(class_inds))

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