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
        self.best_acc = None
    
    def run_train(self, epochs, dataloader, val_dataloader, lr=1e-4, min_lr=1e-6, full_training=False, encoder_only=False, enc_weight=0.5):
        model = self.model.to(self.device)
        best_epoch = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if full_training or encoder_only:
            model.unfreeze_encoder()
        else:
            model.freeze_encoder()

        if encoder_only:
            model.attention.set_trainable(False)
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, cycle_momentum=False, step_size_up=10)

        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            for i, (images, class_inds) in enumerate(dataloader):
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                optimizer.zero_grad()

                # text_embeddings, _, prototypes = model(self.class_labels, images, class_inds)
                text_embeddings, _, prototypes = model(self.class_labels, images)
                if encoder_only:
                    loss =  model.encoder_loss(text_embeddings, prototypes, class_inds)
                else:
                    # loss = 0.5 * model.attention_loss(text_embeddings, prototypes, class_inds)
                    loss = model.attention_loss(text_embeddings, prototypes, class_inds)
                
                if full_training:
                    loss += enc_weight * model.encoder_loss(text_embeddings, prototypes, class_inds)

                # if not encoder_only:
                #     te, _, ptyps = model(self.class_labels, images)
                #     loss += 0.5 * model.attention_loss(te, ptyps, class_inds)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.item(), len(class_inds))
                print(f"Batch {i+1}: loss {loss.item()}")
            
            print(f"Epoch {epoch+1}: Training loss {loss_meter.average()}")

            val_loss, val_racc, val_auc, val_spec, val_rec = self.run_eval(model, val_dataloader, full_training=full_training, encoder_only=encoder_only, enc_weight=enc_weight)
            val_acc =  2 * (val_spec * val_rec) /(val_spec + val_rec)
            print(f"Epoch {epoch+1}: Validation loss {val_loss} | Accuracy {val_racc} | AUC {val_auc} | Acc H-Mean {val_acc} | Spec {val_spec} | Recall {val_rec}")

            if self.best_acc is None or val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model = copy.deepcopy(model)
                best_epoch = epoch
        self.model = model
        print('Best epoch: ', best_epoch+1)

    def run_eval(self, model, dataloader, full_training=False, verbose=False, encoder_only=False, enc_weight=0.5):
        model.eval()
        model = model.to(self.device)
        
        loss_meter = AverageMeter()
        auc_meter = AverageMeter()
        acc_meter = AverageMeter()
        spec_meter = AverageMeter()
        rec_meter = AverageMeter()

        specificity = MultilabelSpecificity(num_labels=len(self.class_labels)).to(self.device)
        recall = MultilabelRecall(num_labels=len(self.class_labels)).to(self.device)
        with torch.no_grad():
            for images, class_inds in dataloader:
                images, class_inds = images.to(self.device), class_inds.to(self.device)
                text_embeddings, _, prototypes = model(self.class_labels, images)

                if encoder_only:
                    loss = model.encoder_loss(text_embeddings, prototypes, class_inds).item()
                else:
                    loss = model.attention_loss(text_embeddings, prototypes, class_inds).item()

                logits_per_image = image_text_logits(text_embeddings, prototypes, model.encoder.get_logit_scale())
                if full_training:
                    loss += enc_weight * model.encoder.contrastive_logit_loss(logits_per_image.t(), logits_per_image, class_inds).item()
        
                loss_meter.update(loss, len(class_inds))

                auc = calculate_auc(logits_per_image, class_inds)
                auc_meter.update(auc, len(class_inds))
            
                acc = multilabel_logit_accuracy(logits_per_image, class_inds)
                acc_meter.update(acc, len(class_inds))

                spec = specificity(logits_per_image, class_inds)
                spec_meter.update(spec.item(), len(class_inds))
                
                rec = recall(logits_per_image, class_inds)
                rec_meter.update(rec.item(), len(class_inds))

                if verbose:
                    print(f"Loss {loss} | Accuracy {acc} | AUC {auc} | Specificity {spec} | Recall {rec}")

        return loss_meter.average(),  acc_meter.average(), auc_meter.average(), spec_meter.average(), rec_meter.average()

