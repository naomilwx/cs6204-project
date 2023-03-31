import torch
from utils.metrics import AverageMeter, calculate_auc, multilabel_logit_accuracy
from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity, MultilabelPrecision, MultilabelF1Score

def run_baseline_eval(dataloader, num_labels, device, baseline_type='rand'):
    auc_meter = AverageMeter()
    acc_meter = AverageMeter()
    spec_meter = AverageMeter()
    rec_meter = AverageMeter()
    f1_meter = AverageMeter()

    specificity = MultilabelSpecificity(num_labels=num_labels).to(device)
    recall = MultilabelRecall(num_labels=num_labels).to(device)
    precision = MultilabelPrecision(num_labels=num_labels).to(device)
    f1_func = MultilabelF1Score(num_labels=num_labels).to(device)
    with torch.no_grad():
         for _, class_inds in dataloader:
                class_inds = class_inds.to(device)

                if baseline_type == 'rand':
                    probs = torch.rand(class_inds.shape[0], num_labels, device=device)
                elif baseline_type == 'pos':
                    probs = torch.ones(class_inds.shape[0], num_labels, device=device)
                else:
                    probs = torch.zeros(class_inds.shape[0], num_labels, device=device)
                
                f1 = f1_func(probs, class_inds)
                f1_meter.update(f1.item(), len(class_inds))

                auc = calculate_auc(probs, class_inds)
                auc_meter.update(auc, len(class_inds))
            
                acc = multilabel_logit_accuracy(probs, class_inds)
                acc_meter.update(acc, len(class_inds))

                spec = specificity(probs, class_inds)
                spec_meter.update(spec.item(), len(class_inds))
                rec = recall(probs, class_inds)
                rec_meter.update(rec.item(), len(class_inds))
                prec = precision(probs, class_inds)
                print(f"F1 {f1} | Accuracy {acc} | AUC {auc} | Specificity {spec} | Recall {rec} | Precision {prec}")
            
    return acc_meter.average(), f1_meter.average(), auc_meter.average(), spec_meter.average(), rec_meter.average()