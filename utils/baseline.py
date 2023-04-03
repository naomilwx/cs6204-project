import torch
from utils.metrics import AverageMeter, multilabel_accuracy
from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity, MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy

from models.attention.model import image_text_logits
from models.metaclassifier.trainer import DataloaderIterator

def run_baseline_eval(dataloader, num_labels, device, baseline_type='rand', episodes=50):
    prec_meter = AverageMeter()
    acc_meter = AverageMeter()
    spec_meter = AverageMeter()
    rec_meter = AverageMeter()
    f1_meter = AverageMeter()

    specificity = MultilabelSpecificity(num_labels=num_labels).to(device)
    recall = MultilabelRecall(num_labels=num_labels).to(device)
    precision = MultilabelPrecision(num_labels=num_labels).to(device)
    f1_func = MultilabelF1Score(num_labels=num_labels, average='micro').to(device)
    with torch.no_grad():
         iterator = DataloaderIterator(dataloader)
         for i in range(episodes):
                class_inds = iterator.next_batch()[1]
                class_inds = class_inds.to(device)

                if baseline_type == 'rand':
                    probs = torch.rand(class_inds.shape[0], num_labels, device=device)
                    acc = multilabel_accuracy(probs, class_inds)
                elif baseline_type == 'pos':
                    probs = torch.ones(class_inds.shape[0], num_labels, device=device)
                    acc = multilabel_accuracy(probs, class_inds)
                else:
                    probs = torch.zeros(class_inds.shape[0], num_labels, device=device)
                    acc = multilabel_accuracy(probs, class_inds)
                
                f1 = f1_func(probs, class_inds)
                f1_meter.update(f1.item(), len(class_inds))
            
                acc_meter.update(acc.item(), len(class_inds))

                spec = specificity(probs, class_inds)
                spec_meter.update(spec.item(), len(class_inds))
                rec = recall(probs, class_inds)
                rec_meter.update(rec.item(), len(class_inds))
                prec = precision(probs, class_inds)
                prec_meter.update(prec.item(), len(class_inds))
                print(f"Episode {i+1} | Raw Acc {acc} | F1 {f1} | Specificity {spec} | Recall {rec} | Precision {prec} | Bal Acc {(spec+rec)/2}")
            
    return acc_meter.average(), f1_meter.average(), spec_meter.average(), rec_meter.average(),  prec_meter.average()


def run_attention_zeroshot_eval(model, dataloader, num_labels, device,episodes=40):
    model.eval()
    model = model.to(device)
    
    acc_meter = AverageMeter()
    spec_meter = AverageMeter()
    rec_meter = AverageMeter()
    f1_meter = AverageMeter()
    prec_meter = AverageMeter()

    acc_func = MultilabelAccuracy(num_labels=num_labels).to(device)
    specificity = MultilabelSpecificity(num_labels=num_labels).to(device)
    recall = MultilabelRecall(num_labels=num_labels).to(device)
    f1_func = MultilabelF1Score(num_labels=num_labels, average='micro').to(device)
    prec_func = MultilabelPrecision(num_labels=num_labels).to(device)

    with torch.no_grad():
         iterator = DataloaderIterator(dataloader)
         for i in range(episodes):
                images, class_inds, class_labels = iterator.next_batch()
                images, class_inds = images.to(device), class_inds.to(device)

                text_embeddings, _, prototypes = model(class_labels, images)

                logits_per_image = image_text_logits(text_embeddings, prototypes, model.encoder.get_logit_scale())
                
                f1 = f1_func(logits_per_image, class_inds)
                f1_meter.update(f1.item(), len(class_inds))
            
                acc = acc_func(logits_per_image, class_inds)
                acc_meter.update(acc.item(), len(class_inds))

                spec = specificity(logits_per_image, class_inds)
                spec_meter.update(spec.item(), len(class_inds))
                rec = recall(logits_per_image, class_inds)
                rec_meter.update(rec.item(), len(class_inds))

                prec = prec_func(logits_per_image, class_inds)
                prec_meter.update(prec.item(), len(class_inds))

                print(f"Episode {i+1} | Raw Acc {acc} | F1 {f1} | Specificity {spec} | Recall {rec} | Precision {prec} | Bal Acc {(spec+rec)/2}")
                
    return acc_meter.average(), f1_meter.average(), spec_meter.average(), rec_meter.average(),  prec_meter.average()


def run_img_txt_zeroshot_eval(model, dataloader, num_labels, device, episodes=50):
    model.eval()
    model = model.to(device)
    
    acc_meter = AverageMeter()
    spec_meter = AverageMeter()
    rec_meter = AverageMeter()
    f1_meter = AverageMeter()
    prec_meter = AverageMeter()

    acc_func = MultilabelAccuracy(num_labels=num_labels).to(device)
    specificity = MultilabelSpecificity(num_labels=num_labels).to(device)
    recall = MultilabelRecall(num_labels=num_labels).to(device)
    f1_func = MultilabelF1Score(num_labels=num_labels, average='micro').to(device)
    prec_func = MultilabelPrecision(num_labels=num_labels).to(device)
    with torch.no_grad():
        iterator = DataloaderIterator(dataloader)
        for i in range(episodes):
            images, class_inds, class_labels = iterator.next_batch()
            images, class_inds = images.to(device), class_inds.to(device)
            text_embeddings, image_embeddings = model(class_labels, images, pool=True)

            _, logits_per_image = model.compute_logits(text_embeddings, image_embeddings)

            f1 = f1_func(logits_per_image, class_inds)
            f1_meter.update(f1.item(), len(class_inds))
        
            acc = acc_func(logits_per_image, class_inds)
            acc_meter.update(acc.item(), len(class_inds))

            spec = specificity(logits_per_image, class_inds)
            spec_meter.update(spec.item(), len(class_inds))
            rec = recall(logits_per_image, class_inds)
            rec_meter.update(rec.item(), len(class_inds))

            prec = prec_func(logits_per_image, class_inds)
            prec_meter.update(prec.item(), len(class_inds))
        
            print(f"Episode {i+1} | Raw Acc {acc} | F1 {f1} | Specificity {spec} | Recall {rec} | Precision {prec} | Bal Acc {(spec+rec)/2}")
            
    return acc_meter.average(), f1_meter.average(), spec_meter.average(), rec_meter.average(),  prec_meter.average()
