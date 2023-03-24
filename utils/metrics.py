# adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py
from sklearn.metrics import roc_auc_score
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def average(self):
        return self.sum / self.count

def accuracy(output, target, topk=(1,)):
    if len(output.size()) == 1:
        output = output.unsqueeze(1)
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) / batch_size for k in topk]

def multilabel_logit_accuracy(output, target):
    # Returns the average accuracy across classes    
    # Target is 1 at indices where class label is true
    # Output should be higher at indices where class label is true
    pred = output.sigmoid().round()
    correct = pred.eq(target)
    return correct.float().sum().item()/target.numel()

def calculate_auc(y_score, y_true):
    '''AUC metric.
    :param y_score: the predicted score of each class,
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    y_score = y_score.squeeze().cpu()
    y_true = y_true.squeeze().cpu().numpy().astype(int)

    auc = 0
    count = 0
    for i in range(y_score.shape[1]):
        y_true_i = y_true[:, i]
        if len(np.unique(y_true_i)) < 2:
            continue
        label_auc = roc_auc_score(y_true_i, y_score[:, i])
        auc += label_auc
        count += 1
    ret = auc / count

    return ret