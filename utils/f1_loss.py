import torch
import torch.nn.functional as F
from torch import nn

class F1Loss(nn.Module):
    '''Calculate F1 score.
    The original implmentation by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor `ndim` == 1.
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    '''
    def __init__(self, epsilon=1e-7, logits=True, spec_weight=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.logits = logits
        self.spec_weight = spec_weight
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 2
        if self.logits:
            y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        spec = tn/ (tn + fp + self.epsilon)
        return 1-f1.mean() + self.spec_weight*(1-spec.mean())
    

class BalAccuracyLoss(nn.Module):
    def __init__(self, epsilon=1e-7, logits=True, harmonic_mean=True):
        super().__init__()
        self.epsilon = epsilon
        self.logits = logits
        self.harmonic_mean = harmonic_mean
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 2
        if self.logits:
            y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        recall = tp / (tp + fn + self.epsilon)
        spec = tn/ (tn + fp + self.epsilon)

        if self.harmonic_mean:
            acc = 2*(recall*spec)/(recall+spec+self.epsilon)
        else:
            acc = (recall + spec)/2
        acc = acc.clamp(min=self.epsilon, max=1-self.epsilon)

        return 1 - acc.mean()