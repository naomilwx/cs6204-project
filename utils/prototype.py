import torch

def squared_euclidean(v1, v2):
    # (D) or (L, D)
    return ((v1-v2)**2).sum(axis=-1)

def class_prototype_rrp(features, label_inds=None):
    # features: (N, D) or (N, L, D) -> transform to (L, N, D)
    # label_inds: (N, L)
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
    else:
        features = features.permute(1,0,2) # (L, N, D)
    
    classes_count = features.shape[-2]
    if classes_count < 2:
        return features
    
    if label_inds is not None:
        classes_count = torch.nonzero(label_inds)[:,1].bincount()
        # mask: (L, N)
        mask = label_inds.t()
        features = mask.unsqueeze(-1).expand_as(features)*features
    ftotal = features.sum(axis=-2)

    weights = torch.zeros(features.shape[0])
    pc = torch.zeros(features.shape[0], features.shape[2])
    for i in range(features.shape[-2]):
        p = features[:,i]

        if label_inds is not None:
            div = (classes_count - 1).clamp(min=1)
            m = (ftotal - p).div(div.unsqueeze(-1).expand_as(p))
            a = mask[:, i]*(1 / squared_euclidean(p, m))
        else:
            m = (ftotal - p)/(classes_count - 1)
            a = 1 / squared_euclidean(p, m)
        pc += a.unsqueeze(-1).expand_as(p) * p
        weights += a
    return pc.t().div(weights).t()

def class_prototype_inf(features, label_inds=None):
    # features: (N, D) or (N, L, D) -> transform to (L, N, D)
    # label_inds: (N, L)
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
    else:
        features = features.permute(1,0,2)

    classes_count = features.shape[-2]
    if classes_count < 2:
        return features

    if label_inds is not None:
        classes_count = torch.nonzero(label_inds)[:,1].bincount()
        # mask: (L, N)
        mask = label_inds.t()
        features = mask.unsqueeze(-1).expand_as(features)*features
        ftotal = features.sum(axis=-2)
        fmean = ftotal.div(classes_count.unsqueeze(-1).expand_as(ftotal))
    else:
        ftotal = features.sum(axis=-2)
        fmean = ftotal/classes_count

    inf_total = torch.zeros(features.shape[0])
    pc = torch.zeros(features.shape[0], features.shape[2])
    for i in range(features.shape[-2]):
        p = features[:,i]
        if label_inds is not None:
            div = (classes_count - 1).clamp(min=1)
            mmd = (ftotal - p).div(div.unsqueeze(-1).expand_as(p)) - fmean
            inf = mask[:, i]*(1 - mmd.norm(dim=-1))
        else:
            mmd = (ftotal - p) / (classes_count - 1) - fmean
            inf = 1 - mmd.norm(dim=-1)
        inf_total += inf
        pc += inf.unsqueeze(-1).expand_as(p) * p
    return pc.t().div(inf_total).t()

def class_prototype_mean(features, label_inds=None):
    # features: (N, D) or (N, L, D)
    # label_inds: (N, L)

    if label_inds is None:
        return features.mean(axis=0)
    
    classes_count = torch.nonzero(label_inds)[:,1].bincount()
    ftotal = (label_inds.unsqueeze(-1).expand_as(features)*features).sum(axis=0)

    return ftotal.div(classes_count.unsqueeze(-1).expand_as(ftotal))

def class_variance(features, label_inds=None):
    # features: (N, D) or (N, L, D)
    # label_inds: (N, L)
    if label_inds is None:
        return features.var(axis=0)
    
    mean = class_prototype_mean(features, label_inds).unsqueeze(0)

    classes_count = torch.nonzero(label_inds)[:,1].bincount()

    total = ((features - mean) ** 2)
    total = label_inds.unsqueeze(-1).expand_as(total)*total
    total = total.sum(axis=0) # D or (L, D)
    return total.div((classes_count - 1).unsqueeze(-1).expand_as(total))