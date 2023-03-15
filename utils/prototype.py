import torch

def squared_euclidean(v1, v2):
    # (D) or (L, D)
    return ((v1-v2)**2).sum(axis=-1)

def class_prototype_rrp(features):
    # features: (N, D) or (L, N, D)
    dims = len(features.shape)
    num_points = features.shape[-2]
    if num_points < 2:
        return features
    
    ftotal = features.sum(axis=-2)
    weights = 0 if dims == 2 else torch.zeros(features.shape[0])
    pc = torch.zeros(features.shape[1]) if dims == 2 else torch.zeros(features.shape[0], features.shape[2])
    for i in range(num_points):
        p = features[i] if dims == 2 else features[:,i]
        m = (ftotal - p)/(num_points - 1)
        a = 1 / squared_euclidean(p, m)
        pc += (p.t()*a).t()
        weights += a
    print(pc.shape, weights.shape)
    return pc.t().div(weights).t()

def class_prototype_inf(features):
    # features: (N, D) or (L, N, D)
    dims = len(features.shape)
    num_points = features.shape[-2]
    if num_points < 2:
        return features
    
    ftotal = features.sum(axis=-2)
    fmean = ftotal/num_points
    
    inf_total = 0 if dims == 2 else torch.zeros(features.shape[0])
    
    pc = torch.zeros(features.shape[1]) if dims == 2 else torch.zeros(features.shape[0], features.shape[2])
    for i in range(num_points):
        p = features[i] if dims == 2 else features[:,i]
        mmd = (ftotal - p)/(num_points - 1) - fmean
        inf = 1 - mmd.norm(dim=-1)
        inf_total += inf
        pc += (p.t()*inf).t()
    return pc.t().div(inf_total).t()

def class_prototype_mean(features):
    # features: (N, D) or (L, N, D)
    return features.mean(axis=-2)

def class_variance(features):
    # features: (N, D) or (L, N, D)
    return features.var(axis=-2)