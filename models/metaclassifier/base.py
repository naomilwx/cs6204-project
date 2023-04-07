import torch
from torch import nn
from abc import abstractmethod

from utils.prototype import class_variance
from utils.f1_loss import BalAccuracyLoss

def euclidean_distance(prototype, query):
    # prototype: (L, D) | query: (N, L, D)
    prototype = prototype.unsqueeze(0).expand(query.shape[0], -1, -1)
    # out: (N, L)
    return ((prototype-query)**2).sum(2)

def cosine_similarity(prototype, query):
    # prototype: (L, D) | query: (N, L, D)
    prototype = prototype / prototype.norm(dim=-1, keepdim=True)
    prototype = prototype.unsqueeze(0).expand(query.shape[0], -1, -1)

    query = query / query.norm(dim=-1, keepdim=True)
    cos = (prototype * query).sum(2)
    return cos
    
def cosine_distance(prototype, query):
    # prototype: (L, D) | query: (N, L, D)
    # should return a value in [0, 2]
    return 1 - cosine_similarity(prototype, query)

def image_prototype_logits(prototypes, image_embeddings, scale=1):
    # prototypes: (L, D) x image_embeddings: (N, L, D) -> (N, L)
    # fac = prototypes.unsqueeze(0).expand_as(image_embeddings)
    # (fac * image_embeddings).sum(axis=2)
    return cosine_similarity(prototypes, image_embeddings) * scale

def laplace_dist_prob(dist, scale):
    return 1 / ((dist * scale).exp())

def gaussian_dist_prob(dist, scale):
    v = (dist*scale)**2
    return 1/v.exp()


class MetaModelBase(nn.Module):
    def __init__(self, imgtxt_encoder, class_prototype_aggregator):
        super().__init__()
        self.encoder = imgtxt_encoder
        self.class_prototype_aggregator = class_prototype_aggregator
        self.loss_fn = BalAccuracyLoss(logits=True)
        self.use_variance = False
        self.other_loss_weight = 0

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn
    
    def set_class_prototype_details(self, class_labels, support_images, support_label_inds):
        image_embeddings = self.encoder.embed_image(support_images, pool=True) # (B, D)
        image_embeddings = image_embeddings.unsqueeze(1).repeat(1, support_label_inds.shape[1], 1)

        self.class_prototypes = self.class_prototype_aggregator(image_embeddings, support_label_inds)
        if self.use_variance:
            self.class_prototypes_var = class_variance(image_embeddings, support_label_inds)
    
    def update_support_and_classify(self, class_labels, support_images, support_label_inds, query_images):
        self.set_class_prototype_details(class_labels, support_images, support_label_inds)
        return self.forward(query_images)
    
    @abstractmethod
    def forward(self, query_images):
        return query_images, query_images
    
    def loss(self, query_prototypes, predictions, label_inds):
        return self.loss_fn(predictions, label_inds.float())