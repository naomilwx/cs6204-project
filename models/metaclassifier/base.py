import torch
from torch import nn
from abc import abstractmethod

def euclidean_distance(prototype, query):
    # prototype: (L, D) | query: (N, L, D)
    prototype = prototype.unsqueeze(0).expand(query.shape[0], -1, -1)
    return ((prototype-query)**2).sum(2)
    
def cosine_distance(prototype, query):
    prototype = prototype / prototype.norm(dim=-1, keepdim=True)
    prototype = prototype.unsqueeze(0).expand(query.shape[0], -1, -1)
    query = query / query.norm(dim=-1, keepdim=True)
    cos = prototype * query
    return -cos

class MetaModelBase(nn.Module):
    def __init__(self, imgtxt_encoder, class_prototype_aggregator):
        super().__init__()
        self.encoder = imgtxt_encoder
        self.class_prototype_aggregator = class_prototype_aggregator
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def set_class_prototype_details(self, class_labels, support_images, support_label_inds):
        image_embeddings = self.encoder.embed_image(support_images, pool=True) # (B, D)
        image_embeddings = image_embeddings.unsqueeze(1).expand(-1, support_label_inds.shape[1], -1)

        self.class_prototypes = self.class_prototype_aggregator(image_embeddings, support_label_inds)
    
    def update_support_and_classify(self, class_labels, support_images, support_label_inds, query_images):
        self.set_class_prototype_details(class_labels, support_images, support_label_inds)
        return self.forward(query_images)
    
    @abstractmethod
    def forward(self, query_images):
        return query_images
    
    def loss(self, predictions, label_inds):
        return self.loss_fn(predictions, label_inds.float())