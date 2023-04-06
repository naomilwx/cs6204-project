import torch
from torch import nn

from models.metaclassifier.base import MetaModelBase, image_prototype_logits, laplace_dist_prob
from utils.f1_loss import BalAccuracyLoss

class ProtoNet(MetaModelBase):
    def __init__(self, imgtxt_encoder, class_prototype_aggregator, distance_func, scale=1.0, trainable_base=True, dist_prob=laplace_dist_prob):
        super(ProtoNet, self).__init__(imgtxt_encoder, class_prototype_aggregator)
        self.distance_func = distance_func
        self.scale = nn.Parameter(torch.tensor(scale))
        self.loss_fn = BalAccuracyLoss(logits=False)
        self.set_trainable(trainable_base, include_logit_scale=False)
        self.dist_prob = dist_prob
    
    def set_trainable(self, trainable, include_logit_scale=False):
        self.encoder.set_trainable(trainable, trainable, include_logit_scale=include_logit_scale)

    def get_scale(self):
        self.scale.data = torch.clamp(self.scale.data, 1e-5)
        return self.scale

    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=True)
        query_image_embeddings = query_image_embeddings.unsqueeze(1).repeat(1, self.class_prototypes.shape[0], 1)

        # Distances should be >= 0. Want to have probability increasing with distance, and map 0 to 1
        # probabilities = (-self.distance_func(self.class_prototypes, query_image_embeddings) * self.get_scale()).exp()
        probabilities = self.dist_prob(self.distance_func(self.class_prototypes, query_image_embeddings), self.get_scale())
        return probabilities, query_image_embeddings


class DotProtoNet(MetaModelBase):
    def __init__(self, imgtxt_encoder, class_prototype_aggregator, scale=1.0, trainable_base=True, bias=-1e-3):
        super(DotProtoNet, self).__init__(imgtxt_encoder, class_prototype_aggregator)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bias = nn.Parameter(torch.tensor(bias))
        self.loss_fn = BalAccuracyLoss(logits=True)
        self.set_trainable(trainable_base, include_logit_scale=False)

    def set_trainable(self, trainable, include_logit_scale=False):
        self.encoder.set_trainable(trainable, trainable, include_logit_scale=include_logit_scale)

    def get_scale(self):
        self.scale.data = torch.clamp(self.scale.data, 1e-5)
        return self.scale
    
    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=True) 
        query_image_embeddings = query_image_embeddings.unsqueeze(1).repeat(1, self.class_prototypes.shape[0], 1)

        logits = image_prototype_logits(self.class_prototypes, query_image_embeddings, self.get_scale()) + self.bias
        return logits, query_image_embeddings

        
    
class RelationNet(MetaModelBase):
    def __init__(self, imgtxt_encoder, embed_dim, class_prototype_aggregator, fc_hidden_size=16, activation=nn.ReLU, dropout=0.3, use_variance=False):
        super(RelationNet, self).__init__(imgtxt_encoder, class_prototype_aggregator)
        self.use_variance = use_variance
        if use_variance:
            self.cls = nn.Sequential(
                nn.Linear(embed_dim*3, fc_hidden_size),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_size, 1)
            )
        else:
            self.cls = nn.Sequential(
                nn.Linear(embed_dim*2, fc_hidden_size),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_size, 1)
            )
        self.encoder.set_trainable(False, False)

    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=True)
        query_image_embeddings = query_image_embeddings.unsqueeze(1).repeat(1, self.class_prototypes.shape[0], 1)
        class_prototypes = self.class_prototypes.repeat(query_image_embeddings.shape[0], 1, 1)

        if self.use_variance:
            class_prototypes_var = self.class_prototypes_var.repeat(query_image_embeddings.shape[0], 1, 1)

        if self.use_variance:
            out = self.cls(torch.cat((class_prototypes, class_prototypes_var, query_image_embeddings), dim=2))
        else:
            out = self.cls(torch.cat((class_prototypes, query_image_embeddings), dim=2))
       
        return out.squeeze(2), query_image_embeddings # NxLx1 -> NxL