import torch
from torch import nn

from models.metaclassifier.base import MetaModelBase
from utils.prototype import class_variance

class MetaModelWithAttention(MetaModelBase):
    def __init__(self, imgtxt_encoder, attn_model, class_prototype_aggregator, use_variance=False):
        super(MetaModelWithAttention, self).__init__(imgtxt_encoder, class_prototype_aggregator)
        self.attn_model = attn_model
        self.use_variance = use_variance
        self.freeze_child_models()

    def freeze_child_models(self):
        self.encoder.set_trainable(False, False, include_text_bb=False, include_logit_scale=False)
        self.attn_model.set_trainable(False)

    def set_class_prototype_details(self, class_labels, support_images, support_label_inds):
        text_embeddings, image_embeddings = self.encoder(class_labels, support_images, pool=False)
        self.class_label_embeddings = text_embeddings
        support_prototypes = self.attn_model(text_embeddings, image_embeddings, support_label_inds)
        self.class_prototypes = self.class_prototype_aggregator(support_prototypes, support_label_inds)
        if self.use_variance:
            self.class_prototypes_var = class_variance(support_prototypes, support_label_inds)

class ClsModel(MetaModelWithAttention):
    def __init__(self, imgtxt_encoder, attn_model, embed_dim, class_prototype_aggregator, fc_hidden_size=16, use_variance=False, activation=nn.ReLU, dropout=0.3) -> None:
        super(ClsModel, self).__init__(imgtxt_encoder, attn_model, class_prototype_aggregator, use_variance)
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
    
    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=False)
        query_prototypes = self.attn_model(self.class_label_embeddings, query_image_embeddings)

        # Prototypes: LxD (to repeat N times), variance: LxD (to repeat N times), query class prototype: NxLxD
        class_prototypes = self.class_prototypes.repeat(query_prototypes.shape[0], 1, 1)
        if self.use_variance:
            class_prototypes_var = self.class_prototypes_var.repeat(query_prototypes.shape[0], 1, 1)

        if self.use_variance:
            out = self.cls(torch.cat((class_prototypes, class_prototypes_var, query_prototypes), dim=2))
        else:
            out = self.cls(torch.cat((class_prototypes, query_prototypes), dim=2))
        return out.squeeze(2) # NxLx1 -> NxL
    
class ProtoNetAttention(MetaModelWithAttention):
    def __init__(self, imgtxt_encoder, attn_model, class_prototype_aggregator, distance_func, scale=1.0):
        super(ProtoNetAttention, self).__init__(imgtxt_encoder, attn_model, class_prototype_aggregator)
        self.distance_func = distance_func
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=False)
        query_prototypes = self.attn_model(self.class_label_embeddings, query_image_embeddings)

        return -self.distance_func(self.class_prototypes, query_prototypes) * self.scale