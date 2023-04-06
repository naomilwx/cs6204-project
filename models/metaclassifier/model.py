import torch
from torch import nn

from models.metaclassifier.base import MetaModelBase, image_prototype_logits, laplace_dist_prob
from utils.prototype import class_variance
from utils.f1_loss import BalAccuracyLoss

class MetaModelWithAttention(MetaModelBase):
    def __init__(self, imgtxt_encoder, attn_model, class_prototype_aggregator, use_variance=False):
        super(MetaModelWithAttention, self).__init__(imgtxt_encoder, class_prototype_aggregator)
        self.attn_model = attn_model
        self.use_variance = use_variance
        self.freeze_child_models()

    def loss(self, query_prototypes, predictions, label_inds):
        attn_loss = 0
        if self.attention_loss_weight != 0:
            # attn_loss = self.attn_model.loss(self.class_label_embeddings, query_prototypes, label_inds)
            attn_loss = self.encoding_space_loss(self.class_label_embeddings, query_prototypes, label_inds)
        return super().loss(query_prototypes, predictions, label_inds) + self.attention_loss_weight * attn_loss
    
    def encoding_space_loss(self, text_embeddings, prototypes, label_inds):
        classes = torch.nonzero(label_inds)[:,1] # (Np,)
        class_prototypes = prototypes[label_inds.bool()] # (Np, D)

        labels = nn.functional.one_hot(classes)
        return self.encoder.loss(text_embeddings, class_prototypes, labels)

    def freeze_child_models(self):
        self.encoder.set_trainable(False, False, include_text_bb=False, include_logit_scale=False)
        self.set_attention_model_trainable(False)

    def set_attention_model_trainable(self, trainable, weight = 0.1):
        self.attn_model.set_trainable(trainable)
        if trainable:
            self.attention_loss_weight = weight
        else:
            self.attention_loss_weight = 0
    
    def set_class_prototype_details(self, class_labels, support_images, support_label_inds):
        text_embeddings, image_embeddings = self.encoder(class_labels, support_images, pool=False)
        # support_prototypes = self.attn_model(text_embeddings, image_embeddings, support_label_inds)
        support_prototypes = self.attn_model(text_embeddings, image_embeddings)

        self.class_label_embeddings = text_embeddings
        # print(support_prototypes.shape, support_label_inds.shape)
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
        return out.squeeze(2), query_prototypes # NxLx1 -> NxL
    
class ProtoNetAttention(MetaModelWithAttention):
    def __init__(self, imgtxt_encoder, attn_model, class_prototype_aggregator, distance_func, scale=1.0, dist_prob=laplace_dist_prob):
        super(ProtoNetAttention, self).__init__(imgtxt_encoder, attn_model, class_prototype_aggregator)
        self.distance_func = distance_func
        self.scale = nn.Parameter(torch.tensor(scale))
        self.loss_fn = BalAccuracyLoss(logits=False)
        self.dist_prob = dist_prob

    def reset_scale(self):
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def get_scale(self):
        self.scale.data = torch.clamp(self.scale.data, 1e-5)
        return self.scale

    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=False)
        query_prototypes = self.attn_model(self.class_label_embeddings, query_image_embeddings)
        # probabilities = (-self.distance_func(self.class_prototypes, query_prototypes) * self.get_scale()).exp()
        probabilities = self.dist_prob(self.distance_func(self.class_prototypes, query_prototypes), self.get_scale())


        return probabilities, query_prototypes
    
class DotProtoNetAttention(MetaModelWithAttention):
    def __init__(self, imgtxt_encoder, attn_model, class_prototype_aggregator, scale=1.0, bias=-1e-3):
        super(DotProtoNetAttention, self).__init__(imgtxt_encoder, attn_model, class_prototype_aggregator)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bias = nn.Parameter(torch.tensor(bias))
        self.loss_fn = BalAccuracyLoss(logits=True)

    def get_scale(self):
        self.scale.data = torch.clamp(self.scale.data, 1e-5)
        return self.scale

    def forward(self, query_images):
        query_image_embeddings = self.encoder.embed_image(query_images, pool=False)
        query_prototypes = self.attn_model(self.class_label_embeddings, query_image_embeddings)
        logits = image_prototype_logits(self.class_prototypes, query_prototypes, self.get_scale()) + self.bias

        return logits, query_prototypes
