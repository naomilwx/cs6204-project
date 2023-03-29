import torch
from torch import nn
from utils.contrastive_loss import SupConLoss
from abc import abstractmethod

def image_text_logits(text_embeddings, prototypes, scale=1):
    # text_embeddings: (14, 512) x prototypes: (140, 14, 512) -> (140, 14)
    fac = text_embeddings.unsqueeze(0).expand_as(prototypes)

    return (fac * prototypes).sum(axis=2) * scale
    
class BaseLabelImageAttention(nn.Module):
    def __init__(self,  temperature=1, cls_weight=1, cls_loss=nn.CrossEntropyLoss()) -> None:
        super().__init__()
        self.con_loss = SupConLoss(temperature=temperature, contrast_mode='one')
        self.class_loss = cls_loss
        self.cls_weight = cls_weight

    @abstractmethod
    def forward(self, text_embedings, images, label_inds=None):
        pass

    def set_trainable(self, trainable):
        for param in self.attn.parameters():
            param.requires_grad = trainable

    def image_text_logits(self, text_embedings, images, label_inds=None):
        prototypes = self.forward(text_embedings, images, label_inds)
        return image_text_logits(text_embedings, prototypes)
    
    def loss(self, text_embeddings, prototypes, label_inds):
        logits = image_text_logits(text_embeddings, prototypes)
        cls_loss = 0
        if self.cls_weight != 0:
            cls_loss = self.classification_loss(logits, label_inds)
        return self.contrastive_loss(prototypes, label_inds) + self.cls_weight * cls_loss
    
    def classification_loss(self, logits, label_inds):
        return self.class_loss(logits, label_inds.float())
    
    def contrastive_loss(self, results, label_inds):
        # results: (N, L, D), labels: (N, L)
        classes = torch.nonzero(label_inds)[:,1] # (Np,)
        prototypes = results[label_inds.bool()] # (Np, D)
        return self.con_loss(prototypes.unsqueeze(1), classes)
    
class LabelImageMHAttention(BaseLabelImageAttention):
    def __init__(self, dim_in, n_head, device='cpu', dropout=0.1, temperature=1, cls_weight=1, cls_loss=nn.CrossEntropyLoss()):
        super().__init__(temperature=temperature, cls_weight=cls_weight, cls_loss=cls_loss)
        self.kMap = nn.Linear(dim_in, dim_in)
        self.qMap = nn.Linear(dim_in, dim_in)
        self.vMap = nn.Linear(dim_in, dim_in)
        self.attn = nn.MultiheadAttention(dim_in, n_head, dropout=dropout, batch_first=True, device=device)

    def forward(self, text_embeddings, images, label_inds=None):
        # attn: Value: (N, S, E) Key (N, S, E), Query: (N, L, E) -> (N, L, E)
        # texts: (L,D) , images: (N,D,H,W), label_inds: (N, L)
        text_embeddings = text_embeddings.repeat(images.shape[0], 1, 1) # (N, L, D)
        images = images.flatten(start_dim=2).permute(0, 2, 1)
        
        if label_inds is not None:
            mask = label_inds.unsqueeze(-1).repeat(1, 1, text_embeddings.shape[2])
            text_embeddings = mask * text_embeddings
        
        key = self.kMap(images)
        value = self.vMap(images)
        query = self.qMap(text_embeddings)
        out, _ = self.attn(query, key, value, need_weights=False)
        
 
        return out / out.norm(dim=-1, keepdim=True)

class LabelImageAttention(BaseLabelImageAttention):
    def __init__(self, dim_in, n_head, dropout=0.1, num_layers=6, temperature=1, cls_weight=1, cls_loss=nn.CrossEntropyLoss()):
        super().__init__(temperature=temperature, cls_weight=cls_weight, cls_loss=cls_loss)
        self.attn = nn.Transformer(dim_in, batch_first=True, nhead=n_head, dropout=dropout, num_decoder_layers=num_layers, num_encoder_layers=num_layers)
        # self.con_loss = SupConLoss(temperature=temperature, contrast_mode='one')
        # self.class_loss = cls_loss
        # self.cls_weight = cls_weight

    # def set_trainable(self, trainable):
    #     for param in self.attn.parameters():
    #         param.requires_grad = trainable

    # def image_text_logits(self, text_embedings, images, label_inds=None):
    #     prototypes = self.forward(text_embedings, images, label_inds)
    #     return image_text_logits(text_embedings, prototypes)
    
    # def loss(self, text_embeddings, prototypes, label_inds):
    #     logits = image_text_logits(text_embeddings, prototypes)
    #     return self.contrastive_loss(prototypes, label_inds) + self.cls_weight * self.classification_loss(logits, label_inds)
    
    # def classification_loss(self, logits, label_inds):
    #     return self.class_loss(logits, label_inds.float())
    
    # def contrastive_loss(self, results, label_inds):
    #     # results: (N, L, D), labels: (N, L)
    #     classes = torch.nonzero(label_inds)[:,1] # (Np,)
    #     prototypes = results[label_inds.bool()] # (Np, D)
    #     return self.con_loss(prototypes.unsqueeze(1), classes)

    def forward(self, text_embeddings, images, label_inds=None):
        # transformer: (N, S, E), (N, T, E) -> (N, T, E)
        # texts: (L,D) , images: (N,D,H,W), label_inds: (N, L)
        text_embeddings = text_embeddings.repeat(images.shape[0], 1, 1)
        images = images.flatten(start_dim=2).permute(0, 2, 1)
        mask = None
        if label_inds is not None:
            mask = (1 - label_inds).bool()
        
        # Texts: NxLxD (decode)
        # Mask irrelevant labels with tgt_key_padding_mask, set masked positions to True
        # Images: Nx(HxW)xD
        # Output: (N, L, D)
        out = self.attn(images, text_embeddings, tgt_key_padding_mask=mask)
        return out / out.norm(dim=-1, keepdim=True)



class LabelImagePrototypeModel(nn.Module):
    def __init__(self, encoder, n_head, dim_in=512, dropout=0.1, num_layers=6, temperature=1, cls_weight=1, cls_loss=nn.CrossEntropyLoss(), attn_model=None):
        super().__init__()
        self.encoder = encoder
        if attn_model is None:
            self.attention = LabelImageAttention(dim_in, n_head, dropout, num_layers, temperature, cls_weight=cls_weight, cls_loss=cls_loss)
        else:
            self.attention = attn_model

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, unfreeze_img_bb=False):
        self.encoder.set_trainable(True, unfreeze_img_bb)
    
    def forward(self, class_labels, images, label_inds=None):
        text_embedding, image_emedding = self.encoder(class_labels, images, False)
        prototypes = self.attention(text_embedding, image_emedding, label_inds)
        return text_embedding, image_emedding, prototypes
    
    def attention_loss(self, text_embeddings, prototypes, label_inds):
        return self.attention.loss(text_embeddings, prototypes, label_inds)
    
    def encoder_loss(self, text_embeddings, prototypes, label_inds):
        logits_per_image = image_text_logits(text_embeddings, prototypes, self.encoder.get_logit_scale())
        return self.encoder.contrastive_logit_loss(logits_per_image.t(), logits_per_image, label_inds)
