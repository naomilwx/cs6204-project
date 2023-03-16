import torch
from torch import nn
from utils.contrastive_loss import SupConLoss

def image_text_logits(text_embeddings, prototypes, scale=1):
    # text_embeddings: (14, 512) x prototypes: (140, 14, 512) -> (140, 14)
    fac = text_embeddings.unsqueeze(0).expand_as(prototypes)
    return (fac * prototypes).sum(axis=2) * scale

class LabelImageAttention(nn.Module):
    def __init__(self, dim_in, n_head, dropout=0.1, num_layers=6, temperature=1):
        super().__init__()
        self.attn = nn.Transformer(dim_in, batch_first=True, nhead=n_head, dropout=dropout, num_decoder_layers=num_layers, num_encoder_layers=num_layers)
        self.con_loss = SupConLoss(temperature=temperature)

    def forward(self, texts, images, label_inds=None):
        # transformer: (N, S, E), (N, T, E) -> (N, T, E)
        # texts: (L,D) , images: (N,D,H,W), label_inds: (N, L)
        texts = texts.repeat(images.shape[0], 1, 1)
        # view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        images = images.flatten(start_dim=2).permute(0, 2, 1)
        mask = None
        if label_inds is not None:
            mask = (1 - label_inds).bool()
        
        # Texts: NxLxD (decode)
        # Mask irrelevant labels with tgt_key_padding_mask, set masked positions to True
        # Images: Nx(HxW)xD
        # Output: (N, L, D)
        out = self.attn(images, texts, tgt_key_padding_mask=mask)
        return out / out.norm(dim=-1, keepdim=True)
    
    def loss(self, results, label_inds):
        # results: (N, L, D), labels: (N, L)
        classes = torch.nonzero(label_inds)[:,1] # (Np,)
        prototypes = results[label_inds.bool()] # (Np, D)
        return self.con_loss(prototypes.unsqueeze(1), classes)


class LabelImagePrototypeModel(nn.Module):
    def __init__(self, encoder, n_head, dim_in=512, dropout=0.1, num_layers=4, temperature=1):
        super().__init__()
        self.encoder = encoder
        self.attention = LabelImageAttention(dim_in, n_head, dropout, num_layers, temperature)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.encoder.img_model.proj.requires_grad = True
        self.encoder.text_model.proj.requires_grad = True
    
    def forward(self, class_labels, images, label_inds):
        text_embedding, image_emedding = self.encoder(class_labels, images, False)
        prototypes = self.attention(text_embedding, image_emedding, label_inds)
        return text_embedding, image_emedding, prototypes
    
    def attention_loss(self, prototypes, label_inds):
        return self.attention.loss(prototypes, label_inds)
