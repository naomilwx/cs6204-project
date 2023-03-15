import torch
from torch import nn
from utils.contrastive_loss import SupConLoss

class LabelImageAttention(nn.Module):
    def __init__(self, dim_in, n_head, dropout=0.1, num_layers=4, temperature=1):
        super().__init__()
        self.attn = nn.Transformer(dim_in, batch_first=True, nhead=n_head, dropout=dropout, num_decoder_layers=num_layers, num_encoder_layers=num_layers)
        self.con_loss = SupConLoss(temperature=temperature)

    def forward(self, texts, images, label_inds=None):
        # transformer: (N, S, E), (N, T, E) -> (N, T, E)
        # texts: (L,D) , images: (N,D,H,W), label_inds: (N, L)
        texts = texts.expand(images.shape[0], -1, -1)
        images = images.flatten(start_dim=2).permute(0, 2, 1)
        mask = None
        if label_inds is not None:
            mask = 1 - label_inds
        # Output: (N, L, D)
        # Texts: NxLxD (decode)
        # Mask irrelevant labels with tgt_key_padding_mask, set masked positions to True
        # Images: Nx(HxW)xD
        return self.attn(images, texts, tgt_key_padding_mask=mask)
    
    def loss(self, results, label_inds):
        # results: (N, L, D), labels: (N, L)
        classes = torch.nonzero(label_inds)[:,1]
        prototypes = results[label_inds.bool()]
        return self.con_loss(prototypes, classes)