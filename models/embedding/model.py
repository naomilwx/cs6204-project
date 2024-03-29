import torch
from torch import nn
import torchvision

from transformers import AutoTokenizer, AutoModel

def adapt_resnet_input_channels(model, img_channels):
    if model.conv1.in_channels != img_channels:
        model.conv1 = torch.nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def load_pretrained_resnet(img_channels, num_classes, save_path, fc_bias=True):
    model = torchvision.models.resnet50(num_classes=num_classes, weights=None)
    if fc_bias == False:
        model.fc = nn.Linear(2048, num_classes, bias=False)
    model = adapt_resnet_input_channels(model, img_channels)
    model.load_state_dict(torch.load(save_path))
    return model

def resnet_backbone(model):
    return torch.nn.Sequential(*(list(model.children())[:-2]))

def load_medclip_retrained_resnet(path):
    return resnet_backbone(load_pretrained_resnet(1, 512, path, False))

class ImageEncoder(nn.Module):
    def __init__(self, backbone, embed_dims, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(2048, embed_dims)
        if freeze_backbone:
            self.set_backbone_trainable(False)
    
    def set_backbone_trainable(self, trainable):
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def set_trainable(self, trainable, include_backbone):
        for param in self.proj.parameters():
            param.requires_grad = trainable
        self.set_backbone_trainable(include_backbone)

    def set_backbone_layer_trainable(self, trainable, idx):
        for param in self.backbone[idx].parameters():
            param.requires_grad = trainable

    def forward(self, input):
        # B, C, H, W
        img = self.backbone(input)
        # B, C, H, W -> B, H, W, C -> B, D, H, W
        return self.proj(img.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TextEncoder(nn.Module):
    def __init__(self, embed_dims, device='cpu', freeze_backbone=True, bert_pretrained_type='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_type)
        self.backbone = AutoModel.from_pretrained(bert_pretrained_type)
        self.proj = nn.Linear(768, embed_dims)
        self.device = device
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable):
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def set_trainable(self, trainable, include_backbone=False):
        for param in self.proj.parameters():
            param.requires_grad = trainable
        self.set_backbone_trainable(include_backbone)

    
    def forward(self, input):
        tokens = self.tokenizer(input, max_length=77, return_tensors='pt', padding='max_length').to(self.device)
        out = self.backbone(**tokens)
        enc = out['pooler_output']
        # enc = out['last_hidden_state'][:, 0]
        return self.proj(enc)

class ImageTextEmbedding(nn.Module):
    def __init__(self, img_backbone, embed_dims, logit_scale_init_value=0.1, device='cpu', bert_pretrained_type='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.text_model = TextEncoder(embed_dims, device, bert_pretrained_type=bert_pretrained_type)
        self.img_model = ImageEncoder(img_backbone, embed_dims)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def set_trainable(self, trainable, include_image_bb, include_text_bb=False, include_logit_scale=False):
        self.logit_scale.requires_grad = include_logit_scale
        self.img_model.set_trainable(trainable, include_image_bb)
        self.text_model.set_trainable(trainable, include_text_bb)
    
    def embed_text(self, text):
        text_emb = self.text_model(text)
        return text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    def embed_image(self, image, pool=False):
        img_emb = self.img_model(image) # B, D, H, W
        if pool:
            img_emb = self.flatten(self.gap(img_emb)) # B, D
        return img_emb / img_emb.norm(dim=-1, keepdim=True)

    def get_logit_scale(self):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        return self.logit_scale.exp()
    
    def compute_logits(self, text_emb, img_emb):
        # text_emb: (L, E), img_embed: (N, E)
        logit_scale = self.get_logit_scale()        
        if len(img_emb.shape) == 4:
            logits_per_image = logit_scale * torch.matmul(img_emb.permute(2,3,0,1), text_emb.t())
        else:
            logits_per_image = logit_scale * torch.matmul(img_emb, text_emb.t())
        
        if len(img_emb.shape) == 4:
            logits_per_text = logits_per_image.permute(0,1,3,2) # HxWxBxN
        else:
            logits_per_text = logits_per_image.t()
        # (L, N), (N, L)
        return logits_per_text, logits_per_image
        
    def forward(self, text, img, pool=False):
        text_emb = self.embed_text(text)
        img_emb = self.embed_image(img, pool)

        return text_emb, img_emb
    
    def contrastive_logit_loss(self, logits_per_text, logits_per_image, labels):
         # Image-label contrastive loss, which is similar to classification loss, except using the computed logits
        labels = labels.float()
        itl = self.criterion(logits_per_image, labels)
        til = self.criterion(logits_per_text, labels.t())
        return (itl+til) / 2
    
    def loss(self, text_emb, img_emb, labels):
        # text_embed should be an NxD matrix where N is the number of classes, so each row is the text embedding for the ith class
        # image embed: BxD
        # labels is an BxN indicator matrix with 1 for each class an image belongs to
        logits_per_text, logits_per_image = self.compute_logits(text_emb, img_emb)
        
        return self.contrastive_logit_loss(logits_per_text, logits_per_image, labels)
    
class DummyEncoder(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
    
    def set_trainable(self, trainable, include_image_bb, include_logit_scale=False):
        pass

    def embed_image(self, image, pool=False):
        emb = image[:, :self.embed_dims, :, :]
        if pool:
            emb = self.flatten(self.gap(emb)) # B, D
        return emb / emb.norm(dim=-1, keepdim=True)
    
    def forward(self, img, pool=False):
        return  self.embed_image(img, pool)
    
class ImageOnlyEmbedding(nn.Module):
    def __init__(self, img_backbone, embed_dims, logit_scale_init_value=0.1):
        super().__init__()
        self.img_model = ImageEncoder(img_backbone, embed_dims)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

    def set_trainable(self, trainable, include_image_bb, include_logit_scale=False):
        self.logit_scale.requires_grad = include_logit_scale
        self.img_model.set_trainable(trainable, include_image_bb)

    def embed_image(self, image, pool=False):
        img_emb = self.img_model(image) # B, D, H, W
        if pool:
            img_emb = self.flatten(self.gap(img_emb)) # B, D
        return img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    def forward(self, img, pool=False):
        return  self.embed_image(img, pool)
    
    def get_logit_scale(self):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        return self.logit_scale.exp()
    
    def loss(self, img_emb, labels):
        logit_scale = self.get_logit_scale()
        
        logits = logit_scale * torch.matmul(labels.t(), img_emb).t()
        return self.criterion(logits, labels.float())
