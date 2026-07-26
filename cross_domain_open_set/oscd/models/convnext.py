import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtFinetune(nn.Module):
    """
    ConvNeXtV2-Tiny:
      - stem + stages 0-2 : frozen
      - stage 3 + final norm : trainable
    Returns L2-normalised embeddings.
    """
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
        for p in backbone.parameters(): p.requires_grad = False
        for p in backbone.stages[3].parameters(): p.requires_grad = True
        if hasattr(backbone, 'norm'):
            for p in backbone.norm.parameters(): p.requires_grad = True
        self.backbone  = backbone
        self.embed_dim = backbone.num_features

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)


