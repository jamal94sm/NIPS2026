import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOFinetune(nn.Module):
    """
    DINOv2 ViT-S/14 (facebookresearch/dinov2):
      - patch embed + blocks 0-9 : frozen
      - blocks 10-11             : trainable  (last 2 of 12 blocks)
    Returns L2-normalised [CLS] token embeddings (dim=384).
    """
    def __init__(self):
        super().__init__()
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14")

        # Freeze all, then selectively unfreeze blocks 10 and 11
        for name, p in backbone.named_parameters():
            p.requires_grad = False
            if "blocks.10" in name or "blocks.11" in name:
                p.requires_grad = True

        self.backbone  = backbone
        self.embed_dim = 384   # ViT-S/14 embedding dimension

    def forward(self, x):
        out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"]    # [B, 384]  CLS token
        return F.normalize(cls, p=2, dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)


