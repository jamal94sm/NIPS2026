import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceBackbone(nn.Module):
    def __init__(self, pretrained_path, freeze_ratio=0.75):
        super().__init__()
        import onnx
        from onnx2torch import convert
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"ONNX not found: {pretrained_path}")
        print(f"  Loading ONNX model: {pretrained_path}")
        self.net = convert(onnx.load(pretrained_path))
        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)): out = out[0]
        return F.normalize(out, p=2, dim=1)


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=64.0, m=0.50):
        super().__init__()
        self.s=s; self.cos_m=math.cos(m); self.sin_m=math.sin(m)
        self.th=math.cos(math.pi-m); self.mm=math.sin(math.pi-m)*m
        self.weight=nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight); self.ce=nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        W=F.normalize(self.weight,p=2,dim=1)
        cos_theta=(embeddings@W.t()).clamp(-1+1e-7,1-1e-7)
        sin_theta=(1.0-cos_theta**2).sqrt()
        cos_theta_m=cos_theta*self.cos_m-sin_theta*self.sin_m
        cos_theta_m=torch.where(cos_theta>self.th,cos_theta_m,cos_theta-self.mm)
        one_hot=torch.zeros_like(cos_theta).scatter_(1,labels.view(-1,1),1.0)
        logits=self.s*(one_hot*cos_theta_m+(1.0-one_hot)*cos_theta)
        return self.ce(logits,labels)

    @torch.no_grad()
    def get_logits(self, embeddings):
        W=F.normalize(self.weight,p=2,dim=1)
        return self.s*(embeddings@W.t())


