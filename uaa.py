# ============================================================
# IMPORTS
# ============================================================

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from PIL import Image
from tqdm import tqdm



# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    dataset_path: str = "DATASET_PATH"
    img_size: int = 112

    batch_size: int = 32
    num_workers: int = 4

    feature_dim: int = 512
    style_dim: int = 16

    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    epochs: int = 60
    warmup_epochs: int = 5

    gamma: float = 0.5

    geo_pgd_steps: int = 1
    tex_pgd_steps: int = 2

    arcface_s: float = 64.0
    arcface_m: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"



# ============================================================
# DATASET
# ============================================================

class PalmprintDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

        self.tf = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        img = Image.open(sample["path"]).convert("RGB")

        return {
            "img": self.tf(img),
            "label": sample["label"]
        }



# ============================================================
# SPATIAL TRANSFORMER
# ============================================================

class SpatialTransformer(nn.Module):

    def forward(self,x,p):

        tx,ty,theta,scale = p[:,0],p[:,1],p[:,2],p[:,3]

        c = torch.cos(theta)
        s = torch.sin(theta)

        row1 = torch.stack([scale*c,-scale*s,tx],1)
        row2 = torch.stack([scale*s, scale*c,ty],1)

        M = torch.stack([row1,row2],1)

        grid = F.affine_grid(M,x.size(),align_corners=False)

        return F.grid_sample(x,grid,align_corners=False)



# ============================================================
# IDENTITY ENCODER
# ============================================================

class IdentityEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet50(weights=None)

        self.layers = nn.Sequential(*list(backbone.children())[:8])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self,x):

        return self.layers(x)



# ============================================================
# PALM GENERATOR
# ============================================================

class PalmGenerator(nn.Module):

    def __init__(self,style_dim):

        super().__init__()

        self.fc = nn.Linear(style_dim,512*4*4)

        self.conv1 = nn.ConvTranspose2d(512+2048,256,4,2,1)
        self.conv2 = nn.ConvTranspose2d(256,128,4,2,1)
        self.conv3 = nn.ConvTranspose2d(128,64,4,2,1)
        self.conv4 = nn.ConvTranspose2d(64,32,4,2,1)

        self.out = nn.Conv2d(32,3,3,1,1)

    def forward(self,z,id_feat):

        B = z.size(0)

        x = self.fc(z).view(B,512,4,4)

        id_feat = F.interpolate(id_feat,size=(4,4))

        x = torch.cat([x,id_feat],1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        return torch.tanh(self.out(x))



# ============================================================
# GENERATION NETWORK
# ============================================================

class PalmGenerationNetwork(nn.Module):

    def __init__(self,style_dim):

        super().__init__()

        self.encoder = IdentityEncoder()
        self.generator = PalmGenerator(style_dim)

    def forward(self,z,x):

        id_feat = self.encoder(x)

        return self.generator(z,id_feat)

    def freeze(self):

        for p in self.parameters():
            p.requires_grad=False



# ============================================================
# RECOGNITION NETWORK
# ============================================================

class RecognitionNet(nn.Module):

    def __init__(self,num_classes,feat_dim):

        super().__init__()

        backbone = resnet50(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.fc = nn.Linear(2048,feat_dim)

        self.cls = nn.Linear(feat_dim,num_classes)

    def forward(self,x):

        f = self.backbone(x).flatten(1)

        f = self.fc(f)

        logits = self.cls(f)

        return logits,f



# ============================================================
# ARCFACE LOSS
# ============================================================

class ArcFaceLoss(nn.Module):

    def __init__(self,feat_dim,num_classes,s=64.0,m=0.5):

        super().__init__()

        self.s=s
        self.m=m

        self.W = nn.Parameter(torch.randn(num_classes,feat_dim))

    def forward(self,feat,labels):

        feat = F.normalize(feat)
        W = F.normalize(self.W)

        logits = feat @ W.T

        return F.cross_entropy(self.s*logits,labels)



# ============================================================
# ADVERSARIAL OPTIMIZER
# ============================================================

class AugPGD:

    def __init__(self,aug_net,rec_net,geo_steps,tex_steps):

        self.aug = aug_net
        self.rec = rec_net

        self.geo_steps = geo_steps
        self.tex_steps = tex_steps

    def optimize(self,x,y,z,mode,steps):

        z = z.clone().detach().requires_grad_(True)

        for _ in range(steps):

            if z.grad is not None:
                z.grad.zero_()

            x_aug = self.aug(x,z,mode)

            logits,_ = self.rec(x_aug)

            loss = F.cross_entropy(logits,y)

            loss.backward()

            grad = torch.sign(z.grad)

            alpha = torch.normal(mean=0.1,std=0.001,size=(1,),device=z.device)

            with torch.no_grad():
                z = z + alpha*grad

        return z.detach()



# ============================================================
# UNIFIED AUGMENTATION MODULE
# ============================================================

class UnifiedAugmentation(nn.Module):

    def __init__(self,style_dim):

        super().__init__()

        self.spatial = SpatialTransformer()
        self.gen = PalmGenerationNetwork(style_dim)

    def forward(self,x,z,mode):

        if mode=="geo":
            return self.spatial(x,z[:,:4])

        if mode=="tex":
            return self.gen(z[:,4:],x)

        if mode=="both":
            x = self.spatial(x,z[:,:4])
            return self.gen(z[:,4:],x)

        return x



# ============================================================
# TRAINER
# ============================================================

class Trainer:

    def __init__(self,cfg,num_classes):

        self.cfg = cfg

        self.rec = RecognitionNet(num_classes,cfg.feature_dim).to(cfg.device)

        self.aug = UnifiedAugmentation(cfg.style_dim).to(cfg.device)

        self.opt = optim.SGD(
            self.rec.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )

        self.pgd = AugPGD(
            self.aug,
            self.rec,
            cfg.geo_pgd_steps,
            cfg.tex_pgd_steps
        )

    def train_epoch(self,loader):

        self.rec.train()

        gamma = self.cfg.gamma

        for batch in tqdm(loader):

            x = batch["img"].to(self.cfg.device)
            y = batch["label"].to(self.cfg.device)

            B = x.size(0)

            k = int(gamma*B)

            idx = torch.randperm(B)[:k]

            x_sub = x[idx]
            y_sub = y[idx]

            z = torch.randn(k,20,device=self.cfg.device)

            z = self.pgd.optimize(x_sub,y_sub,z,"geo",self.cfg.geo_pgd_steps)

            z = self.pgd.optimize(x_sub,y_sub,z,"tex",self.cfg.tex_pgd_steps)

            with torch.no_grad():
                x_aug = self.aug(x_sub,z,"both")

            x_all = torch.cat([x,x_aug])
            y_all = torch.cat([y,y_sub])

            logits,_ = self.rec(x_all)

            loss = F.cross_entropy(logits,y_all)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()



# ============================================================
# MAIN
# ============================================================

def main():

    cfg = Config()

    print("Unified Adversarial Augmentation training ready.")

if __name__ == "__main__":
    main()
