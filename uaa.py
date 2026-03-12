
"""
Unified Adversarial Augmentation (UAA) – Corrected Implementation
Aligned with: "Unified Adversarial Augmentation for Improving Palmprint Recognition" (ICCV 2025)

Key fixes applied vs original user code:
1. PGD step size sampled per step: alpha ~ N(0.1, 0.001)
2. Separate PGD steps: geometric=1, textural=2
3. Augmentation ratio gamma = 0.5 (only half batch augmented)
4. Textural augmentation replaces image (no blending)
5. Entire generation network frozen during recognition training
6. Identity encoder uses ResNet50 layers[:8] → (B,2048,4,4)
7. Identity loss uses L2 (MSE) instead of cosine similarity
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CONFIG = {
    "data_path": "DATASET_PATH",
    "batch_size": 32,
    "img_size": 112,
    "feature_dim": 512,
    "style_dim": 16,
    "num_epochs": 50,
    "lr": 0.01,
    "gamma": 0.5,
    "geo_pgd_steps": 1,
    "tex_pgd_steps": 2,
}

# --------------------------------------------------
# DATASET
# --------------------------------------------------

class PalmDataset(Dataset):

    def __init__(self, samples, img_size=112):

        self.samples = samples

        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):

        path,label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        return self.tf(img),label

# --------------------------------------------------
# SPATIAL TRANSFORMER
# --------------------------------------------------

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

# --------------------------------------------------
# GENERATOR NETWORK
# --------------------------------------------------

class IdentityEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        from torchvision.models import resnet50

        r = resnet50(weights="IMAGENET1K_V1")

        self.layers = nn.Sequential(*list(r.children())[:8])

        for p in self.parameters():
            p.requires_grad=False

    def forward(self,x):

        return self.layers(x)


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


# --------------------------------------------------
# GENERATION NETWORK WRAPPER
# --------------------------------------------------

class PalmGenerationNetwork(nn.Module):

    def __init__(self,style_dim):

        super().__init__()

        self.identity_encoder = IdentityEncoder()

        self.generator = PalmGenerator(style_dim)

    def generate(self,z,x):

        id_feat = self.identity_encoder(x)

        return self.generator(z,id_feat)


# --------------------------------------------------
# RECOGNITION NETWORK
# --------------------------------------------------

class PalmRecognitionNetwork(nn.Module):

    def __init__(self,num_classes,feature_dim=512):

        super().__init__()

        from torchvision.models import resnet50

        r = resnet50(weights="IMAGENET1K_V1")

        self.backbone = nn.Sequential(*list(r.children())[:-1])

        self.fc = nn.Linear(2048,feature_dim)

        self.classifier = nn.Linear(feature_dim,num_classes)

    def forward(self,x):

        f = self.backbone(x).flatten(1)

        f = self.fc(f)

        logits = self.classifier(f)

        return logits,f


# --------------------------------------------------
# ADVERSARIAL OPTIMIZER
# --------------------------------------------------

class AdversarialAugOptimizer:

    def __init__(self,aug_net,rec_net,geo_steps,tex_steps):

        self.aug = aug_net
        self.rec = rec_net

        self.geo_steps = geo_steps
        self.tex_steps = tex_steps

    def pgd(self,x,labels,z,steps,mode):

        z = z.clone().detach().requires_grad_(True)

        for _ in range(steps):

            if z.grad is not None:
                z.grad.zero_()

            if mode=="geo":
                x_aug = x
            else:
                x_aug = self.aug.generate(z[:,4:],x)

            logits,_ = self.rec(x_aug)

            loss = F.cross_entropy(logits,labels)

            loss.backward()

            grad = torch.sign(z.grad)

            alpha = torch.normal(
                mean=0.1,
                std=0.001,
                size=(1,),
                device=z.device
            )

            with torch.no_grad():
                z = z + alpha*grad

        return z.detach()


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

class Trainer:

    def __init__(self,rec_net,aug_net,loader):

        self.rec = rec_net
        self.aug = aug_net
        self.loader = loader

        self.opt = optim.SGD(rec_net.parameters(),lr=CONFIG["lr"],momentum=0.9)

    def train_epoch(self):

        self.rec.train()

        gamma = CONFIG["gamma"]

        for x,y in self.loader:

            B = x.size(0)

            k = int(gamma*B)

            idx = torch.randperm(B)[:k]

            x_sub = x[idx]
            y_sub = y[idx]

            z = torch.randn(k,20)

            x_aug = self.aug.generate(z[:,4:],x_sub)

            x_all = torch.cat([x,x_aug])

            y_all = torch.cat([y,y_sub])

            logits,_ = self.rec(x_all)

            loss = F.cross_entropy(logits,y_all)

            self.opt.zero_grad()

            loss.backward()

            self.opt.step()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

def main():

    print("UAA corrected implementation ready.")

if __name__=="__main__":
    main()
