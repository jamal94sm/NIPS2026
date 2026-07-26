import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2   = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3   = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample; self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x);    out = self.conv1(out)
        out = self.bn2(out);  out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, block, layers, dropout=0.0, num_features=512,
                 groups=1, width_per_group=64):
        super().__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu    = nn.PReLU(64)
        self.layer1   = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2   = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3   = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4   = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2      = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout  = nn.Dropout(p=dropout)
        self.fc       = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x);    x = self.dropout(x)
        x = x.flatten(1);   x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(num_features=512):
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_features=num_features)


class MagFaceBackbone(nn.Module):
    """
    iResNet100 loaded from MagFace checkpoint.
    Freeze ratio: first 75% of parameter tensors.
    Returns RAW (non-normalised) embeddings — MagFace needs the magnitude.
    """
    def __init__(self, pretrained_path, freeze_ratio=0.75):
        super().__init__()
        self.net = iresnet100()
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt  = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("features.module.", ""): v
                     for k, v in state.items()
                     if k.startswith("features.module.")}
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"  Loaded: {pretrained_path}  "
                  f"(epoch {ckpt.get('epoch','?')}  arch={ckpt.get('arch','?')})")
            if missing:    print(f"    Missing keys    : {len(missing)}")
            if unexpected: print(f"    Unexpected keys : {len(unexpected)}")
        else:
            print(f"  [WARN] Pretrained weights not found: {pretrained_path}")

        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def forward(self, x):
        return self.net(x)


class MagFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512,
                 s=64.0, m_l=0.45, m_u=0.80,
                 l_a=10.0, u_a=110.0, lambda_g=20.0):
        super().__init__()
        self.s=s; self.m_l=m_l; self.m_u=m_u
        self.l_a=l_a; self.u_a=u_a; self.lambda_g=lambda_g
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def _adaptive_margin(self, norm):
        a = norm.clamp(self.l_a, self.u_a)
        return self.m_l + (self.m_u - self.m_l) * (a - self.l_a) / (self.u_a - self.l_a)

    def _magnitude_regularizer(self, norm):
        a = norm.clamp(self.l_a, self.u_a)
        return ((1.0 / (self.u_a ** 2)) * a + 1.0 / a).mean()

    def forward(self, embeddings, labels):
        norm      = embeddings.norm(dim=1)
        z_normed  = F.normalize(embeddings, p=2, dim=1)
        W_normed  = F.normalize(self.weight, p=2, dim=1)
        cos_theta = (z_normed @ W_normed.t()).clamp(-1+1e-7, 1-1e-7)
        m         = self._adaptive_margin(norm)
        sin_theta = (1.0 - cos_theta ** 2).sqrt()
        cos_m = torch.cos(m).unsqueeze(1); sin_m = torch.sin(m).unsqueeze(1)
        th = math.cos(math.pi - self.m_u); mm = math.sin(math.pi - self.m_u) * self.m_u
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        cos_theta_m = torch.where(cos_theta > th, cos_theta_m, cos_theta - mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits  = self.s * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)
        L_arc   = self.ce(logits, labels)
        L_g     = self._magnitude_regularizer(norm)
        return L_arc + self.lambda_g * L_g, L_arc.item(), L_g.item()

    @torch.no_grad()
    def get_logits(self, embeddings):
        z = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        return self.s * (z @ W.t())


