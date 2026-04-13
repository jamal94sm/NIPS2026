"""
models.py — CO3Net architecture.
Exact copy of the official CO3Net implementation (unchanged).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ──────────────────────────────────────────────────────────────
#  Learnable Gabor Convolution
# ──────────────────────────────────────────────────────────────

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) layer."""

    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = max(init_ratio, 1e-6)
        self.kernel      = 0

        self._SIGMA = 9.2  * self.init_ratio
        self._FREQ  = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]))
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([self._FREQ]))
        self.psi = nn.Parameter(torch.FloatTensor([0]),
                                requires_grad=False)

    def _gen_gabor_bank(self, ksize, c_in, c_out,
                        sigma, gamma, theta, f, psi):
        half  = ksize // 2
        ksize = 2 * half + 1
        y0    = torch.arange(-half, half + 1).float()
        x0    = torch.arange(-half, half + 1).float()

        y = y0.view(1, -1).repeat(c_out, c_in, ksize, 1)
        x = x0.view(-1, 1).repeat(c_out, c_in, 1, ksize)
        x = x.to(sigma.device)
        y = y.to(sigma.device)

        xt = x * torch.cos(theta.view(-1,1,1,1)) + y * torch.sin(theta.view(-1,1,1,1))
        yt = -x * torch.sin(theta.view(-1,1,1,1)) + y * torch.cos(theta.view(-1,1,1,1))

        gb = -torch.exp(
            -0.5 * ((gamma * xt)**2 + yt**2)
            / (8 * sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2 * math.pi * f.view(-1,1,1,1) * xt
                      + psi.view(-1,1,1,1))
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel     = self._gen_gabor_bank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


# ──────────────────────────────────────────────────────────────
#  Coordinate Attention
# ──────────────────────────────────────────────────────────────

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip         = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        identity   = x
        n, c, h, w = x.size()
        xh         = self.pool_h(x)
        xw         = self.pool_w(x).permute(0, 1, 3, 2)
        y          = self.act(self.bn1(self.conv1(torch.cat([xh, xw], dim=2))))
        xh, xw     = torch.split(y, [h, w], dim=2)
        xw         = xw.permute(0, 1, 3, 2)
        return identity * self.conv_h(xh).sigmoid() * self.conv_w(xw).sigmoid()


# ──────────────────────────────────────────────────────────────
#  Competitive Block
# ──────────────────────────────────────────────────────────────

class CompetitiveBlock(nn.Module):
    """LGC1 → CoordAtt → LGC2 → CoordAtt → soft-argmax → PPU"""

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor1   = GaborConv2d(channel_in, n_competitor, ksize,
                                    stride, ksize // 2, init_ratio)
        self.gabor2   = GaborConv2d(n_competitor, n_competitor, ksize,
                                    1, ksize // 2, init_ratio)
        self.cooratt1 = CoordAtt(n_competitor, n_competitor)
        self.cooratt2 = CoordAtt(n_competitor, n_competitor)
        self.a        = nn.Parameter(torch.FloatTensor([1]))
        self.b        = nn.Parameter(torch.FloatTensor([0]))
        self.argmax   = nn.Softmax(dim=1)
        self.conv1    = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.conv2    = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.cooratt1(self.gabor1(x))
        x = self.cooratt2(self.gabor2(x))
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


# ──────────────────────────────────────────────────────────────
#  ArcFace head
# ──────────────────────────────────────────────────────────────

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features,
                 s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s           = s
        self.m           = m
        self.easy_margin = easy_margin
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if not self.training:
            return self.s * cosine
        sine  = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = (torch.where(cosine > 0, phi, cosine)
                 if self.easy_margin
                 else torch.where(cosine > self.th, phi, cosine - self.mm))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))


# ──────────────────────────────────────────────────────────────
#  CO3Net
# ──────────────────────────────────────────────────────────────

class CO3Net(nn.Module):
    """
    CO3Net = CB1 ∥ CB2 ∥ CB3 + FC + Dropout + ArcFace
    FC dimension:  17328 = (12 + 24 + 12) × 19 × 19  for 128×128 input
    """

    def __init__(self, num_classes, dropout=0.5,
                 arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.cb1 = CompetitiveBlock(1,  9, 35, 3, 17, init_ratio=1.00)
        self.cb2 = CompetitiveBlock(1, 36, 17, 3,  8, init_ratio=0.50, o2=24)
        self.cb3 = CompetitiveBlock(1,  9,  7, 3,  3, init_ratio=0.25)
        self.fc   = nn.Linear(17328, 4096)
        self.fc1  = nn.Linear(4096,  2048)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(2048, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1)
        x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        x  = torch.cat([x1, x2, x3], dim=1)
        h  = self.fc(x)
        e  = self.fc1(h)
        return h, e

    def forward(self, x, y=None):
        h, e = self._backbone(x)
        fe   = F.normalize(torch.cat([h, e], dim=1), dim=-1)  # 6144-d contrastive
        emb  = F.normalize(e, dim=-1)                          # 2048-d matching embedding
        out  = self.arc(self.drop(e), y)
        return out, fe, emb                                     # ← added emb

    @torch.no_grad()
    def get_embedding(self, x):
        """2048-d L2-normalised embedding for matching."""
        _, e = self._backbone(x)
        return F.normalize(e, p=2, dim=1)
