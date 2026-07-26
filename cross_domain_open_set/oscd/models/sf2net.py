import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    os.system("pip install einops --quiet")
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange


class TripletLoss(nn.Module):
    """Triplet loss with SRT (Soft Relative Triplet) distance."""
    def __init__(self, margin=2.0, alpha=0.95, distance="SRT"):
        super().__init__()
        self.margin = margin
        self.alpha  = alpha
        self.distance = distance
        self.tripletMargin = nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean')

    def dis(self, a, b):
        return torch.sum((a - b).pow(2), 1)

    def forward(self, anchor, positive, negative, size_average=True):
        if self.distance == "SRT":
            self.margin = 2.0
            anchor   = F.normalize(anchor,   p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            pos_d  = self.dis(anchor, positive)
            neg_d  = self.dis(anchor, negative)
            pn_d   = self.dis(positive, negative)
            cond   = neg_d.mean() >= pn_d.mean()
            ls     = torch.where(cond,
                                 pos_d + self.margin - pn_d.mean(),
                                 pos_d + self.margin - neg_d)
            losses = F.relu(ls).mean()
            return losses, pos_d.mean(), neg_d.mean(), pn_d.mean()
        else:
            raise ValueError(f"Unsupported distance: {self.distance}. Use 'SRT'.")


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, inp, label=None):
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        return self.s * cosine


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio
        self.kernel      = 0
        self.sigma     = nn.Parameter(torch.FloatTensor([9.2  * init_ratio]), requires_grad=True)
        self.gamma     = nn.Parameter(torch.FloatTensor([2.0]),               requires_grad=True)
        self.theta     = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([0.057 / init_ratio]), requires_grad=True)
        self.psi       = nn.Parameter(torch.FloatTensor([0]),                  requires_grad=False)

    def get_gabor(self):
        half = self.kernel_size // 2
        x_0 = torch.arange(-half, half + 1).float()
        y_0 = torch.arange(-half, half + 1).float()
        k   = self.kernel_size
        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k)
        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, k, 1)
        x = x.float().to(self.sigma.device)
        y = y.float().to(self.sigma.device)
        xt =  x*torch.cos(self.theta.view(-1,1,1,1)) + y*torch.sin(self.theta.view(-1,1,1,1))
        yt = -x*torch.sin(self.theta.view(-1,1,1,1)) + y*torch.cos(self.theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((self.gamma*xt)**2 + yt**2) / (8*self.sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*self.frequency.view(-1,1,1,1)*xt + self.psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self.get_gabor()
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


def get_sequence_feature(feature_tensor, vit_floor_num):
    ft    = torch.softmax(feature_tensor, dim=1)
    front = ft[:, :vit_floor_num, :, :]
    back  = ft[:, -vit_floor_num:, :, :]
    return torch.cat((front, back), dim=1)


class FeatureExtraction(nn.Module):
    def __init__(self, channel_in, filter_num, kernel_size, stride, padding,
                 init_ratio, label_num, vit_floor_num):
        super().__init__()
        self.vit_floor_num  = vit_floor_num
        self.gabor_conv2d_1 = GaborConv2d(channel_in, filter_num, kernel_size,
                                          stride, padding, init_ratio)
        self.gabor_conv2d_2 = GaborConv2d(filter_num, filter_num, kernel_size,
                                          stride, padding, init_ratio)
        self.se     = SEModule(channel=filter_num)
        self.conv_0 = nn.Conv2d(filter_num, 64, 5, 1, 0)
        self.conv_1 = nn.Conv2d(filter_num, 64, 5, 1, 0)
        self.conv_2 = nn.Conv2d(64, 32, 3, 2, 0)
        self.conv_3 = nn.Conv2d(64, 32, 3, 2, 0)
        self.max_pool = nn.MaxPool2d(2, 2)

    def process_block(self, x, conv):
        x = self.se(x); x = conv(x); x = torch.relu(x); x = self.max_pool(x)
        return x

    def forward(self, x):
        f1 = self.gabor_conv2d_1(x)
        f2 = self.gabor_conv2d_2(f1)
        f1p = self.process_block(f1, self.conv_0)
        f2p = self.process_block(f2, self.conv_1)
        out1 = self.conv_2(f1p)
        out2 = self.conv_3(f2p)
        feat = torch.cat((out1.flatten(1), out2.flatten(1)), dim=1)
        seq1 = get_sequence_feature(f1p, self.vit_floor_num)
        seq2 = get_sequence_feature(f2p, self.vit_floor_num)
        return feat, seq1, seq2


class FeedForward(nn.Module):
    def __init__(self, dim, dim_for_mlp, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim_for_mlp), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_for_mlp, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_for_head, dropout=0.1):
        super().__init__()
        self.heads   = heads
        inner        = dim_for_head * heads
        self.scale   = dim_for_head ** -0.5
        self.norm    = nn.LayerNorm(dim)
        self.attend  = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner * 3, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out  = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_for_head, dim_for_mlp, dropout=0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_for_head, dropout),
                FeedForward(dim, dim_for_mlp, dropout)
            ]) for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x; x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, channels, num_classes, depth,
                 heads, dim, dim_for_head, dim_for_mlp, pool='cls',
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        ih, iw = image_size, image_size
        ph, pw = patch_size, patch_size
        assert ih % ph == 0 and iw % pw == 0
        num_patches = (ih // ph) * (iw // pw)
        patch_dim   = channels * ph * pw
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)
        self.transformer   = Transformer(dim, depth, heads, dim_for_head, dim_for_mlp, dropout)
        self.to_latent     = nn.Identity()

    def forward(self, x):
        x = self.to_patch(x)
        b, n, _ = x.shape
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(self.dropout(x))
        return self.to_latent(x)


class SF2Net(nn.Module):
    """SF2Net: Sequence Feature Fusion Network for Palmprint Verification."""
    def __init__(self, num_classes, vit_floor_num=10, weight=0.7,
                 dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.num_classes   = num_classes
        self.vit_floor_num = vit_floor_num
        self.weight        = weight

        self.feature_extraction = FeatureExtraction(
            channel_in=1, filter_num=36, kernel_size=17, stride=2, padding=8,
            init_ratio=0.5, label_num=num_classes, vit_floor_num=vit_floor_num)

        self.vit_0 = ViT(image_size=30, patch_size=5, channels=vit_floor_num*2,
                         num_classes=num_classes, depth=2, heads=16, dim=128,
                         dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)
        self.vit_1 = ViT(image_size=14, patch_size=2, channels=vit_floor_num*2,
                         num_classes=num_classes, depth=2, heads=16, dim=128,
                         dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)

        self.fc1  = nn.Linear(7424, 2048)
        self.fc2  = nn.Linear(2048, 1024)
        self.vfc1 = nn.Linear(11136, 4096)
        self.vfc2 = nn.Linear(4096, 1024)

        self.dropout = nn.Dropout(p=dropout)
        self.arcface = ArcMarginProduct(1024, num_classes, s=arcface_s, m=arcface_m)

    def _process(self, x):
        feat, seq1, seq2 = self.feature_extraction(x)
        vit1 = self.vit_0(seq1); vit2 = self.vit_1(seq2)
        vit_cat = torch.cat((vit1, vit2), dim=1).flatten(1)
        cnn_out = self.fc2(self.fc1(feat))
        vit_out = self.vfc2(self.vfc1(vit_cat))
        return cnn_out * self.weight + vit_out * (1 - self.weight)

    def forward(self, x, y=None):
        x   = self._process(x)
        out = self.arcface(self.dropout(x), y)
        return out, F.normalize(x, dim=-1)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 1024-d embedding for matching."""
        return F.normalize(self._process(x), p=2, dim=1)


