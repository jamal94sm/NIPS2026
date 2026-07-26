import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborLayer(nn.Module):
    def __init__(self, num_filters=32, kernel_size=15):
        super().__init__()
        n  = num_filters
        ks = kernel_size
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, n + 1)[:-1])
        self.sigma = nn.Parameter(torch.full((n,), 3.0))
        self.lambd = nn.Parameter(torch.full((n,), 6.0))
        self.psi   = nn.Parameter(torch.zeros(n))
        self.gamma = nn.Parameter(torch.full((n,), 0.5))
        half = ks // 2
        ys   = torch.arange(-half, half + 1, dtype=torch.float32)
        xs   = torch.arange(-half, half + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
        self.ks = ks

    def _build_filters(self) -> torch.Tensor:
        theta = self.theta
        sigma = self.sigma.abs().clamp(min=0.5)
        lambd = self.lambd.abs().clamp(min=1.0)
        psi   = self.psi
        gamma = self.gamma.abs().clamp(min=0.1)
        xx = self.xx.unsqueeze(0)
        yy = self.yy.unsqueeze(0)
        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)
        sigma = sigma.view(-1, 1, 1)
        lambd = lambd.view(-1, 1, 1)
        psi   = psi.view(-1, 1, 1)
        gamma = gamma.view(-1, 1, 1)
        x_rot =  xx * cos_t + yy * sin_t
        y_rot = -xx * sin_t + yy * cos_t
        envelope = torch.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2.0 * sigma**2))
        kernel   = envelope * torch.cos(2.0 * math.pi * x_rot / lambd + psi)
        kernel   = kernel - kernel.mean(dim=(1, 2), keepdim=True)
        return kernel.unsqueeze(1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filters = self._build_filters()
        return F.conv2d(x, filters, padding=self.ks // 2)


class CompetitivePool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = x.abs().max(dim=1, keepdim=True)
        return out


class PalmBridgeBackbone(nn.Module):
    """PalmBridge's CompNet-style backbone (renamed from `CompNet` in the
    original file to avoid confusion with the separate CompNet baseline)."""
    def __init__(self, num_classes: int, feature_dim=512,
                 num_gabor_filters=32, gabor_kernel_size=15):
        super().__init__()
        self.gabor   = LearnableGaborLayer(num_gabor_filters, gabor_kernel_size)
        self.compete = CompetitivePool()
        self.gbn     = nn.BatchNorm2d(1)

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = _block(1,   32);  self.pool1 = nn.MaxPool2d(2)
        self.block2 = _block(32,  64);  self.pool2 = nn.MaxPool2d(2)
        self.block3 = _block(64,  128); self.pool3 = nn.MaxPool2d(2)
        self.block4 = _block(128, 256); self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.embed  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        z = self.embed(f)
        return F.normalize(z, p=2, dim=1)


class PalmBridge(nn.Module):
    def __init__(self, num_pb_vectors=512, feature_dim=512, w_ori=0.7, w_map=0.3,
                 lambda_con=0.25):
        super().__init__()
        self.K = num_pb_vectors
        self.w_ori = w_ori
        self.w_map = w_map
        self.lambda_con = lambda_con
        self.P = nn.Parameter(
            F.normalize(torch.randn(num_pb_vectors, feature_dim), p=2, dim=1))

    def _nearest_vector(self, z):
        dists = (z.pow(2).sum(1, keepdim=True)
                 + self.P.pow(2).sum(1).unsqueeze(0)
                 - 2.0 * (z @ self.P.t()))
        idx     = dists.argmin(dim=1)
        z_tilde = self.P[idx]
        return z_tilde, idx

    def _blend(self, z, z_tilde):
        return self.w_ori * z + self.w_map * z_tilde

    def forward(self, z):
        z_tilde, indices = self._nearest_vector(z)
        z_hat            = self._blend(z, z_tilde)
        return z_hat, z_tilde, indices

    def loss_consistency(self, z, z_tilde):
        t1 = (z_tilde - z.detach()).pow(2).sum(1).mean()
        t2 = (z - z_tilde.detach()).pow(2).sum(1).mean()
        return t1 + self.lambda_con * t2

    def loss_orthogonal(self):
        W = F.normalize(self.P, p=2, dim=1)
        S = W @ W.t()
        I = torch.eye(self.K, device=S.device, dtype=S.dtype)
        return ((S - I).pow(2)).sum() / (self.K ** 2)

    @torch.no_grad()
    def codebook_usage(self):
        W   = F.normalize(self.P, p=2, dim=1)
        S   = W @ W.t()
        off = S[~torch.eye(self.K, dtype=torch.bool, device=S.device)]
        return {"mean_cosine":    off.mean().item(),
                "near_duplicate": (off > 0.9).float().mean().item()}


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, feature_dim=512, s=48.0, m=0.40):
        super().__init__()
        self.s     = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z, labels):
        W           = F.normalize(self.weight, p=2, dim=1)
        cos_theta   = (z @ W.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta   = (1.0 - cos_theta**2).sqrt()
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits  = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return self.ce(logits, labels)


