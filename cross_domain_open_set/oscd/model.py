"""
model.py
========
One `Baseline` wrapper class per method, all exposing the same interface
so benchmarking.py can drive every method identically:

    baseline = REGISTRY[name](num_classes, cfg, device)
    optimizer, scheduler = baseline.build_optimizer()
    baseline.train_mode()
    loss = baseline.train_step(batch, optimizer)      # zero_grad/backward/step done inside
    baseline.eval_mode()
    emb = baseline.embed(imgs)                          # for utils.evaluate()
    ckpt = baseline.state_dict(); baseline.load_state_dict(ckpt)

Architectures/losses are imported unchanged (module-level) from models/*.py
(extracted verbatim from the original 10 baseline scripts). Only the
*orchestration* (optimizer/scheduler wiring, loss composition, view
unpacking) lives here, and it mirrors each original script's
run_one_epoch / train_one_epoch exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel

from models import arcface as m_arcface
from models import magface as m_magface
from models import ccnet as m_ccnet
from models import co3net as m_co3net
from models import compnet as m_compnet
from models import ppnet as m_ppnet
from models import palmbridge as m_palmbridge
from models import sf2net as m_sf2net

try:
    from models import convnext as m_convnext
    from models import dino as m_dino
    from pytorch_metric_learning import losses as pml_losses
except ImportError as e:
    # timm / pytorch-metric-learning / dinov2 hub deps not installed --
    # the rest of the registry (8/10 methods) still works without them.
    m_convnext = m_dino = None
    pml_losses = None
    print(f"[model.py] convnext/dino unavailable ({e}); "
          f"other 8 methods are unaffected.")


# ══════════════════════════════════════════════════════════════
#  ArcFace
# ══════════════════════════════════════════════════════════════

class BaselineArcFace:
    view_mode = "single"

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.model = m_arcface.ArcFaceBackbone(
            cfg["pretrained_weights"], freeze_ratio=cfg["freeze_ratio"]).to(device)
        self.criterion = m_arcface.ArcFaceLoss(
            num_classes, embedding_size=512, s=cfg["arcface_s"], m=cfg["arcface_m"]).to(device)
        self.cfg = cfg

    def trainable_params(self):
        return ([p for p in self.model.parameters() if p.requires_grad] +
                [p for p in self.criterion.parameters() if p.requires_grad])

    def build_optimizer(self):
        opt = optim.AdamW(self.trainable_params(), lr=self.cfg["lr"],
                           weight_decay=self.cfg["weight_decay"])
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg["num_epochs"], eta_min=1e-6)
        return opt, sch

    def train_mode(self): self.model.train(); self.criterion.train()
    def eval_mode(self): self.model.eval(); self.criterion.eval()

    def train_step(self, batch, optimizer):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        emb = self.model(imgs)
        loss = self.criterion(emb, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params(), 5.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        return self.model(imgs)

    def state_dict(self):
        return {"model": self.model.state_dict(), "criterion": self.criterion.state_dict()}

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.criterion.load_state_dict(ckpt["criterion"])


# ══════════════════════════════════════════════════════════════
#  MagFace
# ══════════════════════════════════════════════════════════════

class BaselineMagFace:
    view_mode = "single"

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.model = m_magface.MagFaceBackbone(
            cfg["pretrained_weights"], freeze_ratio=cfg["freeze_ratio"]).to(device)
        self.criterion = m_magface.MagFaceLoss(
            num_classes, embedding_size=512, s=cfg["arc_s"], m_l=cfg["m_l"], m_u=cfg["m_u"],
            l_a=cfg["l_a"], u_a=cfg["u_a"], lambda_g=cfg["lambda_g"]).to(device)
        self.cfg = cfg

    def trainable_params(self):
        return ([p for p in self.model.parameters() if p.requires_grad] +
                [p for p in self.criterion.parameters() if p.requires_grad])

    def build_optimizer(self):
        opt = optim.AdamW(self.trainable_params(), lr=self.cfg["lr"],
                           weight_decay=self.cfg["weight_decay"])
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg["num_epochs"], eta_min=1e-6)
        return opt, sch

    def train_mode(self): self.model.train(); self.criterion.train()
    def eval_mode(self): self.model.eval(); self.criterion.eval()

    def train_step(self, batch, optimizer):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        emb = self.model(imgs)                    # raw, unnormalised (MagFace needs norm)
        loss, l_arc, l_g = self.criterion(emb, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params(), 5.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        return self.model(imgs)     # utils.evaluate() L2-normalises before cosine sim

    def state_dict(self):
        return {"model": self.model.state_dict(), "criterion": self.criterion.state_dict()}

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.criterion.load_state_dict(ckpt["criterion"])


# ══════════════════════════════════════════════════════════════
#  CompNet  (plain CE via ArcMarginProduct)
# ══════════════════════════════════════════════════════════════

class BaselineCompNet:
    view_mode = "single"

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.net = m_compnet.CompNet(
            num_classes, embedding_dim=cfg["embedding_dim"],
            arcface_s=cfg["arcface_s"], arcface_m=cfg["arcface_m"],
            dropout=cfg["dropout"]).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

    def build_optimizer(self):
        opt = optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        sch = lr_scheduler.StepLR(opt, self.cfg["lr_step"], self.cfg["lr_gamma"])
        return opt, sch

    def train_mode(self): self.net.train()
    def eval_mode(self): self.net.eval()

    def train_step(self, batch, optimizer):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        out = self.net(imgs, labels)
        loss = self.criterion(out, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        return self.net.get_embedding(imgs)

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, ckpt):
        self.net.load_state_dict(ckpt["net"])


# ══════════════════════════════════════════════════════════════
#  PPNet  (CE + L2 reg + Siamese contrastive + dis^2)
# ══════════════════════════════════════════════════════════════

class BaselinePPNet:
    view_mode = "single"

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.net = m_ppnet.ppnet(num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

    def build_optimizer(self):
        opt = optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        sch = lr_scheduler.StepLR(opt, self.cfg["lr_step"], self.cfg["lr_gamma"])
        return opt, sch

    def train_mode(self): self.net.train()
    def eval_mode(self): self.net.eval()

    def train_step(self, batch, optimizer):
        imgs, labels = batch
        if len(labels) % 2 != 0:      # PPNet's Siamese split needs an even batch
            labels = torch.cat((labels, labels[0:1]), dim=0)
            imgs = torch.cat((imgs, imgs[0:1]), dim=0)
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        output, dis = self.net(imgs)
        cross = self.criterion(output, labels)
        l2_reg = torch.norm(self.net.fc2.weight, 2) + torch.norm(self.net.fc3.weight, 2)
        contra = m_ppnet.contrastive_loss(labels, dis, self.cfg["contrastive_margin"], self.device)
        loss = (cross + self.cfg["w_l2"] * l2_reg + self.cfg["w_contra"] * contra
                + self.cfg["w_dis"] * torch.mean(dis ** 2))
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        return self.net.get_embedding(imgs)

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, ckpt):
        self.net.load_state_dict(ckpt["net"])


# ══════════════════════════════════════════════════════════════
#  PalmBridge  (ArcFace + codebook consistency + orthogonality, warmup)
# ══════════════════════════════════════════════════════════════

class BaselinePalmBridge:
    view_mode = "single"

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.backbone = m_palmbridge.PalmBridgeBackbone(
            num_classes, feature_dim=cfg["feature_dim"],
            num_gabor_filters=cfg["num_gabor_filters"],
            gabor_kernel_size=cfg["gabor_kernel_size"]).to(device)
        self.palmbridge = m_palmbridge.PalmBridge(
            num_pb_vectors=cfg["num_pb_vectors"], feature_dim=cfg["feature_dim"],
            w_ori=cfg["w_ori"], w_map=cfg["w_map"], lambda_con=cfg["lambda_con"]).to(device)
        self.arcface = m_palmbridge.ArcFaceLoss(
            num_classes, feature_dim=cfg["feature_dim"], s=cfg["arc_s"], m=cfg["arc_m"]).to(device)
        self.cfg = cfg
        self.epoch = 0    # bumped externally by benchmarking.py each epoch (drives warmup)
        self.apply_pb_at_eval = True   # PalmBridge-enhanced eval (matches paper's default)

    def build_optimizer(self):
        params = (list(self.backbone.parameters()) + list(self.palmbridge.parameters())
                  + list(self.arcface.parameters()))
        opt = optim.Adam(params, lr=self.cfg["lr"], weight_decay=5e-4)
        warmup = self.cfg["warmup_epochs"]
        total = self.cfg["num_epochs"]
        def lr_lambda(epoch):
            if epoch < warmup:
                return 0.1 + 0.9 * (epoch / max(warmup - 1, 1))
            import math
            progress = (epoch - warmup) / max(total - warmup, 1)
            return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return opt, sch

    def train_mode(self): self.backbone.train(); self.palmbridge.train(); self.arcface.train()
    def eval_mode(self): self.backbone.eval(); self.palmbridge.eval(); self.arcface.eval()

    def train_step(self, batch, optimizer):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        pb_active = self.epoch > self.cfg["warmup_epochs"]
        optimizer.zero_grad()
        z = self.backbone(imgs)
        z_hat, z_tilde, _ = self.palmbridge(z)
        feat_for_arc = z_hat if pb_active else z
        L_bak = self.arcface(feat_for_arc, labels)
        L_con = self.palmbridge.loss_consistency(z, z_tilde)
        L_o = self.palmbridge.loss_orthogonal()
        loss = L_bak + self.cfg["alpha"] * L_con + self.cfg["beta"] * L_o
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.palmbridge.parameters()), max_norm=10.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        z = self.backbone(imgs)
        if self.apply_pb_at_eval:
            z_hat, _, _ = self.palmbridge(z)
            return z_hat
        return z

    def state_dict(self):
        return {"backbone": self.backbone.state_dict(),
                "palmbridge": self.palmbridge.state_dict(),
                "arcface": self.arcface.state_dict()}

    def load_state_dict(self, ckpt):
        self.backbone.load_state_dict(ckpt["backbone"])
        self.palmbridge.load_state_dict(ckpt["palmbridge"])
        self.arcface.load_state_dict(ckpt["arcface"])


# ══════════════════════════════════════════════════════════════
#  CCNet / CO3Net  (CE via ArcMarginProduct + SupCon, paired views)
# ══════════════════════════════════════════════════════════════

class _BaselinePairedCE_SupCon:
    """Shared implementation for CCNet and CO3Net (identical composite
    loss / training pattern in the original code, differing only in
    backbone architecture)."""
    view_mode = "paired"
    _net_cls = None
    _supcon_cls = None

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.net = self._net_cls(
            num_classes=num_classes, dropout=cfg["dropout"],
            arcface_s=cfg["arcface_s"], arcface_m=cfg["arcface_m"],
            **({"weight": cfg["comp_weight"]} if "comp_weight" in cfg else {})
        ).to(device)
        if torch.cuda.device_count() > 1:
            self.net = DataParallel(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.con_criterion = self._supcon_cls(
            temperature=cfg["temperature"], base_temperature=cfg["temperature"])
        self.cfg = cfg

    def build_optimizer(self):
        opt = optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        sch = lr_scheduler.StepLR(opt, self.cfg["lr_step"], self.cfg["lr_gamma"])
        return opt, sch

    def train_mode(self): self.net.train()
    def eval_mode(self): self.net.eval()

    def train_step(self, batch, optimizer):
        (img1, img2), labels = batch
        img1, img2 = img1.to(self.device), img2.to(self.device)
        labels = labels.to(self.device)
        optimizer.zero_grad()
        output, fe1 = self.net(img1, labels)
        _, fe2 = self.net(img2, labels)
        fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        ce_loss = self.criterion(output, labels)
        con_loss = self.con_criterion(fe, labels)
        loss = self.cfg["ce_weight"] * ce_loss + self.cfg["con_weight"] * con_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        m = self.net.module if isinstance(self.net, DataParallel) else self.net
        return m.get_embedding(imgs)

    def state_dict(self):
        m = self.net.module if isinstance(self.net, DataParallel) else self.net
        return {"net": m.state_dict()}

    def load_state_dict(self, ckpt):
        m = self.net.module if isinstance(self.net, DataParallel) else self.net
        m.load_state_dict(ckpt["net"])


class BaselineCCNet(_BaselinePairedCE_SupCon):
    _net_cls = m_ccnet.ccnet
    _supcon_cls = m_ccnet.SupConLoss


class BaselineCO3Net(_BaselinePairedCE_SupCon):
    _net_cls = m_co3net.co3net
    _supcon_cls = m_co3net.SupConLoss

    def __init__(self, num_classes, cfg, device):
        # co3net's constructor doesn't take `weight` (no comp_weight arg)
        self.device = device
        self.net = m_co3net.co3net(
            num_classes=num_classes, dropout=cfg["dropout"],
            arcface_s=cfg["arcface_s"], arcface_m=cfg["arcface_m"]).to(device)
        if torch.cuda.device_count() > 1:
            self.net = DataParallel(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.con_criterion = m_co3net.SupConLoss(
            temperature=cfg["temperature"], base_temperature=cfg["temperature"])
        self.cfg = cfg


# ══════════════════════════════════════════════════════════════
#  SF2Net  (CE via ArcMarginProduct + Triplet(SRT), triplet views)
# ══════════════════════════════════════════════════════════════

class BaselineSF2Net:
    view_mode = "triplet"

    def __init__(self, num_classes, cfg, device):
        self.device = device
        self.net = m_sf2net.SF2Net(
            num_classes, vit_floor_num=cfg["vit_floor_num"], dropout=cfg["dropout"],
            arcface_s=cfg["arcface_s"], arcface_m=cfg["arcface_m"]).to(device)
        if torch.cuda.device_count() > 1:
            self.net = DataParallel(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.tl_criterion = m_sf2net.TripletLoss(margin=cfg["triplet_margin"], distance="SRT")
        self.cfg = cfg

    def build_optimizer(self):
        opt = optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        sch = lr_scheduler.StepLR(opt, self.cfg["lr_step"], self.cfg["lr_gamma"])
        return opt, sch

    def train_mode(self): self.net.train()
    def eval_mode(self): self.net.eval()

    def train_step(self, batch, optimizer):
        (anchor, positive, negative), (t_a, t_p, t_n) = batch
        anchor, positive, negative = (anchor.to(self.device), positive.to(self.device),
                                        negative.to(self.device))
        t_a, t_p, t_n = t_a.to(self.device), t_p.to(self.device), t_n.to(self.device)
        optimizer.zero_grad()
        out_a, fe_a = self.net(anchor, t_a)
        out_p, fe_p = self.net(positive, t_p)
        out_n, fe_n = self.net(negative, t_n)
        ce_loss = self.criterion(out_a, t_a)
        tl_loss, _, _, _ = self.tl_criterion(fe_a, fe_p, fe_n)
        loss = self.cfg["ce_weight"] * ce_loss + self.cfg["tl_weight"] * tl_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        m = self.net.module if isinstance(self.net, DataParallel) else self.net
        return m.get_embedding(imgs)

    def state_dict(self):
        m = self.net.module if isinstance(self.net, DataParallel) else self.net
        return {"net": m.state_dict()}

    def load_state_dict(self, ckpt):
        m = self.net.module if isinstance(self.net, DataParallel) else self.net
        m.load_state_dict(ckpt["net"])


# ══════════════════════════════════════════════════════════════
#  ConvNeXt / DINO  (ArcFace(pml) + SupCon(pml), 2-view paired batch --
#  reduced from the original's 4-view [base,aug1,aug2,aug3] per the
#  augmentation-fairness pass; see config.py + README.md)
# ══════════════════════════════════════════════════════════════

class _BaselineArcSupConBackbone:
    view_mode = "paired"
    _backbone_cls = None

    def __init__(self, num_classes, cfg, device):
        if pml_losses is None:
            raise ImportError("pip install pytorch-metric-learning")
        self.device = device
        self.model = self._backbone_cls().to(device)
        self.proj = (m_convnext.ProjectionHead(self.model.embed_dim)
                     if self._backbone_cls is m_convnext.ConvNeXtFinetune
                     else m_dino.ProjectionHead(self.model.embed_dim)).to(device)
        self.criterion_arc = pml_losses.ArcFaceLoss(
            num_classes=num_classes, embedding_size=self.model.embed_dim,
            margin=cfg["margin"], scale=cfg["scale"]).to(device)
        self.criterion_supcon = pml_losses.SupConLoss(temperature=cfg["supcon_temperature"]).to(device)
        self.cfg = cfg

    def all_params(self):
        return (list(self.model.parameters()) + list(self.proj.parameters())
                + list(self.criterion_arc.parameters()))

    def build_optimizer(self):
        opt = optim.AdamW(self.all_params(), lr=self.cfg["lr"], weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg["num_epochs"], eta_min=1e-5)
        return opt, sch

    def train_mode(self): self.model.train(); self.proj.train(); self.criterion_arc.train()
    def eval_mode(self): self.model.eval(); self.proj.eval(); self.criterion_arc.eval()

    def train_step(self, batch, optimizer):
        (aug1, aug2), labels = batch
        aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
        labels = labels.to(self.device)
        imgs_all = torch.cat([aug1, aug2], dim=0)
        y_all = torch.cat([labels, labels], dim=0)
        optimizer.zero_grad()
        emb_all = self.model(imgs_all)
        proj_all = self.proj(emb_all)
        loss_arc = self.criterion_arc(emb_all, y_all)
        loss_con = self.criterion_supcon(proj_all, y_all)
        loss = loss_arc + self.cfg["lambda_supcon"] * loss_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params(), 5.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def embed(self, imgs):
        return self.model(imgs)

    def state_dict(self):
        return {"model": self.model.state_dict(), "proj": self.proj.state_dict(),
                "criterion_arc": self.criterion_arc.state_dict()}

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.proj.load_state_dict(ckpt["proj"])
        self.criterion_arc.load_state_dict(ckpt["criterion_arc"])


if m_convnext is not None:
    class BaselineConvNeXt(_BaselineArcSupConBackbone):
        _backbone_cls = m_convnext.ConvNeXtFinetune

    class BaselineDINO(_BaselineArcSupConBackbone):
        _backbone_cls = m_dino.DINOFinetune
else:
    BaselineConvNeXt = BaselineDINO = None


# ══════════════════════════════════════════════════════════════
#  REGISTRY
# ══════════════════════════════════════════════════════════════

REGISTRY = {
    "arcface":    BaselineArcFace,
    "magface":    BaselineMagFace,
    "compnet":    BaselineCompNet,
    "ppnet":      BaselinePPNet,
    "palmbridge": BaselinePalmBridge,
    "ccnet":      BaselineCCNet,
    "co3net":     BaselineCO3Net,
    "sf2net":     BaselineSF2Net,
}
if BaselineConvNeXt is not None:
    REGISTRY["convnext"] = BaselineConvNeXt
    REGISTRY["dino"] = BaselineDINO
