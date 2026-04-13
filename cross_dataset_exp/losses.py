"""
losses.py — Loss functions used in CO3Net training.
"""

import torch
import torch.nn as nn


def batch_hard_triplet_loss(embeddings, labels, margin=0.25):
    """
    Online batch-hard triplet loss on L2-normalised embeddings.
    For each anchor, selects:
      - hardest positive  : same class,      largest  arc-distance
      - hardest negative  : different class, smallest arc-distance

    embeddings : (N, D) — must be L2-normalised
    labels     : (N,)   — class indices
    margin     : float  — triplet margin
    Returns    : scalar loss
    """
    # pairwise arc-distances  (N, N)
    cos_sim = torch.clamp(embeddings @ embeddings.T, -1 + 1e-6, 1 - 1e-6)
    dist    = torch.acos(cos_sim) / torch.pi          # in [0, 1]

    labels  = labels.view(-1)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N, N) bool
    mask_neg = ~mask_pos

    # mask out self-pairs from positives
    eye      = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    mask_pos = mask_pos & ~eye

    # hardest positive per anchor
    pos_dist = (dist * mask_pos.float()).max(dim=1)[0]

    # hardest negative per anchor (fill positives with inf first)
    neg_dist = dist.masked_fill(~mask_neg, float("inf")).min(dim=1)[0]

    loss = torch.relu(pos_dist - neg_dist + margin).mean()
    return loss



class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    Also supports unsupervised contrastive loss (SimCLR) when labels=None.
    Exact copy from CO3Net/loss.py.
    """

    def __init__(self, temperature=0.07, contrast_mode="all",
                 base_temperature=0.07):
        super().__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` must be [bsz, n_views, ...] "
                             "(at least 3 dimensions).")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`.")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count   = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count   = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits         = anchor_dot_contrast - logits_max.detach()

        # tile mask and mask-out self-contrast cases
        mask        = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits        = torch.exp(logits) * logits_mask
        log_prob          = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
