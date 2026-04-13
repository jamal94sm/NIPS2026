"""
train.py — Training and evaluation loop helpers.
"""

import torch


def run_one_epoch(model, loader, criterion, con_criterion,
                  optimizer, device, phase,
                  ce_weight=0.8, con_weight=0.2):
    """
    Run one epoch of training or evaluation.

    Parameters
    ----------
    phase : "training" | "testing"

    Returns
    -------
    (epoch_loss, epoch_acc)
    """
    is_train = (phase == "training")
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for datas, target in loader:
            data1  = datas[0].to(device)
            data2  = datas[1].to(device)
            target = target.to(device)

            if is_train:
                optimizer.zero_grad()

            out1, fe1 = model(data1, target if is_train else None)
            out2, fe2 = model(data2, target if is_train else None)
            fe        = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

            ce_loss  = criterion(out1, target)
            con_loss = con_criterion(fe, target)
            loss     = ce_weight * ce_loss + con_weight * con_loss

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss    += loss.item() * data1.size(0)
            running_correct += out1.data.max(1)[1].eq(target).sum().item()
            total           += data1.size(0)

    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)
