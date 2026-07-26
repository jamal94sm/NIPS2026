import torch
import torch.nn as nn


class ppnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv", nn.Conv2d(1, 16, 5, 1))
        self.layer1.add_module("bn",   nn.BatchNorm2d(16))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv",    nn.Conv2d(16, 32, 1, 1))
        self.layer2.add_module("bn",      nn.BatchNorm2d(32, momentum=0.001))
        self.layer2.add_module("sigmoid", nn.Sigmoid())
        self.layer2.add_module("avgpool", nn.AvgPool2d(2, 2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv",    nn.Conv2d(32, 64, 3, 1))
        self.layer3.add_module("bn",      nn.BatchNorm2d(64, momentum=0.001))
        self.layer3.add_module("sigmoid", nn.Sigmoid())
        self.layer3.add_module("avgpool", nn.AvgPool2d(2, 2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module("conv", nn.Conv2d(64, 64, 3, 1))
        self.layer4.add_module("bn",   nn.BatchNorm2d(64, momentum=0.001))
        self.layer4.add_module("relu", nn.ReLU())

        self.layer5 = nn.Sequential()
        self.layer5.add_module("conv",    nn.Conv2d(64, 256, 3, 1))
        self.layer5.add_module("bn",      nn.BatchNorm2d(256, momentum=0.001))
        self.layer5.add_module("relu",    nn.ReLU())
        self.layer5.add_module("maxpool", nn.MaxPool2d(2, 2))

        self.fc1   = nn.Linear(43264, 512)
        self.bn1   = nn.BatchNorm1d(512, momentum=0.001)
        self.relu1 = nn.ReLU()

        self.fc2   = nn.Linear(512, 512)
        self.bn2   = nn.BatchNorm1d(512, momentum=0.001)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.25)

        self.dis = nn.PairwiseDistance(p=2)
        self.fc3 = nn.Linear(512, num_classes)

    def _backbone(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.layer4(x); x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        return x

    def forward(self, x, y=None):
        x = self._backbone(x)
        b   = x.size(0)
        o1  = x[:b // 2, :]
        o2  = x[b // 2:, :]
        dis = self.dis(o1, o2)
        x   = self.drop2(x)
        x   = self.fc3(x)
        return x, dis

    @torch.no_grad()
    def get_embedding(self, x):
        """Raw (unnormalised) 512-d embedding for L2 distance matching."""
        return self._backbone(x)


def contrastive_loss(target, dis, margin, device):
    n  = len(target) // 2
    y1 = target[:n]
    y2 = target[n:]
    y  = torch.zeros(n, device=device)
    y[y1 == y2] = 1.0
    margin_t = torch.full((n,), margin, device=device)
    return torch.mean(
        y * torch.pow(dis, 2)
        + (1 - y) * torch.pow(torch.clamp(margin_t - dis, min=0.0), 2))


