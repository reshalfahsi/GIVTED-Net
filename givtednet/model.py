import torch
import torch.nn as nn
import torch.nn.functional as F

from givtednet.module.ghostnet import GhostBottleneck
from givtednet.module.common import ConvBnAct
from givtednet.module.mobilevit import MobileViTBlock


class GIVTEDNet(nn.Module):
    def __init__(self, dropout=0.0):
        super(GIVTEDNet, self).__init__()

        self.encoder0 = nn.Sequential(
            ConvBnAct(3, 16, 3, 2),
            GhostBottleneck(16, 16, 16),
        )
        self.encoder1 = nn.Sequential(
            GhostBottleneck(16, 32, 24, stride=2),
            GhostBottleneck(24, 40, 24, se_ratio=0.25),
        )
        self.encoder2 = nn.Sequential(
            GhostBottleneck(24, 48, 32, stride=2),
            GhostBottleneck(32, 64, 32, se_ratio=0.25),
        )
        self.encoder3 = nn.Sequential(
            GhostBottleneck(32, 72, 40, stride=2),
            GhostBottleneck(40, 80, 40, se_ratio=0.25),
        )

        self.latent = nn.Sequential(
            GhostBottleneck(40, 96, 64, stride=2),
            GhostBottleneck(64, 112, 64, se_ratio=0.25),
            GhostBottleneck(64, 112, 64, se_ratio=0.25),
        )

        self.convdec3 = ConvBnAct(64, 40, 1)
        self.convdec2 = ConvBnAct(40, 32, 1)
        self.convdec1 = ConvBnAct(32, 24, 1)
        self.convdec0 = ConvBnAct(24, 16, 1)

        DIM = [50, 40, 30, 20]
        DEPTH = [1, 1, 2, 2]

        self.decoder3 = MobileViTBlock(
            channel=40,
            dim=DIM[0],
            depth=DEPTH[0],
            dropout=dropout,
        )
        self.decoder2 = MobileViTBlock(
            channel=32,
            dim=DIM[1],
            depth=DEPTH[1],
            dropout=dropout,
        )
        self.decoder1 = MobileViTBlock(
            channel=24,
            dim=DIM[2],
            depth=DEPTH[2],
            dropout=dropout,
        )
        self.decoder0 = MobileViTBlock(
            channel=16,
            dim=DIM[3],
            depth=DEPTH[3],
            dropout=dropout,
        )

        self.out = nn.Conv2d(
            16,
            1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x):

        y = x.clone()
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        x = self.latent(enc3)

        x = self.convdec3(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        )

        x = x + enc3
        x = self.decoder3(x, y)

        x = self.convdec2(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        )

        x = x + enc2
        x = self.decoder2(x, y)

        x = self.convdec1(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        )

        x = x + enc1
        x = self.decoder1(x, y)

        x = self.convdec0(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        )

        x = x + enc0
        x = self.decoder0(x, y)

        x = self.out(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        )
        return x
