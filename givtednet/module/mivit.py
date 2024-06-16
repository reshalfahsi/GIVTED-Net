import torch
import torch.nn as nn
import torch.nn.functional as F

from givtednet.module.common import ConvNormAct, SqueezeExcite
from givtednet.module.involution import Involution


class TokenMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.inp_projection = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.involution = Involution(dim, 3)
        self.out_projection = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        x = self.inp_projection(x)
        x = self.involution(x)
        x = self.out_projection(x)
        return x


class InvoFormer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        mlp_ratio=0.25,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.norm = nn.InstanceNorm2d(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TokenMixer(dim),
                        SqueezeExcite(dim, mlp_ratio, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + self.norm(attn(x))
            x = x + self.norm(ff(x))
        return x


class MIViTBlock(nn.Module):
    def __init__(
        self,
        channel,
        dim,
        depth,
        kernel_size=3,
        mlp_ratio=0.25,
        dropout=0.0,
    ):
        super().__init__()

        self.conv1 = ConvNormAct(
            channel,
            channel,
            kernel_size,
            groups=channel,
            norm="in",
            act="relu")
        self.conv2 = ConvNormAct(
            channel + 3,
            dim,
            kernel_size,
            norm="in",
            act="relu")

        self.invoformer = InvoFormer(dim, depth, mlp_ratio, dropout)

        self.conv3 = ConvNormAct(dim, channel, 1, norm="in", act="relu")
        self.conv4 = ConvNormAct(
            2 * channel,
            channel,
            kernel_size,
            norm="in",
            act="relu")

    def forward(self, x, y):
        N, C, H, W = x.shape
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=True)

        # Local representations
        q = self.conv1(x)
        x = self.conv2(torch.cat([q, y], 1))

        # Global representations
        z = self.invoformer(x)

        # Fusion
        z = self.conv3(z)
        z = torch.cat([q, z], 1)
        x = self.conv4(z)

        return x
