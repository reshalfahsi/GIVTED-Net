import torch
import torch.nn as nn
import torch.nn.functional as F

from givtednet.module.common import _make_divisible, SqueezeExcite
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


class Transformer(nn.Module):
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


class MobileViTBlock(nn.Module):
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channel,
                channel,
                kernel_size,
                padding=kernel_size // 2,
                groups=channel,
                bias=False,
            ),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channel + 3, dim, kernel_size, padding=kernel_size // 2, bias=False
            ),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.transformer = Transformer(dim, depth, mlp_ratio, dropout)

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, channel, 1, bias=False),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                2 * channel, channel, kernel_size, padding=kernel_size // 2, bias=False
            ),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        N, C, H, W = x.shape
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=True)

        # Local representations
        q = self.conv1(x)
        x = self.conv2(torch.cat([q, y], 1))

        # Global representations
        z = self.transformer(x)

        # Fusion
        z = self.conv3(z)
        z = torch.cat([q, z], 1)
        x = self.conv4(z)

        return x
