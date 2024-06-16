import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from givtednet.module.common import ConvNormAct, SqueezeExcite


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, cheap_kernel=3, stride=1, ratio=2, norm="bn", act="relu"
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvNormAct(
            inp,
            init_channels,
            kernel_size,
            stride=stride,
            norm=norm,
            act=act,
        )

        self.cheap_operation = ConvNormAct(
            init_channels,
            new_channels,
            cheap_kernel,
            stride=1,
            groups=init_channels,
            norm=norm,
            act=act,
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0.0,
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, act="relu")

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = ConvNormAct(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                groups=mid_chs,
                norm="bn",
                act=None,
            )

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, act=None)

        # shortcut
        self.shortcut = nn.Sequential(
            ConvNormAct(
                in_chs,
                in_chs,
                dw_kernel_size,
                stride=stride,
                groups=in_chs,
                norm="bn",
                act=None,
            ),
            ConvNormAct(
                in_chs,
                out_chs,
                1,
                stride=1,
                norm="bn",
                act=None,
            ),
        ) if not (in_chs == out_chs and self.stride == 1) else None

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        if self.shortcut is not None:
            x += self.shortcut(residual)
        else:
            x += residual
        return x
