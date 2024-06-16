import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from givtednet.module.common import ConvBnAct, SqueezeExcite


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, cheap_kernel=3, stride=1, ratio=2, bn=True, act=True
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvBnAct(
            inp,
            init_channels,
            kernel_size,
            stride=stride,
            bn=bn,
            act=act,
        )

        self.cheap_operation = ConvBnAct(
            init_channels,
            new_channels,
            cheap_kernel,
            stride=1,
            groups=init_channels,
            bn=bn,
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
        self.ghost1 = GhostModule(in_chs, mid_chs, act=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = ConvBnAct(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                groups=mid_chs,
                bn=True,
                act=False,
            )

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, act=False)

        # shortcut
        self.shortcut = nn.Sequential(
            ConvBnAct(
                in_chs,
                in_chs,
                dw_kernel_size,
                stride=stride,
                groups=in_chs,
                bn=True,
                act=False,
            ),
            ConvBnAct(
                in_chs,
                out_chs,
                1,
                stride=1,
                bn=True,
                act=False,
            ),
        ) if (in_chs == out_chs and self.stride == 1) else None

    def forward(self, x):
        if self.shortcut is not None:
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
        return x
