import torch
import torch.nn as nn
import torch.nn.functional as F

from givtednet.module.common import ConvBnAct, _make_divisible


class Involution(nn.Module):
    def __init__(
        self,
        channel_in,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        reduction_ratio=0.25,
        divisor=4,
    ):
        super(Involution, self).__init__()

        reduced_chs = _make_divisible(channel_in * reduction_ratio, divisor)
        
        self.o = nn.AvgPool2d(stride, stride) if stride > 1 else None
        self.reduce = ConvBnAct(channel_in, reduced_chs, 1, bn=False, act=True)
        self.span = nn.Conv2d(reduced_chs, kernel_size * kernel_size * groups, 1)
        self.unfold = nn.Unfold(kernel_size, dilation, kernel_size // 2, stride)

        self.groups = groups
        self.kernel_size = kernel_size

    def forward(self, x):
        B, C, H, W = x.shape
        x_unfolded = self.unfold(x)  # B,CxKxK,HxW
        x_unfolded = x_unfolded.view(
            B, self.groups, C // self.groups, self.kernel_size * self.kernel_size, H, W
        )

        # kernel generation, Eqn.(6)
        if self.o is not None:
            x = self.o(x)
        kernel = self.reduce(x)
        kernel = self.span(kernel)
        kernel = kernel.view(
            B, self.groups, self.kernel_size * self.kernel_size, H, W
        ).unsqueeze(2)

        # apply softmax
        kernel = F.softmax(kernel, dim=3)

        # Multiply-Add operation, Eqn.(4)
        out = torch.mul(kernel, x_unfolded).sum(dim=3)  # B,G,C/G,H,W
        out = out.view(B, C, H, W)
        return out
