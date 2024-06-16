import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_sigmoid(x, inplace: bool = True):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).mul_(0.16666666666666666)
    else:
        return F.relu6(x + 3.0) * 0.16666666666666666


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            norm="bn",
            act="relu"):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=groups,
            bias=(norm is None),
        )
        assert norm in [
            "bn", "in", None], f"Normalization is not supported: {norm}"
        self.norm = norm
        if norm is not None:
            self.norm = nn.BatchNorm2d(
                out_chs) if self.norm == "bn" else nn.InstanceNorm2d(out_chs)
        assert act in ["relu", "hard_sigmoid",
                       None], f"Activation function is not supported: {act}"
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act == "relu":
            x = torch.relu_(x)
        elif self.act == "hard_sigmoid":
            x = hard_sigmoid(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        divisor=4,
        dropout=0.0,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        reduced_chs = _make_divisible(in_chs * se_ratio, divisor)
        self.conv_reduce = ConvNormAct(
            in_chs, reduced_chs, 1, norm=None, act="relu")
        self.conv_expand = ConvNormAct(
            reduced_chs, in_chs, 1, norm=None, act="hard_sigmoid")
        self.dropout = dropout

    def forward(self, x):
        x_se = torch.mean(x, dim=[2, 3], keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = F.dropout2d(x_se, self.dropout, self.training)
        x_se = self.conv_expand(x_se)
        x = x * x_se
        return x
