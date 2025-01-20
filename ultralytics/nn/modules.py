# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Common modules
"""

import math
import sys
import os
import re
import torch
import torch.nn as nn
from typing import Dict, Union
from pathlib import Path
from types import SimpleNamespace

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
from ultralytics.yolo.utils import DEFAULT_CFG, DEFAULT_CFG_DICT


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(
            k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1,
                              c2,
                              k,
                              s,
                              autopad(k, p, d),
                              groups=g,
                              dilation=d,
                              bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(
            self,
            c1,
            c2,
            k=1,
            s=1,
            d=1,
            act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p1=0,
                 p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    # Convolution transpose 2d layer
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        return self.act(self.conv_transpose(x))


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1,
                                a).transpose(2, 1).softmax(1)).view(b, 4, a)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads)
                                  for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(
            b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 k=(3, 3),
                 e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0)
                                 for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
              for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=False,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(
            self.cv1(
                torch.cat([
                    torch.mean(x, 1, keepdim=True),
                    torch.max(x, 1, keepdim=True)[0]
                ], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class C1(nn.Module):
    # CSP Bottleneck with 1 convolution
    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        y = self.cv1(x)
        return self.m(y) + y


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(
            Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1)
            for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
                       x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(
            c1, c1, k, s, act=False), Conv(
                c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Proto(nn.Module):
    # YOLOv8 mask Proto module for segmentation models
    def __init__(self,
                 c1,
                 c_=256,
                 c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(
            c_, c_, 2, 2, 0,
            bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class CustomConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1,
                              c2,
                              k,
                              s,
                              autopad(k, p, d),
                              groups=g,
                              dilation=d,
                              bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Add different versions of the hierarchical architecture

class Detect1(nn.Module):  # Hierarchical Arch Version 1
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.hier = hier
        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    prev_h_len = len(self.hier[str(k - 1)])
                    mod_list += [
                        nn.Sequential(Conv(x + prev_h_len, c3_h, 3),
                                      Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x + len(self.hier[str(len(self.hier) -
                                               2)]), c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3, len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []

        for i in range(self.nl):
            for l in range(len(self.hier) - 1):
                if l == 0:
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i](x[i]))
                else:
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i](
                        torch.cat((x[i], out_dict[f'cv3_level{l-1}_out'][i]),
                                  1)))

            out_dict[f"cv3_level{len(self.hier) - 1}_out"].append(
                self.cv3[len(self.hier) - 1][i](torch.cat(
                    (x[i], out_dict[f'cv3_level{len(self.hier) - 2}_out'][i]),
                    1)))
            x[i] = torch.cat(
                (self.cv2[i](x[i]), self.cv3[len(self.hier) - 1][i](torch.cat(
                    (x[i], out_dict[f'cv3_level{len(self.hier)-2}_out'][i]),
                    1))), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level
        return y if self.export else (x, out_dict, y_hier)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s)**2)


class Detect2(nn.Module): # Hierarchical Arch Version 2
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.hier = hier
        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    prev_h_len = len(self.hier[str(k - 1)])
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h + prev_h_len, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3 + len(self.hier[str(len(self.hier) - 2)]),
                              len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
        for i in range(self.nl):
            for l in range(len(self.hier) - 1):
                if l == 0:
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i](x[i]))
                else:
                    out0 = self.cv3[l][i][0](x[i])
                    out1 = self.cv3[l][i][1](out0)
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i][2](
                        torch.cat((out1, out_dict[f'cv3_level{l - 1}_out'][i]),
                                  1)))
            out_last0 = self.cv3[len(self.hier) - 1][i][0](x[i])
            out_last1 = self.cv3[len(self.hier) - 1][i][1](out_last0)
            out_dict[f"cv3_level{len(self.hier) - 1}_out"].append(
                self.cv3[len(self.hier) - 1][i][2](torch.cat(
                    (out_last1,
                     out_dict[f'cv3_level{len(self.hier) - 2}_out'][i]), 1)))
            x[i] = torch.cat((self.cv2[i](
                x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)
        
        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level
        return y if self.export else (x, out_dict, y_hier)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s)**2)


class Detect3(nn.Module): # Hierarchical Arch Version 3
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.hier = hier
        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3),
                                      Conv(c3_h * 2, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3 * 2, c3, 3),
                    nn.Conv2d(c3, len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        in_2nd_conv = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
            in_2nd_conv[f"cv3_level{l}_in_2nd_conv"] = []
        for i in range(self.nl):
            for l in range(len(self.hier)):
                if l == 0:
                    out0 = self.cv3[l][i][0](x[i])
                    in_2nd_conv[f"cv3_level{l}_in_2nd_conv"].append(out0)
                    out1 = self.cv3[l][i][1](out0)
                    out_dict[f"cv3_level{l}_out"].append(
                        self.cv3[l][i][2](out1))
                else:
                    out0 = self.cv3[l][i][0](x[i])
                    in_2nd_conv[f"cv3_level{l}_in_2nd_conv"].append(out0)
                    out1 = self.cv3[l][i][1](torch.cat(
                        (out0, in_2nd_conv[f"cv3_level{l}_in_2nd_conv"][i]),
                        1))
                    out_dict[f"cv3_level{l}_out"].append(
                        self.cv3[l][i][2](out1))
            x[i] = torch.cat((self.cv2[i](
                x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)

        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level
        return y if self.export else (x, out_dict, y_hier)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s)**2)


class Detect4(nn.Module): # Hierarchical Arch Version 4
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.hier = hier
        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            coef_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                coef_list.append(c3_h)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    mod_list += [
                        nn.Sequential(
                            Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                            nn.Conv2d(c3_h, h_len, 1),
                            nn.Conv2d(h_len + len(self.hier[str(k - 1)]),
                                      h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3, len(self.hier[str(len(self.hier) - 1)]), 1),
                    nn.Conv2d(
                        len(self.hier[str(len(self.hier) - 1)]) +
                        len(self.hier[str(len(self.hier) - 2)]),
                        len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        out_3nd_conv = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
            out_3nd_conv[f"cv3_level{l}_out_3nd_conv"] = []

        for i in range(self.nl):
            for l in range(len(self.hier)):
                out0 = self.cv3[l][i][0](x[i])
                out1 = self.cv3[l][i][1](out0)
                out2 = self.cv3[l][i][2](out1)

                if l == 0:
                    out_dict[f"cv3_level{l}_out"].append(out2)
                    out_3nd_conv[f"cv3_level{l}_out_3nd_conv"].append(out2)
                else:
                    out_3nd_conv[f"cv3_level{l}_out_3nd_conv"].append(out2)
                    out3 = self.cv3[l][i][3](torch.cat(
                        (out2,
                         out_3nd_conv[f"cv3_level{l-1}_out_3nd_conv"][i]), 1))
                    out_dict[f"cv3_level{l}_out"].append(out3)

            x[i] = torch.cat((self.cv2[i](
                x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level
        return y if self.export else (x, out_dict, y_hier)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s)**2)


class Detect5(nn.Module): # Hierarchical Arch Version 5
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.hier = hier
        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            coef_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                coef_list.append(c3_h)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h * 2, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))
        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3 + c3_h, len(self.hier[str(len(self.hier) -
                                                           1)]), 1))
                for x, c3_h in zip(ch, coef_list)))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        in_3nd_conv = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
            in_3nd_conv[f"cv3_level{l}_in_3nd_conv"] = []

        for i in range(self.nl):
            for l in range(len(self.hier)):
                out0 = self.cv3[l][i][0](x[i])
                out1 = self.cv3[l][i][1](out0)
                in_3nd_conv[f"cv3_level{l}_in_3nd_conv"].append(out1)
                if l == 0:
                    out_dict[f"cv3_level{l}_out"].append(
                        self.cv3[l][i][2](out1))
                else:
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i][2](
                        torch.cat(
                            (out1,
                             in_3nd_conv[f"cv3_level{l-1}_in_3nd_conv"][i]),
                            1)))

            x[i] = torch.cat((self.cv2[i](
                x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level
        return y if self.export else (x, out_dict, y_hier)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s)**2)


class Detect6(nn.Module): # Hierarchical Arch Version 6
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.hier = hier
        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        hier_len = [len(self.hier[str(k)]) for k in range(len(self.hier))]

        for k in range(len(self.hier) - 1):
            mod_list = []
            coef_list = []
            for x in ch:
                c3_h = max(x // 4, hier_len[k])
                coef_list.append(c3_h)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, hier_len[k], 1))
                    ]
                else:
                    mod_list += [
                        nn.Sequential(
                            Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                            nn.Conv2d(c3_h, hier_len[k], 1),
                            nn.Conv2d(sum(hier_len[:k + 1]), hier_len[k], 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                              nn.Conv2d(c3, hier_len[-1], 1),
                              nn.Conv2d(sum(hier_len), hier_len[-1], 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        out_last_conv = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
            out_last_conv[f"cv3_level{l}_out_last_conv"] = []

        for i in range(self.nl):
            for l in range(len(self.hier)):
                out0 = self.cv3[l][i][0](x[i])
                out1 = self.cv3[l][i][1](out0)
                out2 = self.cv3[l][i][2](out1)

                if l == 0:
                    out_last_conv[f"cv3_level{l}_out_last_conv"].append(out2)
                    out_dict[f"cv3_level{l}_out"].append(out2)

                elif l == 1:
                    out3 = self.cv3[l][i][3](torch.cat(
                        (out2,
                         out_last_conv[f"cv3_level{l-1}_out_last_conv"][i]),
                        1))
                    out_last_conv[f"cv3_level{l}_out_last_conv"].append(out3)
                    out_dict[f"cv3_level{l}_out"].append(out3)

                elif l == 2:
                    out3 = self.cv3[l][i][3](torch.cat(
                        (out2,
                         out_last_conv[f"cv3_level{l-1}_out_last_conv"][i],
                         out_last_conv[f"cv3_level{l-2}_out_last_conv"][i]),
                        1))
                    out_last_conv[f"cv3_level{l}_out_last_conv"].append(out3)
                    out_dict[f"cv3_level{l}_out"].append(out3)

                elif l == 3:
                    out3 = self.cv3[l][i][3](torch.cat(
                        (out2,
                         out_last_conv[f"cv3_level{l-1}_out_last_conv"][i],
                         out_last_conv[f"cv3_level{l-2}_out_last_conv"][i],
                         out_last_conv[f"cv3_level{l-3}_out_last_conv"][i]),
                        1))
                    out_last_conv[f"cv3_level{l}_out_last_conv"].append(out3)
                    out_dict[f"cv3_level{l}_out"].append(out3)

                elif l == 4:
                    out3 = self.cv3[l][i][3](torch.cat(
                        (out2,
                         out_last_conv[f"cv3_level{l-1}_out_last_conv"][i],
                         out_last_conv[f"cv3_level{l-2}_out_last_conv"][i],
                         out_last_conv[f"cv3_level{l-3}_out_last_conv"][i],
                         out_last_conv[f"cv3_level{l-4}_out_last_conv"][i]),
                        1))
                    out_dict[f"cv3_level{l}_out"].append(out3)

            x[i] = torch.cat((self.cv2[i](
                x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)

        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level
        return y if self.export else (x, out_dict, y_hier)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s)**2)


# To select the correct Detect architecture, match hier_arch_version to Detect class
# hier_arch_version could be located in default.yaml, or could be passed through CLI
# CLI should override default.yaml

# Get the value of hier_arch_version from default.yaml
hier_arch_version = DEFAULT_CFG.hier_arch_version


def extract_args_from_cli(cfg: Union[str, Path, Dict,
                                     SimpleNamespace] = DEFAULT_CFG_DICT,
                          overrides: Dict = None):
    """
    Extract command-line arguments passed to the script, or use debug arguments if specified.
    
    Args:
    cfg (Union[str, Path, Dict, SimpleNamespace], optional): Configuration data. Default is DEFAULT_CFG_DICT.
    overrides (Dict, optional): Additional override parameters. Default is None.
    Returns:
    List[str]: A list of command-line arguments.
    """
    debug = ''
    args = (debug.split(' ') if debug else sys.argv)[1:]
    return args


all_overrides = extract_args_from_cli()

# Extract hier_arch_version if available in CLI
hier_arch_version_cli = next(
    (item.split('=')[1]
     for item in all_overrides if item.startswith('hier_arch_version=')), None)

if len(sys.argv) > 1 and "detect" in sys.argv:
    if hier_arch_version_cli is not None:
        hier_arch_version = int(hier_arch_version_cli)


# Get the architecture corresponding to hier_arch_version
def get_detect_class(hier_arch_version):
    if hier_arch_version == 1:
        return Detect1
    elif hier_arch_version == 2:
        return Detect2
    elif hier_arch_version == 3:
        return Detect3
    elif hier_arch_version == 4:
        return Detect4
    elif hier_arch_version == 5:
        return Detect5
    elif hier_arch_version == 6:
        return Detect6
    else:
        raise ValueError(f"Unsupported hier_arch_version: {hier_arch_version}")

Detect = get_detect_class(hier_arch_version)


class Segment(Detect):
    # YOLOv8 Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3),
                          nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1),
                p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc,
                                                                   p))


class Classify(nn.Module):
    # YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)
