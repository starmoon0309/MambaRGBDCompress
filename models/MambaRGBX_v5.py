import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride, conv1x1
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import os
from compressai.ans import BufferedRansEncoder, RansDecoder
import copy
from skimage import morphology
import numpy as np

from typing import Any, Callable, List, Optional, Tuple, Union
from torch import Tensor

import torch.nn.init as init


try:
    from .ckbd import *
except:
    from ckbd import *


import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import pdb
from datetime import datetime
import random
import time

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
from functools import partial
from typing import Optional, Callable, Any
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint


from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath, to_2tuple
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
import numpy as np
import math



### add new

import time
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

###################################  SelectiveScan  ################################### 


class DynamicLayerNorm(nn.Module):
    def __init__(self):
        super(DynamicLayerNorm, self).__init__()

    def forward(self, x):
        normalized_shape = x.size()[1:]
        layer_norm = nn.LayerNorm(normalized_shape).to(x.device)
        return layer_norm(x)


class CheckMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == "A":
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        else:
            self.mask[:, :, 0::2, 0::2] = 1
            self.mask[:, :, 1::2, 1::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out

class Hyperprior(CompressionModel):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int=192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        self.hyper_encoder = nn.Sequential(
            conv(in_planes, mid_planes, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
        )
        if out_planes == 2 * in_planes:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, in_planes * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(in_planes * 3 // 2, out_planes, stride=1, kernel_size=3),
            )
        else:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(mid_planes, out_planes, stride=1, kernel_size=3),
            )

    def forward(self, y, out_z=False):
        z = self.hyper_encoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        params = self.hyper_decoder(z_hat)
        if out_z:
            return params, z_likelihoods, z_hat
        else:
            return params, z_likelihoods

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.hyper_decoder(z_hat)
        return params, z_hat, z_strings

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        params = self.hyper_decoder(z_hat)
        return params, z_hat



###################################  SelectiveScan  ################################### 

try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)
try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)
try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
    
class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs
    
def cross_selective_scan(
    x1: torch.Tensor=None, 
    x2: torch.Tensor=None, 
    x1_proj_weight: torch.Tensor=None,
    x2_proj_weight: torch.Tensor=None,
    x1_proj_bias: torch.Tensor=None,
    x2_proj_bias: torch.Tensor=None,
    dt1_projs_weight: torch.Tensor=None,
    dt2_projs_weight: torch.Tensor=None,
    dt1_projs_bias: torch.Tensor=None,
    dt2_projs_bias: torch.Tensor=None,
    A1_logs: torch.Tensor=None,
    A2_logs: torch.Tensor=None,
    D1s: torch.Tensor=None,
    D2s: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan,
    CrossMerge=CrossMerge,
    no_einsum=False, # replace einsum with linear or conv1d to raise throughput
    dt_low_rank=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x1.shape
    D, N = A1_logs.shape
    K, D, R = dt1_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    if (not dt_low_rank):
        x1_dbl = F.conv1d(x1.view(B, -1, L), x1_proj_weight.view(-1, D, 1), bias=(x1_proj_bias.view(-1) if x1_proj_bias is not None else None), groups=K)
        dt1s, B1s, C1s = torch.split(x1_dbl.view(B, -1, L), [D, 4 * N, 4 * N], dim=1)
        x1s = CrossScan.apply(x1)
        dt1s = CrossScan.apply(dt1s)
        x2_dbl = F.conv1d(x2.view(B, -1, L), x2_proj_weight.view(-1, D, 1), bias=(x2_proj_bias.view(-1) if x2_proj_bias is not None else None), groups=K)
        dt2s, B2s, C2s = torch.split(x2_dbl.view(B, -1, L), [D, 4 * N, 4 * N], dim=1)
        x2s = CrossScan.apply(x2)
        dt2s = CrossScan.apply(dt2s)
        
    elif no_einsum:
        x1s = CrossScan.apply(x1)
        x1_dbl = F.conv1d(x1s.view(B, -1, L), x1_proj_weight.view(-1, D, 1), bias=(x1_proj_bias.view(-1) if x1_proj_bias is not None else None), groups=K)
        dt1s, B1s, C1s = torch.split(x1_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dt1s = F.conv1d(dt1s.contiguous().view(B, -1, L), dt1_projs_weight.view(K * D, -1, 1), groups=K)
        x2s = CrossScan.apply(x2)
        x2_dbl = F.conv1d(x2s.view(B, -1, L), x2_proj_weight.view(-1, D, 1), bias=(x2_proj_bias.view(-1) if x2_proj_bias is not None else None), groups=K)
        dt2s, B2s, C2s = torch.split(x2_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dt2s = F.conv2d(dt2s.contiguous().view(B, -1, L), dt2_projs_weight.view(K * D, -1, 1), groups=K)
        
    else:
        x1s = CrossScan.apply(x1)
        x1_dbl = torch.einsum("b k d l, k c d -> b k c l", x1s, x1_proj_weight)
        if x1_proj_bias is not None:
            x1_dbl = x1_dbl + x1_proj_bias.view(1, K, -1, 1)
        dt1s, B1s, C1s = torch.split(x1_dbl, [R, N, N], dim=2)
        dt1s = torch.einsum("b k r l, k d r -> b k d l", dt1s, dt1_projs_weight)
        
        
        x2s = CrossScan.apply(x2)
        x2_dbl = torch.einsum("b k d l, k c d -> b k c l", x2s, x2_proj_weight)
        if x2_proj_bias is not None:
            x2_dbl = x2_dbl + x2_proj_bias.view(1, K, -1, 1)
        dt2s, B2s, C2s = torch.split(x2_dbl, [R, N, N], dim=2)
        dt2s = torch.einsum("b k r l, k d r -> b k d l", dt2s, dt2_projs_weight)
        

    x1s = x1s.view(B, -1, L)
    dt1s = dt1s.contiguous().view(B, -1, L)
    A1s = -torch.exp(A1_logs.to(torch.float)) 
    B1s = B1s.contiguous().view(B, K, N, L)
    C1s = C1s.contiguous().view(B, K, N, L)
    D1s = D1s.to(torch.float) 
    delta1_bias = dt1_projs_bias.view(-1).to(torch.float)
    x2s = x2s.view(B, -1, L)
    dt2s = dt2s.contiguous().view(B, -1, L)
    A2s = -torch.exp(A2_logs.to(torch.float)) 
    B2s = B2s.contiguous().view(B, K, N, L)
    C2s = C2s.contiguous().view(B, K, N, L)
    D2s = D2s.to(torch.float) 
    delta2_bias = dt2_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        x1s = x1s.to(torch.float)
        dt1s = dt1s.to(torch.float)
        B1s = B1s.to(torch.float)
        C1s = C1s.to(torch.float)
        x2s = x2s.to(torch.float)
        dt2s = dt2s.to(torch.float)
        B2s = B2s.to(torch.float)
        C2s = C2s.to(torch.float)

    
    y1s: torch.Tensor = selective_scan(
        x1s, dt1s, A1s, B1s, C2s, D1s, delta1_bias, delta_softplus
    ).view(B, K, -1, H, W)
    
    y2s: torch.Tensor = selective_scan(
        x2s, dt2s, A2s, B2s, C1s, D2s, delta2_bias, delta_softplus
    ).view(B, K, -1, H, W)
    
    
    
    
    y1: torch.Tensor = CrossMerge.apply(y1s)
    y2: torch.Tensor = CrossMerge.apply(y2s)
    

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y1 = out_norm(y1.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
        y2 = out_norm(y2.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y1 = y1.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y1 = out_norm(y1).view(B, H, W, -1)
        y2 = y2.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y2 = out_norm(y2).view(B, H, W, -1)
    return (y1.to(x1.dtype) if to_dtype else y1), (y2.to(x2.dtype) if to_dtype else y2)

class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)
    
class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)
    

class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x
    
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)

class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        # ======================
        **kwargs,
    ):
        if forward_type.startswith("v0"):
            self.__initv0__(d_model, d_state, ssm_ratio, dt_rank, dropout, seq=("seq" in forward_type))
            return
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        
        self.dynamic_layer_norm = DynamicLayerNorm() # new add

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        if forward_type.startswith("debug"):
            from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations, cross_selective_scanv2
            FORWARD_TYPES.update(dict(
                debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
                debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
                debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
                debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
                debugforward_core_mambassm_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
                debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
                debugforward_core_sscore_fusecscm_fwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fwdnrow, self),
                debugforward_core_sscore_fusecscm_bwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_bwdnrow, self),
                debugforward_core_sscore_fusecscm_fbnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fbnrow, self),
                debugforward_core_ssoflex_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
                debugforward_core_ssoflex_fusecscm_i16o32=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
                debugscan_sharessm=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scanv2),
            ))
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj1 = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_fused1 = nn.Linear(d_model, d_inner, bias=bias, **factory_kwargs)
        self.in_proj_fused2 = nn.Linear(d_model, d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x1_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x1_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x1_proj], dim=0)) # (K, N, inner)
        del self.x1_proj
        
        
        self.x2_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x2_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x2_proj], dim=0)) # (K, N, inner)
        del self.x2_proj
        
        # out proj =======================================
        self.out_proj1 = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.out_proj2 = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:

                       
            # dt proj ============================
            self.dt1_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
        
            self.dt2_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
        
            self.dt1_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt1_projs], dim=0)) # (K, inner, rank)
            self.dt1_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt1_projs], dim=0)) # (K, inner)
        
            self.dt2_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt2_projs], dim=0)) # (K, inner, rank)
            self.dt2_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt2_projs], dim=0)) # (K, inner)
        
            del self.dt1_projs
            del self.dt2_projs
            
            # A, D =======================================
            self.A1_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.D1s = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     
        
            self.A2_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.D2s = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)    
            
        
 
            
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
    
        if forward_type.startswith("xv"):
            self.d_state = d_state
            self.dt_rank = dt_rank
            self.d_inner = d_inner

            if d_conv > 1:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            self.act: nn.Module = act_layer()
            self.out_act: nn.Module = nn.Identity()
            del self.x_proj_weight

            if forward_type.startswith("xv1"):
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = self.forwardxv

            if forward_type.startswith("xv2"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight

            if forward_type.startswith("xv3"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv4"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            if forward_type.startswith("xv5"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight
                self.out_act = nn.GELU()

    # only used to run previous version
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj1 = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x1_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x1_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x1_proj], dim=0)) # (K, N, inner)
        del self.x1_proj
        
        
        self.x2_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x2_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x2_proj], dim=0)) # (K, N, inner)
        del self.x2_proj

        # dt proj ============================
        self.dt1_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        
        self.dt2_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        
        self.dt1_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt1_projs], dim=0)) # (K, inner, rank)
        self.dt1_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt1_projs], dim=0)) # (K, inner)
        
        self.dt2_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt2_projs], dim=0)) # (K, inner, rank)
        self.dt2_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt2_projs], dim=0)) # (K, inner)
        
        del self.dt1_projs
        del self.dt2_projs
            
        # A, D =======================================
        self.A1_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.D1s = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     
        
        self.A2_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.D2s = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj1 = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.out_proj2 = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x1: torch.Tensor, x2: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x1_proj_weight = self.x1_proj_weight
        dt1_projs_weight = self.dt1_projs_weight
        dt1_projs_bias = self.dt1_projs_bias
        A1_logs = self.A1_logs
        D1s = self.D1s
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        x2_proj_weight = self.x2_proj_weight
        dt2_projs_weight = self.dt2_projs_weight
        dt2_projs_bias = self.dt2_projs_bias
        A2_logs = self.A2_logs
        D2s = self.D2s
        
        res1, res2 = cross_selective_scan(
            x1, x2, x1_proj_weight, x2_proj_weight, None, None, dt1_projs_weight, dt2_projs_weight, dt1_projs_bias, dt2_projs_bias,
            A1_logs, A2_logs, D1s, D2s, delta_softplus=True,
            out_norm=out_norm,
            out_norm_shape=out_norm_shape,
            **kwargs,
        )

        return res1, res2
    
    def forwardv0(self, x: torch.Tensor, SelectiveScan = SelectiveScanMamba, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        
        x1_fused = self.in_proj_fused1(x1) # new add
        x2_fused = self.in_proj_fused2(x2) # new add
        
        x1 = self.in_proj1(x1) # new add
        x2 = self.in_proj2(x2) # new add
        
        
        
        if not self.disable_z:
            x1, z1 = x1.chunk(2, dim=-1) # (b, h, w, d)
            if not self.disable_z_act:
                z1 = self.act(z1)
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x1 = self.conv2d(x1) # (b, d, h, w)
        x1 = self.act(x1)
        
        
        if not self.disable_z:
            x2, z2 = x2.chunk(2, dim=-1) # (b, h, w, d)
            if not self.disable_z_act:
                z2 = self.act(z2)
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x2 = self.conv2d(x2) # (b, d, h, w)
        x2 = self.act(x2)
        
        
        y1, y2 = self.forward_core(x1, x2)
        
        
        

        if not self.disable_z:
            y1 = y1 * z1
            
        if not self.disable_z:
            y2 = y2 * z2
        
            
        y1 = self.dynamic_layer_norm(y1)
        y2 = self.dynamic_layer_norm(y2)   
        y1_tem = x1_fused * y1 + x1_fused * y2
        y2_tem = x2_fused * y1 + x2_fused * y2  
        y1 = y1_tem
        y2 = y2_tem
        out1 = self.dropout(self.out_proj1(y1))
        out2 = self.dropout(self.out_proj2(y2))
        return out1, out2

    def forwardxv(self, x: torch.Tensor, mode="xv1", **kwargs):
        B, H, W, C = x.shape
        L = H * W
        K = 4
        dt_projs_weight = getattr(self, "dt_projs_weight", None)
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        x = x.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x = self.conv2d(x) # (b, d, h, w)
            x = self.act(x)
        x = self.in_proj(x)

        if mode in ["xv1"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            dts = CrossScanTriton.apply(dts)
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        elif mode in ["xv2"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            dts = CrossScanTriton.apply(dts).contiguous().view(B, -1, L)
        elif mode in ["xv3"]:
            us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            dts = CrossScanTriton1b1.apply(dts.contiguous().view(B, K, -1, H, W))
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)

        us = CrossScanTriton.apply(us.contiguous()).view(B, -1, L)
        Bs, Cs = Bs.view(B, K, -1, L).contiguous(), Cs.view(B, K, -1, L).contiguous()
    
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
            
        y: torch.Tensor = CrossMergeTriton.apply(ys)

        if out_norm_shape in ["v1"]: # (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
        else: # (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
            y = out_norm(y).view(B, H, W, -1)

        y = (y.to(x.dtype) if to_dtype else y)
        out = self.dropout(self.out_proj(self.out_act(y)))
        return out
    
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        # print(drop_path)
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
        )
        
        self.drop_path = DropPath(drop_path)

    def _forward(self, input1: torch.Tensor, input2: torch.Tensor):
        if self.post_norm:
            out1, out2  = self.op(input1,input2)
            x1 = input1 + self.drop_path(self.norm(out1))
            x2 = input2 + self.drop_path(self.norm(out2))
        else:
            # pdb.set_trace()
            input1_tem = self.norm(input1)
            input2_tem = self.norm(input2)
            out1, out2  = self.op(input1_tem,input2_tem)
            x1 = input1 + self.drop_path(out1)
            x2 = input2 + self.drop_path(out2)
        return x1, x2

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input1, input2)
        else:
            x1 = input1.permute(0, 2, 3, 1)
            x2 = input2.permute(0, 2, 3, 1)
            out1,out2 = self._forward(x1,x2)
            out1 = out1.permute(0, 3, 1, 2)
            out2 = out2.permute(0, 3, 1, 2)
            return out1,out2


class VSSBlockSequential(nn.Module):
    def __init__(self, *args):
        super(VSSBlockSequential, self).__init__()
        self.blocks = nn.ModuleList(args)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        for block in self.blocks:
            input1, input2 = block(input1, input2)
        return input1, input2
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

        
    
        


class Encoder(nn.Module):
    def __init__(self, N, M, depths=[1, 1, 1, 1], drop_path_rate=0.1,  **kwargs):
        super().__init__()
        self.g_a_conv1_l = conv(3, N)
        self.g_a_conv2_l = conv(N, N)
        self.g_a_conv3_l = conv(N, N)
        self.g_a_conv4_l = conv(N, M)
        self.g_a_conv1_r = conv(3, N)
        self.g_a_conv2_r = conv(N, N)
        self.g_a_conv3_r = conv(N, N)
        self.g_a_conv4_r = conv(N, M)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cmfb1 = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])])
        self.cmfb2 = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i + depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[1])])
        self.cmfb3 = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i + depths[0] + depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[2])])
        

    def forward(self, x, y):
        x_1 = self.g_a_conv1_l(x)
        y_1 = self.g_a_conv1_r(y)
        x_2_out, y_2_out = self.cmfb1(x_1,y_1)
        
        x_3 = self.g_a_conv2_l(x_2_out)
        y_3 = self.g_a_conv2_r(y_2_out)
        x_4_out, y_4_out = self.cmfb2(x_3,y_3)
        
        x_5 = self.g_a_conv3_l(x_4_out)
        y_5 = self.g_a_conv3_r(y_4_out)
        x_6_out, y_6_out = self.cmfb3(x_5,y_5)
        
        x_7 = self.g_a_conv4_l(x_6_out)
        y_7 = self.g_a_conv4_r(y_6_out)
        
        return x_7, y_7


class Decoder(nn.Module):
    def __init__(self, N, M, depths=[1, 1, 1, 1], drop_path_rate=0.1,  **kwargs):
        super().__init__()
        self.g_s_conv1_l = deconv(M, N)
        self.g_s_conv2_l = deconv(N, N)
        self.g_s_conv3_l = deconv(N, N)
        self.g_s_conv4_l = deconv(N, 3)
        
        self.g_s_conv1_r = deconv(M, N)
        self.g_s_conv2_r = deconv(N, N)
        self.g_s_conv3_r = deconv(N, N)
        self.g_s_conv4_r = deconv(N, 3)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cmfb1 = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i + depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[1])])
        self.cmfb2 = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i + depths[0] + depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[2])])
        self.cmfb3 = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[3])])
        

    def forward(self, x, y):
        x_1 = self.g_s_conv1_l(x)
        y_1 = self.g_s_conv1_r(y)
        x_2_out, y_2_out = self.cmfb1(x_1,y_1)
        
        x_3 = self.g_s_conv2_l(x_2_out)
        y_3 = self.g_s_conv2_r(y_2_out)
        x_4_out, y_4_out = self.cmfb2(x_3,y_3)
        
        x_5 = self.g_s_conv3_l(x_4_out)
        y_5 = self.g_s_conv3_r(y_4_out)
        x_6_out, y_6_out = self.cmfb3(x_5,y_5)
        
        x_7 = self.g_s_conv4_l(x_6_out)
        y_7 = self.g_s_conv4_r(y_6_out)
        
        return x_7, y_7


class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        channel_params = self.fushion(channel_params)
        return channel_params



class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        gaussian_params = self.fusion(params)
        return gaussian_params
        


class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, depths=[1, 1, 1, 1], drop_path_rate=0.1,  act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.conv_1_1 = conv3x3(M, N)
        self.conv_2_1 = conv3x3(M, N)
        self.relu = nn.ReLU()
        self.conv_1_2 = conv(N, N)
        self.conv_2_2 = conv(N, N)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cmfb = VSSBlockSequential(*[VSSBlock(hidden_dim = N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[3])])
        self.conv_1_3 = conv(N, N)
        self.conv_2_3 = conv(N, N)

    def forward(self, x1, x2):
        x1 = self.relu(self.conv_1_1(x1))
        x2 = self.relu(self.conv_2_1(x2))
        x1 = self.relu(self.conv_1_2(x1))
        x2 = self.relu(self.conv_2_2(x2))
        x1,x2 = self.cmfb(x1,x2)
        x1 = self.conv_1_3(x1)
        x2 = self.conv_2_3(x2)
        return x1, x2



class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, depths=[1, 1, 1, 1], drop_path_rate=0.1,  act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M),
            act(),
            deconv(M, M * 3 // 2),
            act(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )
        
        self.conv_1_1 = deconv(N, M)
        self.conv_2_1 = deconv(N, M)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cmfb = VSSBlockSequential(*[VSSBlock(hidden_dim = M, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])])
        self.relu = nn.ReLU()
        self.conv_1_2 = deconv(M, M * 3 // 2)
        self.conv_2_2 = deconv(M, M * 3 // 2)
        self.conv_1_3 = deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1)
        self.conv_2_3 = deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1)

    def forward(self, x1, x2):
        x1 = self.conv_1_1(x1)
        x2 = self.conv_2_1(x2)
        x1,x2 = self.cmfb(x1,x2)
        x1 = self.conv_1_2(self.relu(x1))
        x2 = self.conv_2_2(self.relu(x2))
        x1 = self.conv_1_3(self.relu(x1))
        x2 = self.conv_2_3(self.relu(x2))
        return x1, x2



class MambaRGBD(CompressionModel):
    def __init__(self, N = 192, M = 320):
        super().__init__()
        
        self.encoder = Encoder(N,M)
        self.decoder = Decoder(N,M)
        
        slice_num = 5
        slice_ch = [16, 16, 32, 64, 192]
        self.quant = "ste"
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = Encoder(N,M)
        self.g_s = Decoder(N,M)
        
        
        self.h_a = HyperAnalysisEX(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEX(N, M, act=nn.ReLU)
        
        # Channel Fusion Model
        self.local_context1 = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        
        self.local_context2 = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        
        self.local_context3 = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        
        self.channel_context1 = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor1 = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor1 = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Gussian Conditional
        self.gaussian_conditional1 = GaussianConditional(None)
        self.entropy_bottleneck1 = EntropyBottleneck(N)
        
        
        
        # Channel Fusion Model
        self.channel_context2 = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        
        
        self.channel_context3 = nn.ModuleList(
            ChannelContextEX(in_dim=slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        
        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor2 = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 6, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor2 = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 8, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 6, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Gussian Conditional
        self.gaussian_conditional2 = GaussianConditional(None)
        self.entropy_bottleneck2 = EntropyBottleneck(N)
        
        
        
    def forward(self, x1, x2):
        y1,y2 = self.g_a(x1,x2)
        z1,z2 = self.h_a(y1,y2)
        z1_hat, z1_likelihoods = self.entropy_bottleneck1(z1)
        if self.quant == 'ste':
            z1_offset = self.entropy_bottleneck1._get_medians()
            z1_hat = quantize_ste(z1 - z1_offset) + z1_offset
            
        z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)
        if self.quant == 'ste':
            z2_offset = self.entropy_bottleneck2._get_medians()
            z2_hat = quantize_ste(z2 - z2_offset) + z2_offset   
        
        # Hyper-parameters
        hyper_params1, hyper_params2 = self.h_s(z1_hat,z2_hat)
        y1_slices = [y1[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y1_hat_slices = []
        y1_likelihoods = []
        y2_slices = [y2[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y2_hat_slices = []
        y2_likelihoods = []
        
        for idx, (y1_slice, y2_slice) in enumerate(zip(y1_slices, y2_slices)):
            slice_anchor, slice_nonanchor = ckbd_split(y1_slice)
            if idx == 0:
                # depth 
                # Anchor
                params_anchor = self.entropy_parameters_anchor1[idx](hyper_params1)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == 'ste':
                    slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = self.gaussian_conditional1.quantize(
                        slice_anchor, "noise" if self.training else "dequantize"
                    )
                    slice_anchor = ckbd_anchor(slice_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context1[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor1[idx](torch.cat([local_ctx, hyper_params1], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y1_slice_likelihoods = self.gaussian_conditional1(y1_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == 'ste':
                    slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = self.gaussian_conditional1.quantize(
                        slice_nonanchor, "noise" if self.training else "dequantize"
                    )
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y1_hat_slice = slice_anchor + slice_nonanchor
                y1_hat_slices.append(y1_hat_slice)
                y1_likelihoods.append(y1_slice_likelihoods)
                
                
                
                
                # rgb 
                # Anchor
                local_ctx2 = self.local_context2[idx](slice_nonanchor)
                
                slice_anchor, slice_nonanchor = ckbd_split(y2_slice)
                
                params_anchor = self.entropy_parameters_anchor2[idx](torch.cat([local_ctx, local_ctx2, hyper_params2], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == 'ste':
                    slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = self.gaussian_conditional2.quantize(
                        slice_anchor, "noise" if self.training else "dequantize"
                    )
                    slice_anchor = ckbd_anchor(slice_anchor)
                
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx3 = self.local_context3[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor2[idx](torch.cat([local_ctx, local_ctx2, local_ctx3, hyper_params2], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y2_slice_likelihoods = self.gaussian_conditional2(y2_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == 'ste':
                    slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = self.gaussian_conditional2.quantize(
                        slice_nonanchor, "noise" if self.training else "dequantize"
                    )
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y2_hat_slice = slice_anchor + slice_nonanchor
                y2_hat_slices.append(y2_hat_slice)
                y2_likelihoods.append(y2_slice_likelihoods)
                
                
                

            else:
                # depth
                channel_ctx1 = self.channel_context1[idx](torch.cat(y1_hat_slices, dim=1))
                channel_ctx2 = self.channel_context2[idx](torch.cat(y2_hat_slices, dim=1))
                channel_ctx = self.channel_context3[idx](torch.cat([channel_ctx1,channel_ctx2], dim=1))
                
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor1[idx](torch.cat([channel_ctx, hyper_params1], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == 'ste':
                    slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = self.gaussian_conditional1.quantize(
                        slice_anchor, "noise" if self.training else "dequantize"
                    )
                    slice_anchor = ckbd_anchor(slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context1[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor1[idx](torch.cat([local_ctx, channel_ctx, hyper_params1], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y1_slice_likelihoods = self.gaussian_conditional1(y1_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == 'ste':
                    slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = self.gaussian_conditional1.quantize(
                        slice_nonanchor, "noise" if self.training else "dequantize"
                    )
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y1_hat_slice = slice_anchor + slice_nonanchor
                y1_hat_slices.append(y1_hat_slice)
                y1_likelihoods.append(y1_slice_likelihoods)
                
                
                # rgb
                local_ctx2 = self.local_context2[idx](slice_nonanchor)
                
                slice_anchor, slice_nonanchor = ckbd_split(y2_slice)
                
                params_anchor = self.entropy_parameters_anchor2[idx](torch.cat([local_ctx, local_ctx2, channel_ctx, hyper_params2], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                if self.quant == 'ste':
                    slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                else:
                    slice_anchor = self.gaussian_conditional2.quantize(
                        slice_anchor, "noise" if self.training else "dequantize"
                    )
                    slice_anchor = ckbd_anchor(slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx3 = self.local_context3[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor2[idx](torch.cat([local_ctx, local_ctx2, local_ctx3, channel_ctx, hyper_params2], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y2_slice_likelihoods = self.gaussian_conditional2(y2_slice, scales_slice, means_slice)
                # round slice_nonanchor
                if self.quant == 'ste':
                    slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                else:
                    slice_nonanchor = self.gaussian_conditional2.quantize(
                        slice_nonanchor, "noise" if self.training else "dequantize"
                    )
                    slice_nonanchor = ckbd_nonanchor(slice_nonanchor)
                y2_hat_slice = slice_anchor + slice_nonanchor
                y2_hat_slices.append(y2_hat_slice)
                y2_likelihoods.append(y2_slice_likelihoods)
                
                     
                
        y1_hat = torch.cat(y1_hat_slices, dim=1)
        y1_likelihoods = torch.cat(y1_likelihoods, dim=1)
        y2_hat = torch.cat(y2_hat_slices, dim=1)
        y2_likelihoods = torch.cat(y2_likelihoods, dim=1)
        x1_hat, x2_hat = self.g_s(y1_hat, y2_hat)
        
        return {
            "x1_hat": x1_hat.clamp(0, 1),
            "likelihoods1": {"y1_likelihoods": y1_likelihoods, "z1_likelihoods": z1_likelihoods},
            "x2_hat": x2_hat.clamp(0, 1),
            "likelihoods2": {"y2_likelihoods": y2_likelihoods, "z2_likelihoods": z2_likelihoods}
        }


    
    def compress(self, x1, x2):
        
        y1,y2 = self.g_a(x1,x2)
        z1,z2 = self.h_a(y1, y2)

        torch.backends.cudnn.deterministic = True
        z1_strings = self.entropy_bottleneck1.compress(z1)
        z1_hat = self.entropy_bottleneck1.decompress(z1_strings, z1.size()[-2:])
        z2_strings = self.entropy_bottleneck2.compress(z2)
        z2_hat = self.entropy_bottleneck2.decompress(z2_strings, z2.size()[-2:])
        
        
        hyper_params1, hyper_params2 = self.h_s(z1_hat, z2_hat)
        y1_slices = [y1[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y1_hat_slices = []

        cdf1 = self.gaussian_conditional1.quantized_cdf.tolist()
        cdf_lengths1 = self.gaussian_conditional1.cdf_length.reshape(-1).int().tolist()
        offsets1 = self.gaussian_conditional1.offset.reshape(-1).int().tolist()
        encoder1 = BufferedRansEncoder()
        symbols_list1 = []
        indexes_list1 = []
        y1_strings = []
        
        
        y2_slices = [y2[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y2_hat_slices = []
        cdf2 = self.gaussian_conditional2.quantized_cdf.tolist()
        cdf_lengths2 = self.gaussian_conditional2.cdf_length.reshape(-1).int().tolist()
        offsets2 = self.gaussian_conditional2.offset.reshape(-1).int().tolist()
        encoder2 = BufferedRansEncoder()
        symbols_list2 = []
        indexes_list2 = []
        y2_strings = []
        
        
        

        for idx, (y1_slice, y2_slice) in enumerate(zip(y1_slices, y2_slices)):
            slice_anchor, slice_nonanchor = ckbd_split(y1_slice)
            if idx == 0:
                # depth
                # Anchor
                params_anchor = self.entropy_parameters_anchor1[idx](hyper_params1)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional1, slice_anchor, scales_anchor, means_anchor, symbols_list1, indexes_list1)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context1[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor1[idx](torch.cat([local_ctx, hyper_params1], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional1, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list1, indexes_list1)
                y1_slice_hat = slice_anchor + slice_nonanchor
                y1_hat_slices.append(y1_slice_hat)

                
                # rgb
                # Anchor
                local_ctx2 = self.local_context2[idx](slice_nonanchor)
                
                slice_anchor, slice_nonanchor = ckbd_split(y2_slice)
                
                params_anchor = self.entropy_parameters_anchor2[idx](torch.cat([local_ctx, local_ctx2, hyper_params2], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional2, slice_anchor, scales_anchor, means_anchor, symbols_list2, indexes_list2)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx3 = self.local_context3[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor2[idx](torch.cat([local_ctx, local_ctx2, local_ctx3, hyper_params2], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional2, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list2, indexes_list2)
                y2_slice_hat = slice_anchor + slice_nonanchor
                y2_hat_slices.append(y2_slice_hat)
            
            
            
            
            else:
                # depth
                # Anchor
                channel_ctx1 = self.channel_context1[idx](torch.cat(y1_hat_slices, dim=1))
                channel_ctx2 = self.channel_context2[idx](torch.cat(y2_hat_slices, dim=1))
                channel_ctx = self.channel_context3[idx](torch.cat([channel_ctx1,channel_ctx2], dim=1))
                params_anchor = self.entropy_parameters_anchor1[idx](torch.cat([channel_ctx, hyper_params1], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional1, slice_anchor, scales_anchor, means_anchor, symbols_list1, indexes_list1)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context1[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor1[idx](torch.cat([local_ctx, channel_ctx, hyper_params1], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional1, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list1, indexes_list1)
                y1_hat_slices.append(slice_nonanchor + slice_anchor)
                
                
                # rgb
                # Anchor
                local_ctx2 = self.local_context2[idx](slice_nonanchor)
                
                slice_anchor, slice_nonanchor = ckbd_split(y2_slice)
                
                params_anchor = self.entropy_parameters_anchor2[idx](torch.cat([local_ctx, local_ctx2, channel_ctx, hyper_params2], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional2, slice_anchor, scales_anchor, means_anchor, symbols_list2, indexes_list2)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx3 = self.local_context3[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor2[idx](torch.cat([local_ctx, local_ctx2, local_ctx3, channel_ctx, hyper_params2], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional2, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list2, indexes_list2)
                y2_hat_slices.append(slice_nonanchor + slice_anchor)

        
        encoder1.encode_with_indexes(symbols_list1, indexes_list1, cdf1, cdf_lengths1, offsets1)
        y1_string = encoder1.flush()
        y1_strings.append(y1_string)
        
        encoder2.encode_with_indexes(symbols_list2, indexes_list2, cdf2, cdf_lengths2, offsets2)
        y2_string = encoder2.flush()
        y2_strings.append(y2_string)
        

        torch.backends.cudnn.deterministic = False
        return {
            "strings": [y1_strings, z1_strings, y2_strings, z2_strings],
            "shape": z1.size()[-2:],
        }

    def decompress(self, strings, shape):
        
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()

        y1_strings = strings[0][0]
        z1_strings = strings[1]
        y2_strings = strings[2][0]
        z2_strings = strings[3]
        
        z1_hat = self.entropy_bottleneck1.decompress(z1_strings, shape)
        z2_hat = self.entropy_bottleneck2.decompress(z2_strings, shape)
        hyper_params1, hyper_params2 = self.h_s(z1_hat, z2_hat)
        
        y1_hat_slices = []
        cdf1 = self.gaussian_conditional1.quantized_cdf.tolist()
        cdf_lengths1 = self.gaussian_conditional1.cdf_length.reshape(-1).int().tolist()
        offsets1 = self.gaussian_conditional1.offset.reshape(-1).int().tolist()
        decoder1 = RansDecoder()
        decoder1.set_stream(y1_strings)
        
        y2_hat_slices = []
        cdf2 = self.gaussian_conditional2.quantized_cdf.tolist()
        cdf_lengths2 = self.gaussian_conditional2.cdf_length.reshape(-1).int().tolist()
        offsets2 = self.gaussian_conditional2.offset.reshape(-1).int().tolist()
        decoder2 = RansDecoder()
        decoder2.set_stream(y2_strings)
        
        
        

        for idx in range(self.slice_num):
            if idx == 0:
                # depth
                # Anchor
                params_anchor = self.entropy_parameters_anchor1[idx](hyper_params1)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional1, scales_anchor, means_anchor, decoder1, cdf1, cdf_lengths1, offsets1)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context1[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor1[idx](torch.cat([local_ctx, hyper_params1], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional1, scales_nonanchor, means_nonanchor, decoder1, cdf1, cdf_lengths1, offsets1)
                y1_hat_slice = slice_nonanchor + slice_anchor
                y1_hat_slices.append(y1_hat_slice)
                
                
                # rgb
                # Anchor
                local_ctx2 = self.local_context2[idx](slice_nonanchor)
                params_anchor = self.entropy_parameters_anchor2[idx](torch.cat([local_ctx, local_ctx2, hyper_params2], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional2, scales_anchor, means_anchor, decoder2, cdf2, cdf_lengths2, offsets2)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx3 = self.local_context3[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor2[idx](torch.cat([local_ctx, local_ctx2, local_ctx3, hyper_params2], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional2, scales_nonanchor, means_nonanchor, decoder2, cdf2, cdf_lengths2, offsets2)
                y2_hat_slice = slice_nonanchor + slice_anchor
                y2_hat_slices.append(y2_hat_slice)
                
                

            else:
                # depth
                # Anchor
                channel_ctx1 = self.channel_context1[idx](torch.cat(y1_hat_slices, dim=1))
                channel_ctx2 = self.channel_context2[idx](torch.cat(y2_hat_slices, dim=1))
                channel_ctx = self.channel_context3[idx](torch.cat([channel_ctx1,channel_ctx2], dim=1))
                params_anchor = self.entropy_parameters_anchor1[idx](torch.cat([channel_ctx, hyper_params1], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional1, scales_anchor, means_anchor, decoder1, cdf1, cdf_lengths1, offsets1)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context1[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor1[idx](torch.cat([local_ctx, channel_ctx, hyper_params1], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional1, scales_nonanchor, means_nonanchor, decoder1, cdf1, cdf_lengths1, offsets1)
                y1_hat_slice = slice_nonanchor + slice_anchor
                y1_hat_slices.append(y1_hat_slice)
                
                # rgb
                # Anchor
                local_ctx2 = self.local_context2[idx](slice_nonanchor)
                params_anchor = self.entropy_parameters_anchor2[idx](torch.cat([local_ctx, local_ctx2, channel_ctx, hyper_params2], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional2, scales_anchor, means_anchor, decoder2, cdf2, cdf_lengths2, offsets2)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx3 = self.local_context3[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor2[idx](torch.cat([local_ctx, local_ctx2, local_ctx3, channel_ctx, hyper_params2], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional2, scales_nonanchor, means_nonanchor, decoder2, cdf2, cdf_lengths2, offsets2)
                y2_hat_slice = slice_nonanchor + slice_anchor
                y2_hat_slices.append(y2_hat_slice)
                
        
        
        y1_hat = torch.cat(y1_hat_slices, dim=1)
        y2_hat = torch.cat(y2_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False
        x1_hat, x2_hat = self.g_s(y1_hat, y2_hat)

        torch.cuda.synchronize()
        
        return {
            "x_hat": [x1_hat.clamp(0, 1), x2_hat.clamp(0, 1)],
        }