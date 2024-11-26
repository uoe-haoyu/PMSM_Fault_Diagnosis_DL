
from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import os


class moving_avg(nn.Module):
    """
    Moving average block
    """
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, padding=0)

    def forward(self, x):
        x = self.avg(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, seq_len, kernel_size=3, stride=1, groups=1,
                 padding=None, use_norm=True, use_act=True):
        super().__init__()

        kernels_downsampling = [2, 4, 8]
        self.seq_len = seq_len
        self.cal_scale = [moving_avg(k) for k in kernels_downsampling]
        self.t_scale = seq_len
        for k in kernels_downsampling:
            self.t_scale+=self.seq_len//k

        block = []
        padding = padding or kernel_size // 2
        block.append(nn.Conv1d(
            in_channel, out_channel, kernel_size, stride, padding=padding, groups=groups, bias=False
        ))
        if use_norm:
            block.append(nn.BatchNorm1d(out_channel))
        if use_act:
            block.append(nn.GELU())

        self.block = nn.Sequential(*block)



    def forward(self, x):

        # Multi-scale downsampling
        x_scale = []
        for cal in self.cal_scale:
            x_scale.append(cal(x))
        x_scale = torch.cat(x_scale,dim=2)


        return self.block(x_scale)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        return x.transpose(-1, -2)

class Add(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        return weight[0] * x[0] + weight[1] * x[1]


class Embedding(nn.Module):
    def __init__(self, d_in, d_out, stride=2, n=4):
        super(Embedding, self).__init__()
        d_hidden = d_out // n
        self.conv1 = nn.Conv1d(d_in, d_hidden, 1, 1)

        # adaptive weights with initial value 1
        self.weights = nn.Parameter(torch.ones(n), requires_grad=True)

        self.sconv = nn.ModuleList([
            nn.Conv1d(d_hidden, d_hidden, 2 * i + 2 * stride - 1,
                      stride=stride, padding=stride + i - 1, groups=d_hidden, bias=False)
            for i in range(n)
        ])
        self.act_bn = nn.Sequential(
            nn.BatchNorm1d(d_out), nn.GELU())

    def forward(self, x):
        signals = []
        x = self.conv1(x)

        for sconv in self.sconv:
            signals.append(sconv(x))

        # Softmax weights
        norm_weights = torch.softmax(self.weights, dim=0)
        weighted_signals = [w * signal for w, signal in zip(norm_weights, signals)]
        x = torch.cat(weighted_signals, dim=1)

        return self.act_bn(x)


class BroadcastAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 proj_drop=0.,
                 attn_drop=0.,
                 qkv_bias=True
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads."

        # 多头的查询、键和值通过卷积进行线性变换
        self.qkv_proj = nn.Conv1d(dim, num_heads * (1 + 2 * self.head_dim), kernel_size=1, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, N = x.shape
        qkv = self.qkv_proj(x).view(B, self.num_heads, 1 + 2 * self.head_dim, N)

        # get q,k,v
        query, key, value = torch.split(qkv, [1, self.head_dim, self.head_dim], dim=2)

        # Cal attention score of each head
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Cal context_vector
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(value) * context_vector.expand_as(value)

        # Concatenate heads
        out = out.permute(0, 1, 3, 2)
        out = out.contiguous().view(B, self.num_heads, N, self.head_dim)
        out = torch.cat([out[:, i, :, :] for i in range(self.num_heads)], dim=-1)

        out = out.permute(0, 2, 1)

        # output of MCBA
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BA_FFN_Block(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads=1,
                 drop=0.,
                 attn_drop=0.
                 ):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.add1 = Add()
        self.attn = BroadcastAttention(dim=dim,
                                       num_heads=num_heads,
                                       attn_drop=attn_drop,
                                       proj_drop=drop)

        self.norm2 = LayerNorm(dim)
        self.add2 = Add()
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, 1, 1, bias=True),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Conv1d(ffn_dim, dim, 1, 1, bias=True),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.add1([self.attn(self.norm1(x)), x])
        x = self.add2([self.ffn(self.norm2(x)), x])
        return x


class LFEL(nn.Module):
    def __init__(self, d_in, d_out, drop, num_heads=1):
        super(LFEL, self).__init__()

        self.embed = Embedding(d_in, d_out, stride=2, n=4)
        self.block = BA_FFN_Block(dim=d_out,
                                  ffn_dim=d_out // 4,
                                  num_heads=num_heads,
                                  drop=drop,
                                  attn_drop=drop)

    def forward(self, x):
        x = self.embed(x)
        return self.block(x)


class Net(nn.Module):
    def __init__(self, _, in_channel=3, out_channel=4, drop=0.1, dim=8, num_heads=4, seq_len =1500):
        super(Net, self).__init__()
        self.name = os.path.basename(__file__).split('.')[0]


        self.in_layer = nn.Sequential(
            ConvBNReLU(in_channel = in_channel, out_channel=dim, seq_len=seq_len,  kernel_size=15, stride=2)
        )

        self.in_frequency_layer = nn.Sequential(
            ConvBNReLU(in_channel = 2, out_channel=dim, seq_len=seq_len,  kernel_size=15, stride=2)
        )

        self.LFELs = nn.Sequential(
            LFEL(dim, 4 * dim, drop, num_heads),
            nn.AdaptiveAvgPool1d(1)
        )

        self.out_layer = nn.Linear(4 * dim, out_channel)


        self.fc_input_freqency = nn.Sequential(
            nn.Linear(18, 1500),
        )


        self.fc_output = nn.Linear(32, 4)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.weight1 = nn.Parameter(torch.ones(1, 32, 1) * 0.25)  # inital_0.25
        self.weight2 = nn.Parameter(torch.ones(1, 32, 1) * 0.75)  # inital_0.75

    def forward(self, x):

        input_time = x[:, :3, :]
        input_freqency = x[:,3:,:18]

        # time_domain
        input_time = self.in_layer(input_time)
        input_time = self.LFELs(input_time)


        # frequency_domain
        input_freqency = self.conv1(input_freqency)

        # fusion
        output = self.weight1 * input_time + self.weight2 * input_freqency
        output=output.view(output.size(0),-1)
        output = self.fc_output(output)

        return output





