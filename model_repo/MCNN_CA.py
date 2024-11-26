import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(1, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 94, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AvgPool1d(2)
        self.max_pool = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(64, 64, 1,stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        x_weight = torch.cat([avg_out, max_out], dim=2)
        x_weight = self.conv1(x_weight)
        x_weight = self.sigmoid(x_weight)
        x = x*x_weight

        return x

class ResidualBlockWithSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlockWithSpatialAttention, self).__init__()
        self.name = os.path.basename(__file__).split('.')[0]

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1)
        self.spatial_attention = SpatialAttention()
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        out_spatial_attention =  self.spatial_attention(residual)
        out = out_spatial_attention + out
        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5))
        out = torch.bmm(attention, V)
        return out

class MultiScaleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvLayer, self).__init__()
        self.pool = nn.MaxPool1d(2,stride=4)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=16, padding=6,stride=4)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=16, padding=6, stride=4)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=16, padding=6,stride=4)
        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=16, padding=6,stride=4)

        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        out1 = self.pool(out1)
        out2 = self.pool(out2)
        out3 = self.pool(out3)
        out4 = self.pool(out4)

        out1 = out1 * self.channel_attention(out1)
        out2 = out2 * self.channel_attention(out2)
        out3 = out3 * self.channel_attention(out3)
        out4 = out4 * self.channel_attention(out4)

        out = torch.cat((out1, out2, out3, out4), dim=1)

        return out

class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.name = os.path.basename(__file__).split('.')[0]
        num_classes = 4
        self.num_classes = num_classes
        self.multi_scale_conv = MultiScaleConvLayer(3, 16)
        self.residual_block = ResidualBlockWithSpatialAttention(64, 64)
        self.self_attention = SelfAttentionLayer(47)

        # need to be changed: self.attention:47, fc:3008, ChannelAttention:94
        self.fc = nn.Sequential(
            nn.Linear(3008, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )



        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)

    def forward(self, x):
        x = x[:,:3,:]

        x = self.multi_scale_conv(x)

        x = self.residual_block(x)
        x = self.pool(x)

        x = self.self_attention(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x
