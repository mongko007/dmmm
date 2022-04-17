import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_num):
        super(ResBlock, self).__init__()
        self.res_num = res_num
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0)

    def forward(self, x):
        for i in range(self.res_num):
            x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
            x1 = self.conv(x)
            x1 = F.relu(x1)
            x1 = self.conv(x1)
            x = x1 + x
        return x


class Encoder(nn.Module):
    def __init__(self, enc_in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=enc_in_channels,
                out_channels=16,
                kernel_size=7,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            ResBlock(32, 32, 3),
        )
        self.conv_text_repr = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            ResBlock(32, 32, 2)
        )
        self.conv_shape_repr = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            ResBlock(32, 32, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        text_repr = self.conv_text_repr(x)
        shape_repr = self.conv_shape_repr(x)
        return text_repr, shape_repr


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            ResBlock(64, 64, 9),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=7,
                stride=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def Manipulator(Ma, Mb, alpha):
    x = Mb - Ma
    g_ = nn.Conv2d(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1
    )
    x = g_(x)
    x = F.relu(x)
    x = x * alpha
    f_ = nn.Conv2d(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1
    )
    x = f_(x)
    res = ResBlock(32, 32, 1)
    x = res(x)
    x = Ma + x
    return x






