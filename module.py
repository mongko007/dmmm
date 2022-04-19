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
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0)
        )

    def forward(self, x):
        for i in range(self.res_num):
            x1 = self.conv(x)
            x = x + x1
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=7,
                stride=1),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            ResBlock(32, 32, 3),
        )
        self.conv_text_repr = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2),
            nn.ReLU(),
            ResBlock(32, 32, 2)
        )
        self.conv_shape_repr = nn.Sequential(
            nn.ReflectionPad2d(1),
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


class Manipulator(nn.Module):
    def __init__(self):
        super(Manipulator, self).__init__()
        self.g_ = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU()
        )
        self.f_ = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1
            )
        )
        self.res = ResBlock(32, 32, 1)

    def forward(self, m_a, m_b, amp_factor):
        diff = m_b - m_a
        diff = self.g_(diff)
        diff = diff * (amp_factor - 1.0)
        diff = self.f_(diff)
        diff = self.res(diff)

        return m_a + diff


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_text_repr = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1),
            nn.ReLU()
        )

        self.conv_common = nn.Sequential(
            ResBlock(64, 64, 9),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
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

    def forward(self, text_repr, shape_repr):
        text_repr = self.conv_text_repr(text_repr)
        x = torch.cat([text_repr, shape_repr], 1)
        x = self.conv_common(x)

        return x


