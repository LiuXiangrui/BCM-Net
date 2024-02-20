import torch
from torch import nn as nn

from Modules.BasicBlock import ResBlock


class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, R: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(channels=out_channels) for _ in range(R)])
        self.tail = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.tail(x)
        return x


class FeatureExtractionModule(nn.Module):
    def __init__(self, channels_X: int, channels_F: int, R: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels_X, out_channels=channels_F, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            FeatureExtractionBlock(in_channels=channels_F, out_channels=channels_F, R=R, stride=2),
            FeatureExtractionBlock(in_channels=channels_F, out_channels=channels_F, R=R, stride=1),
            nn.Conv2d(in_channels=channels_F, out_channels=channels_F, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        return x if x is None else self.net(x)
