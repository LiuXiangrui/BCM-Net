import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class FeedForward(nn.Module):
    def __init__(self, channels: int, hidden_ratio: int = 4) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels, out_channels=channels * hidden_ratio, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=channels * hidden_ratio, out_channels=channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class LayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c h w -> b h w c')
        x = rearrange(self.norm(x), 'b h w c -> b c h w')
        return x


class AtrousConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.atrous_convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, dilation=4, padding=4)
        ])

        self.output_conv = nn.Conv2d(in_channels=3 * in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([atrous_conv(x) for atrous_conv in self.atrous_convs], dim=1)
        x = self.output_conv(x)
        return x


class CrissCrossSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.query_map = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.key_map = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.value_map = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.out_map = nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        query = self.query_map(x)
        key = self.key_map(x)
        value = self.value_map(x)

        # query vertical information
        query_ver = rearrange(query, 'b (k c) h w -> (b w) k h c', k=self.num_heads)
        key_ver = rearrange(key, 'b (k c) h w -> (b w) k h c', k=self.num_heads)
        value_ver = rearrange(value, 'b (k c) h w -> (b w) k h c', k=self.num_heads)
        attn_ver = torch.softmax(torch.einsum("bkhc, bklc -> bkhl", query_ver, key_ver), dim=-1)
        out_ver = rearrange(torch.einsum("bkhl, bklc -> bkhc", attn_ver, value_ver),  '(b w) k h c -> b (k c) h w', w=W)

        # query horizontal information
        query_hor = rearrange(query, 'b (k c) h w -> (b h) k w c', k=self.num_heads)
        key_hor = rearrange(key, 'b (k c) h w -> (b h) k w c', k=self.num_heads)
        value_hor = rearrange(value, 'b (k c) h w -> (b h) k w c', k=self.num_heads)
        attn_hor = torch.softmax(torch.einsum("bkwc, bklc -> bkwl", query_hor, key_hor), dim=-1)
        out_hor = rearrange(torch.einsum("bkwl, bklc -> bkwc", attn_hor, value_hor),  '(b h) k w c -> b (k c) h w', h=H)

        # merge vertical and horizontal information
        out = self.out_map(torch.cat([out_ver, out_hor], dim=1))
        return out


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps

        self.query_map = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)

        self.key_map_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.key_map_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)

        self.value_map_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.value_map_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)

        self.out_map = nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, ref_1: torch.Tensor, ref_2: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # generate query
        query = torch.softmax(rearrange(self.query_map(x), 'b (k c) h w -> b k (h w) c ', k=self.num_heads), dim=-1)
        # generate key and value based on the first reference
        key_1 = torch.softmax(rearrange(self.key_map_1(ref_1), 'b (k c) h w -> b k c (h w)', k=self.num_heads), dim=-1)
        value_1 = rearrange(self.value_map_1(ref_1), 'b (k c) h w -> b k (h w) c', k=self.num_heads)
        # query reference from the first reference
        ctx_1 = torch.einsum("bkcn, bknd -> bkcd", key_1, value_1)
        out_1 = rearrange(torch.einsum("bkcd, bknd -> bknc", ctx_1, query), 'b k (h w) c -> b (k c) h w', h=H, w=W)

        # generate key and value based on the second reference
        key_2 = torch.softmax(rearrange(self.key_map_2(ref_2), 'b (k c) h w -> b k c (h w)', k=self.num_heads), dim=-1)
        value_2 = rearrange(self.value_map_2(ref_2), 'b (k c) h w -> b k (h w) c', k=self.num_heads)
        # query reference from the second reference
        ctx_2 = torch.einsum("bkcn, bknd -> bkcd", key_2, value_2)
        out_2 = rearrange(torch.einsum("bkcd, bknd -> bknc", ctx_2, query), 'b k (h w) c -> b (k c) h w', h=H, w=W)

        # merge bidirectional references
        out = self.out_map(torch.cat([out_1, out_2], dim=1))
        return out


class AffineTransform(nn.Module):
    def __init__(self, channels: int, downsampled_size: int = 2, down_sample_out_channels: int = 3) -> None:
        super().__init__()
        self.downsampled_size = downsampled_size
        self.downsample_out_channels = down_sample_out_channels
        self.affine_matrix_height = 2
        self.affine_matrix_width = 3

        self.downsample_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels // 4, out_channels=channels // 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels // 8, out_channels=channels // 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels // 16, out_channels=self.downsample_out_channels, kernel_size=3, stride=2, padding=1)
        )

        self.theta_estimator = nn.Linear(in_features=(self.downsampled_size ** 2) * self.downsample_out_channels,
                                         out_features=self.affine_matrix_height * self.affine_matrix_width)
        # initialize weights of theta estimator
        self.theta_estimator.weight.data.fill_(0.)
        self.theta_estimator.bias = nn.Parameter(torch.Tensor([1., 0., 0., 0., 1., 0.]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> tuple:
        downsampled_x = rearrange(F.interpolate(self.downsample_layers(x), size=(self.downsampled_size, self.downsampled_size)), 'b c h w -> b (c h w)')
        theta = rearrange(self.theta_estimator(downsampled_x), 'b (h w) -> b h w', h=self.affine_matrix_height, w=self.affine_matrix_width)  # 2 x 3 matrix
        inv_theta = torch.cat([theta[:, :, :-1].permute(0, 2, 1), -1 * theta[:, :, -1:]], dim=-1)
        affine_grid = F.affine_grid(theta, size=x.shape, align_corners=False)
        inv_affine_grid = F.affine_grid(inv_theta, size=x.shape, align_corners=False)
        return affine_grid, inv_affine_grid
