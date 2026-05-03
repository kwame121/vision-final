"""U-Net generator for pix2pix"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """One stride-2 block: spatial halve, double channels. No pool + double strided stack."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        p = 1
        s = 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, s, p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upconv (2× spatial), concat skip (same H×W), then two 3×3 stride-1 convs → out_channels.
    `in_from_below` = channels from deeper layer; after ConvTranspose, tensor has out_channels;
    then concat with skip: out_channels + skip_channels → out_channels.
    """

    def __init__(
        self,
        in_from_below: int,
        out_channels: int,
        skip_channels: int,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        p = 1
        s = 2
        self.up = nn.ConvTranspose2d(in_from_below, out_channels, kernel_size, s, p, bias=False)
        # concat depth = out (from up) + skip
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutputBlock(nn.Module):
    """1×1 conv to RGB (or N channels), then tanh for targets in [-1, 1]."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.conv(x))


class Unet(nn.Module):
    """
    4 down / 4 up. Channel schedule 64-128-256-512-256-128-64-64 → out via 1×1.
    Input spatial size should be a multiple of 16 (e.g. 256) for clean alignment; odd sizes
    are handled with bilinear resize on skip path when needed.
    """

    def __init__(self, in_channels: int, out_channels: int, ngf: int = 64, use_skip_connections: bool = True) -> None:
        super().__init__()
        self.use_skip_connections = use_skip_connections
        c1, c2, c3, c4 = ngf, ngf * 2, ngf * 4, ngf * 8
        k = 4
        self.down1 = DownBlock(in_channels, c1, k)
        self.down2 = DownBlock(c1, c2, k)
        self.down3 = DownBlock(c2, c3, k)
        self.down4 = DownBlock(c3, c4, k)
        self.up1 = UpBlock(c4, c3, c3, k)  # 512 → 256, skip 256
        self.up2 = UpBlock(c3, c2, c2, k)  # 256 → 128, skip 128
        self.up3 = UpBlock(c2, c1, c1, k)  # 128 → 64,  skip 64
        self.up4 = UpBlock(c1, c1, in_channels, k)  # 64 → 64,  skip in_channels
        self.output_block = OutputBlock(c1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        if self.use_skip_connections:
            s1, s2, s3, s4 = d3, d2, d1, x
        else:
            s1, s2, s3, s4 = torch.zeros_like(d3), torch.zeros_like(d2), torch.zeros_like(d1), torch.zeros_like(x)
        u1 = self.up1(d4, s1)
        u2 = self.up2(u1, s2)
        u3 = self.up3(u2, s3)
        u4 = self.up4(u3, s4)
        return self.output_block(u4)
