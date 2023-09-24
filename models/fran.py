import torch
from torch import nn

from .blurpool import BlurPool


class FRAN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input consists of RGB + 2 age channels
        self.in_conv = DoubleConv(in_channels=5, out_channels=64)

        self.down1 = DownLayer(in_channels=64, out_channels=128)
        self.down2 = DownLayer(in_channels=128, out_channels=256)
        self.down3 = DownLayer(in_channels=256, out_channels=512)
        self.down4 = DownLayer(in_channels=512, out_channels=1024 // 2)

        self.up1 = UpLayer(in_channels=1024, out_channels=512 // 2)
        self.up2 = UpLayer(in_channels=512, out_channels=256 // 2)
        self.up3 = UpLayer(in_channels=256, out_channels=128 // 2)
        self.up4 = UpLayer(in_channels=128, out_channels=64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x0, src_age, tgt_age):
        if src_age.ndim == 1:
            x0_with_age = add_age_channel(x0, src_age)
        else:
            x0_with_age = torch.cat([x0, src_age / 100], dim=-3)

        if tgt_age.ndim == 1:
            x0_with_age = add_age_channel(x0_with_age, tgt_age)
        else:
            x0_with_age = torch.cat([x0_with_age, tgt_age / 100], dim=-3)

        x1 = self.in_conv(x0_with_age)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)

        return x + x0


def add_age_channel(img, age):
    age_channel = age[:, None, None, None] * torch.ones_like(img)[..., :1, :, :] / 100
    return torch.cat([img, age_channel], dim=-3)


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            BlurPool(channels=in_channels),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # Bias not necessary due to affine parameter in BatchNorm
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.double_conv(x)
