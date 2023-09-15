from torch import nn

from .stylegan2.model import Downsample
from .fran import add_age_channel


class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels, out_channels=64, n_layers=3):
        """Construct a PatchGAN discriminator

        Parameters:
            in_channels (int)  -- the number of channels in input images
            out_channels (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super().__init__()

        sequence = [
            Downsample(kernel=[1, 3, 3, 1], factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Downsample(kernel=[1, 3, 3, 1], factor=2),
                nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(out_channels * nf_mult, 1, kernel_size=3, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, age):
        x = add_age_channel(x, age)
        return self.model(x)
