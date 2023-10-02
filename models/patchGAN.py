from torch import nn

from models.blurpool import BlurPool
from .fran import add_age_channel


class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels, out_channels=64, n_layers=3,
                 padding_mode='zeros'):
        """Construct a PatchGAN discriminator

        Parameters:
            in_channels (int)  -- the number of channels in input images
            out_channels (int)  -- the number of filters in the last conv
                layer
            n_layers (int)  -- the number of conv layers in the discriminator
            padding_mode (str) -- the padding mode in Conv layers
        """
        super().__init__()

        sequence = [
            BlurPool(channels=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode=padding_mode),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                BlurPool(channels=out_channels * nf_mult_prev),
                nn.Conv2d(out_channels * nf_mult_prev,
                          out_channels * nf_mult,
                          kernel_size=3, padding=1, bias=False,
                          padding_mode=padding_mode),
                nn.BatchNorm2d(out_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult,
                      kernel_size=3, padding=1, bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(out_channels * nf_mult, 1, kernel_size=3,
                               padding=1, padding_mode=padding_mode)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x, age):
        x = add_age_channel(x, age)
        return self.model(x)
