from torch import nn


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

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, out_channels, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(out_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(out_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(out_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
