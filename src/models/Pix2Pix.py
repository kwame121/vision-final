"""Combined generator/discriminator wrapper for pix2pix training."""

import torch
from torch import nn

from models import PatchGAN, Unet
from config import Config


class Pix2Pix(nn.Module):
    """Takes paired (x, y) and returns tensors needed for D and G losses."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.discriminator = PatchGAN.PatchGAN(in_channels=3, out_channels=1)
        self.generator = Unet.Unet(
            in_channels=3,
            out_channels=3,
            use_skip_connections=cfg.train.use_skip_connections,
        )

    def get_param_groups(self):
        """Return disjoint parameter groups for D and G optimizers."""
        return self.discriminator.parameters(), self.generator.parameters()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # G(x): satellite -> roadmap
        fake = self.generator(x)
        # D losses need detached fake; G adversarial loss needs non-detached fake.
        fake_d_detached = self.discriminator(x, fake.detach())
        fake_d_for_g = self.discriminator(x, fake)
        real_d = self.discriminator(x, y)
        return {
            "fake": fake,
            "fake_d_detached": fake_d_detached,
            "fake_d_for_g": fake_d_for_g,
            "real_d": real_d,
        }
