import random

import torch
from PIL import ImageFilter, ImageOps, Image
from torch import nn
from torchvision import models, transforms


class BarlowTwins(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 projector_in_dim: int = 128,
                 proj_hidden_dim: int = 2048,
                 proj_output_dim: int = 2048):
        super().__init__()

        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=False),
        )


    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=1)
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        return z1, z2

