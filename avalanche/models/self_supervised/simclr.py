import torch
from torch import nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        projector_in_dim: int = 128,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 256,
    ):
        super().__init__()

        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=1)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)

        return {"z": [z1, z2], "f": [f1, f2]}
