import torch
from torch import nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def  __init__(self,
                 backbone: nn.Module,
                 projector_in_dim: int = 128,
                 proj_hidden_dim: int = 2048,
                 proj_output_dim: int = 2048,
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
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        return F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

