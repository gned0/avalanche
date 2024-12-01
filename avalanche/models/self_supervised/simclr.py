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
        return self.projector(self.backbone(x))

