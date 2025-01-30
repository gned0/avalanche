import torch
import torch.nn as nn
from typing import Optional, Dict, List

from avalanche.models.self_supervised import SelfSupervisedModel


class BarlowTwins(SelfSupervisedModel):
    def __init__(
        self,
        backbone: nn.Module,
        projector_in_dim: int,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 2048,
        num_classes: Optional[int] = None
    ):
        super().__init__(backbone=backbone, num_classes=num_classes)

        # Define the projector
        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        out = super().forward(x)
        f1, f2 = out["f"]

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        out["z"] = [z1, z2]
        return out
