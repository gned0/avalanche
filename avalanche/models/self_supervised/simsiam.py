import torch
import torch.nn as nn
from typing import Optional, Dict, List
from avalanche.models.self_supervised.base import SelfSupervisedModel

class SimSiam(SelfSupervisedModel):
    def __init__(
        self,
        backbone: nn.Module,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 2048,
        pred_hidden_dim: int = 512,
        num_classes: Optional[int] = None
    ):
        projector_in_dim = backbone.feature_dim
        super().__init__(backbone=backbone, num_classes=num_classes)

        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        out = super().forward(x)
        f1, f2 = out["f"]

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        out["z"] = [z1, z2]
        out["p"] = [p1, p2]
        return out