import torch
import torch.nn as nn
from typing import Optional, Dict, List

from avalanche.models.self_supervised.base import SelfSupervisedMomentumModel


class BYOL(SelfSupervisedMomentumModel):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 4096,
        out_dim: int = 256,
        num_classes: Optional[int] = None,
        momentum: float = 0.999,
    ):
        super().__init__(backbone, num_classes=num_classes, momentum=momentum)
        self.feature_dim = backbone.feature_dim
        # Online Projector
        self.online_projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Online Predictor
        self.online_predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Target Projector
        self.target_projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_target_projector()

    @torch.no_grad()
    def _init_target_projector(self):
        """
        Initialize target projector to match online projector.
        """
        for param_o, param_t in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def momentum_update_projector(self):
        """
        Update target projector parameters with EMA of online projector parameters.
        """
        for param_o, param_t in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    @torch.no_grad()
    def momentum_update_all(self):
        """
        Update all target parameters (backbone + projector).
        """
        self.momentum_update()
        self.momentum_update_projector()

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass for BYOL.

        Args:
            x (torch.Tensor): Input batch with shape (B, 2, C, H, W).

        Returns:
            Dict[str, List[torch.Tensor]]: {
                "p": [p1, p2],          # Predictions from the online network
                "z": [z1_target, z2_target],  # Projections from the target network (no grad)
                "logits": logits (optional)   # Classifier logits if enabled
            }
        """
        (x1, x2) = torch.unbind(x, dim=1)

        # online network
        f1_online = self.backbone(x1)
        f2_online = self.backbone(x2)

        z1_online = self.online_projector(f1_online)
        z2_online = self.online_projector(f2_online)

        p1_online = self.online_predictor(z1_online)
        p2_online = self.online_predictor(z2_online)

        # target network
        with torch.no_grad():
            f1_target = self.target_backbone(x1)
            f2_target = self.target_backbone(x2)

            z1_target = self.target_projector(f1_target)
            z2_target = self.target_projector(f2_target)

        out = {
            "p": [p1_online, p2_online],
            "z": [z1_target.detach(), z2_target.detach()],
        }

        # If classifier exists, add logits
        if self.classifier is not None:
            f = 0.5 * (f1_online.detach() + f2_online.detach())
            out["logits"] = self.classifier(f)

        return out
