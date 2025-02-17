import torch
import torch.nn as nn
from typing import Optional, Dict, List

from avalanche.models.self_supervised.base import SelfSupervisedMomentumModel

"""
This module implements the BYOL (Bootstrap Your Own Latent) self-supervised learning model, 
introduced in https://arxiv.org/pdf/2006.07733, based on the SelfSupervisedMomentumModel class. 
BYOL employs an online projector, an online predictor, and a target projector. The
target projector is updated via an exponential moving average (EMA) of the online projector's
parameters.
"""
class BYOL(SelfSupervisedMomentumModel):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 4096,
        out_dim: int = 256,
        num_classes: Optional[int] = None,
        momentum: float = 0.999,
    ):
        """
        Implementation of the BYOL self-supervised learning model.

        Args:
            backbone (nn.Module): The feature extractor network. Must have a `feature_dim` attribute.
            hidden_dim (int): Dimensionality of the hidden layers in the projector. Default is 4096.
            out_dim (int): Dimensionality of the output layer of the projector. Default is 256.
            num_classes (Optional[int]): If provided, an additional online classifier is instantiated.
            momentum (float): Momentum for the target network update. Default is 0.999.
        """
        if not hasattr(backbone, 'feature_dim'):
            raise AttributeError("Backbone must have an attribute `feature_dim` indicating the feature dimension.")

        super().__init__(backbone, num_classes=num_classes, momentum=momentum)
        self.feature_dim = backbone.feature_dim

        self.online_projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

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

        This method performs the following steps:
            1. Uses the base momentum model to extract online and target features from the input.
            2. Processes the online features through the online projector and predictor.
            3. Processes the target features through the target projector (without gradient computation).
            4. Returns a dictionary containing predictions from the online network ("p")
               and target projections ("z").

        Args:
            x (torch.Tensor): Input tensor expected to contain multiple augmented views.
                              The base model handles splitting these into online and target features.

        Returns:
            Dict[str, List[torch.Tensor]]: A dictionary with:
                - "p": List of predictions from the online network for each view.
                - "z": List of target projections (detached) for each view.
        """

        out = super().forward(x)

        f1_online, f2_online = out["f_online"]
        f1_target, f2_target = out["f_target"]

        z1_online = self.online_projector(f1_online)
        z2_online = self.online_projector(f2_online)

        p1_online = self.online_predictor(z1_online)
        p2_online = self.online_predictor(z2_online)

        with torch.no_grad():
            z1_target = self.target_projector(f1_target)
            z2_target = self.target_projector(f2_target)

        out["p"] = [p1_online, p2_online]
        out["z"] = [z1_target.detach(), z2_target.detach()]

        return out
