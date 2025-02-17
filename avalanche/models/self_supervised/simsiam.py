import torch
import torch.nn as nn
from typing import Optional, Dict, List
from avalanche.models.self_supervised.base import SelfSupervisedModel

"""
This module implements the SimSiam self-supervised learning model, introduced in https://arxiv.org/pdf/2011.10566.
SimSiam uses a backbone network to extract features from augmented inputs and employs a projector and predictor network 
to learn meaningful representations without requiring negative samples.
"""

class SimSiam(SelfSupervisedModel):
    def __init__(
        self,
        backbone: nn.Module,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 2048,
        pred_hidden_dim: int = 512,
        num_classes: Optional[int] = None
    ):
        """
        Implementation of the SimSiam self-supervised learning model.

        Args:
            backbone (nn.Module): The feature extractor network. Must have a `feature_dim` attribute.
            proj_hidden_dim (int): Dimensionality of the hidden layers in the projector. Default is 2048.
            proj_output_dim (int): Dimensionality of the output layer of the projector. Default is 2048.
            pred_hidden_dim (int): Dimensionality of the hidden layer in the predictor. Default is 512.
            num_classes (Optional[int]): If provided, an additional online classifier is instantiated.
        """
        if not hasattr(backbone, 'feature_dim'):
            raise AttributeError("Backbone must have an attribute `feature_dim` indicating the feature dimension.")

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
        """
        Forward pass through the SimSiam model.

        Args:
            x (torch.Tensor): Input tensor expected to have two views. The superclass
                              method unpacks this into at least two augmented views.

        Returns:
            Dict[str, List[torch.Tensor]]: A dictionary containing:
                - "f": List of feature tensors extracted by the backbone.
                - "z": List of projected tensors after applying the projector.
                - "p": List of prediction tensors after applying the predictor.
        """
        out = super().forward(x)
        f1, f2 = out["f"]

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        out["z"] = [z1, z2]
        out["p"] = [p1, p2]
        return out