import torch
from torch import nn
from typing import Optional, Dict, List
from avalanche.models.self_supervised.base import SelfSupervisedModel

"""

This module implements the SimCLR self-supervised learning model, introduced in https://arxiv.org/pdf/2002.05709.
The SimCLR model employs a backbone network to extract features and
uses a projector (a two-layer MLP with a ReLU activation in between) to map
these features into a lower-dimensional embedding space where contrastive loss
can be applied. 
"""
class SimCLR(SelfSupervisedModel):
    def __init__(
        self,
        backbone: nn.Module,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 256,
        num_classes: Optional[int] = None
    ):
        """
        Args:
            backbone (nn.Module): The feature extractor network. Must have a `feature_dim` attribute.
            proj_hidden_dim (int): Dimensionality of the hidden layers in the projector. Default is 2048.
            proj_output_dim (int): Dimensionality of the output layer of the projector. Default is 256.
            num_classes (Optional[int]): If provided, an additional online classifier is instantiated.
        """
        if not hasattr(backbone, 'feature_dim'):
            raise AttributeError("Backbone must have an attribute `feature_dim` indicating the feature dimension.")

        projector_in_dim = backbone.feature_dim

        super().__init__(backbone=backbone, num_classes=num_classes)

        # Define the projector
        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass through the SimCLR model.

        Args:
            x (torch.Tensor): The superclass
                              method unpacks this into at least two augmented views.

        Returns:
            Dict[str, List[torch.Tensor]]: A dictionary containing:
                - "f": List of feature tensors obtained from the backbone.
                - "z": List of projected tensors after passing the features through the projector.
        """
        out = super().forward(x)
        f1, f2 = out["f"]

        z1 = self.projector(f1)
        z2 = self.projector(f2)
        out["z"] = [z1, z2]

        return out