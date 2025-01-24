import torch
from torch import nn
from typing import Optional, Dict, List
from avalanche.models.self_supervised.self_supervised_model import SelfSupervisedModel

class SimCLR(SelfSupervisedModel):
    def __init__(
        self,
        backbone: nn.Module,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 256,
        num_classes: Optional[int] = None
    ):
        """
        SimCLR model for contrastive self-supervised learning with an optional classifier.

        Args:
            backbone (nn.Module): The backbone neural network.
            projector_hidden_dim (int): Hidden dimension size for the projector.
            projector_output_dim (int): Output dimension size for the projector.
            num_classes (Optional[int]): Number of classes for classification.
                                         If provided, a classifier is instantiated.
        """
        # Ensure backbone has 'feature_dim' attribute
        if not hasattr(backbone, 'feature_dim'):
            raise AttributeError("Backbone must have an attribute `feature_dim` indicating the feature dimension.")

        # Set projector_in_dim based on backbone's feature_dim
        projector_in_dim = backbone.feature_dim

        super().__init__(backbone=backbone, num_classes=num_classes)

        # Define the projector
        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:

        out = super().forward(x)

        # Apply projector to both features
        z1 = self.projector(out["f"][0])
        z2 = self.projector(out["f"][1])

        # Add projections to the output dictionary
        out["z"] = [z1, z2]

        return out
