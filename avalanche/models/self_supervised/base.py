import copy
import torch
from torch import nn
from typing import Optional, Dict, List

"""
This module defines base classes for self-supervised learning (SSL) models.
It includes:
  - `SelfSupervisedModel`: A base SSL model.
  - `SelfSupervisedMomentumModel`: An SSL model with both an online and a momentum-updated target network,
    inspired by methods such as MoCo and BYOL.
Both classes include an optional online classifier that may be used if labels are provided in the dataset.
"""
class SelfSupervisedModel(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            num_classes: Optional[int] = None
    ):
        """
        Base class for self-supervised models with an optional online classifier.

        Args:
            backbone (nn.Module): The backbone neural network to extract features.
            num_classes (Optional[int]): Number of classes for classification.
                                         If provided, a classifier is instantiated.
        """
        super(SelfSupervisedModel, self).__init__()
        self.backbone = backbone

        if num_classes is not None:
            if not hasattr(backbone, 'feature_dim'):
                raise AttributeError("Backbone must have an attribute `feature_dim` indicating the feature dimension.")
            self.classifier = nn.Linear(backbone.feature_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        x, x1, x2 = torch.unbind(x, dim=1)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        out = {"f": [f1, f2]}

        if self.classifier is not None:
            with torch.no_grad():
                f = self.backbone(x.detach())
            out["logits"] = self.classifier(f)

        return out

class SelfSupervisedMomentumModel(nn.Module):
    """
    A base class to handle an online and a target network
    with a momentum update for the target (like MoCo and BYOL). Includes an optional classifier.
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: Optional[int] = None,
        momentum: float = 0.999
    ):
        super().__init__()
        self.backbone = backbone
        self.target_backbone = copy.deepcopy(backbone)
        self.momentum = momentum

        if num_classes is not None:
            if not hasattr(backbone, 'feature_dim'):
                raise AttributeError(
                    "Online backbone must have an attribute `feature_dim` indicating the feature dimension."
                )
            self.classifier = nn.Linear(backbone.feature_dim, num_classes)
        else:
            self.classifier = None

        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        for param_t in self.target_backbone.parameters():
            param_t.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """
        Update target network parameters with exponential moving average of online network.
        """
        for param_o, param_t in zip(
            self.backbone.parameters(), self.target_backbone.parameters()
        ):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        x, x1, x2 = torch.unbind(x, dim=1)

        f_online1 = self.backbone(x1)
        f_online2 = self.backbone(x2)
        with torch.no_grad():
            f_target1 = self.target_backbone(x1)
            f_target2 = self.target_backbone(x2)

        out = {"f_online": [f_online1, f_online2], "f_target": [f_target1, f_target2]}

        if self.classifier is not None:
            with torch.no_grad():
                f = self.backbone(x.detach())
            out["logits"] = self.classifier(f)

        return out




