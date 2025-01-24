import torch
from torch import nn
from typing import Optional, Dict, List


class SelfSupervisedModel(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            num_classes: Optional[int] = None
    ):
        """
        Base class for self-supervised models with an optional classifier.

        Args:
            backbone (nn.Module): The backbone neural network to extract features.
            num_classes (Optional[int]): Number of classes for classification.
                                         If provided, a classifier is instantiated.
        """
        super(SelfSupervisedModel, self).__init__()
        self.backbone = backbone

        # Initialize classifier only if num_classes is provided
        if num_classes is not None:
            if not hasattr(backbone, 'feature_dim'):
                raise AttributeError("Backbone must have an attribute `feature_dim` indicating the feature dimension.")
            self.classifier = nn.Linear(backbone.feature_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        x1, x2 = torch.unbind(x, dim=1)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        output = {"f": [f1, f2]}

        if self.classifier is not None:
            logits = self.classifier(f1.detach())
            output["l"] = [logits]

        return output


