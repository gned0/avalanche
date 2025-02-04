import torch
from torch import nn
from torchvision.models import resnet18, resnet50


class ResNet(nn.Module):
    """
    Base model with a ResNet backbone.

    Args:
        feature_dim (int): The dimension used for the classification head. This is
                           passed to the model's `num_classes` parameter.
        cifar (bool): If True, a ResNet-18 is created and modified for CIFAR.
                      Otherwise, a ResNet-50 is instantiated.
    """

    def __init__(self, feature_dim: int, cifar: bool = False):
        super().__init__()
        self.cifar = cifar

        if cifar:
            self.model = resnet18(num_classes=feature_dim)
            self.model.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.model.maxpool = nn.Identity()
        else:
            self.model = resnet50(num_classes=feature_dim)

        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ProbeResNet(nn.Module):
    """
    Linear probe model on top of a frozen backbone.

    This module wraps a ModelBase instance (the backbone), freezes its parameters,
    and attaches a trainable linear classifier (probe) on top.

    Args:
        model_base (ModelBase): The backbone model.
        backbone_feature_dim (int): The dimension of the backbone's output features.
                                    (E.g., 512 for ResNet-18 after modifications,
                                    2048 for ResNet-50.)
        num_classes (int): The number of output classes for the linear classifier.
    """

    def __init__(self, model_base: ResNet, backbone_feature_dim: int, num_classes: int = 100):
        super().__init__()
        self.backbone = model_base

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(backbone_feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.linear(features)

    def to(self, device):
        self.backbone.to(device)
        self.linear.to(device)
        return self


__all__ = ["ResNet", "ProbeResNet"]