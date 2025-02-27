import torch
from torch import nn
from torchvision import models
from torchvision.models import resnet18, resnet50


class ResNet(nn.Module):
    """
    ResNet model with a CIFAR variant for smaller inputs.

    Args:
        feature_dim (int): The dimension used for the classification head. This is
                           passed to the model's `num_classes` parameter.
        cifar (bool): If True, a ResNet-18 is created and modified for CIFAR or other smaller inputs.
                      Otherwise, a ResNet-50 is instantiated.
    """

    def __init__(self, architecture = "resnet18" , cifar: bool = False):
        super().__init__()
        self.cifar = cifar
        self.architecture = architecture
        self.feature_dim = 512 if architecture=="resnet18" else 2048

        if architecture == "resnet18":
            self.model = models.resnet18()
        elif architecture == "resnet50":
            self.model = models.resnet50()
        else:
            raise ValueError("Unsupported architecture. Choose 'resnet18' or 'resnet50'.")

        if cifar:
            self.model.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.model.maxpool = nn.Identity()

        self.model.fc = nn.Identity()
        print(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ProbeResNet(nn.Module):
    """
    Linear probe model on top of a frozen backbone.

    This module wraps a ResNet instance (the backbone), freezes its parameters,
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

    def train(self, mode=True):
        """ Override train() to prevent backbone from switching to train mode. """
        super().train(mode)
        self.backbone.eval()  # Always keep backbone in eval mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.linear(features)

    def to(self, device):
        self.backbone.to(device)
        self.linear.to(device)
        return self


__all__ = ["ResNet", "ProbeResNet"]