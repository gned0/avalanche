from functools import partial
import torch
from torchvision.models import resnet
from torch import nn

# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting along the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py


class SplitBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, num_splits, **kw):

        super().__init__(num_features, **kw)

        self.num_splits = num_splits

    def forward(self, input):

        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:

            running_mean_split = self.running_mean.repeat(self.num_splits)

            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(N, C, H, W)

            self.running_mean.data.copy_(
                running_mean_split.view(self.num_splits, C).mean(dim=0)
            )

            self.running_var.data.copy_(
                running_var_split.view(self.num_splits, C).mean(dim=0)
            )

            return outcome

        else:

            return nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )


class ModelBase(nn.Module):
    """

    Common CIFAR ResNet recipe.

    Comparing with ImageNet ResNet recipe, it:

    (i) replaces conv1 with kernel=3, str=1

    (ii) removes pool1

    """

    def __init__(self, feature_dim=128, arch=None, bn_splits=16):

        super(ModelBase, self).__init__()

        # use split batchnorm

        norm_layer = (
            partial(SplitBatchNorm, num_splits=bn_splits)
            if bn_splits > 1
            else nn.BatchNorm2d
        )

        resnet_arch = getattr(resnet, arch)

        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []

        for name, module in net.named_children():

            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )

            if isinstance(module, nn.MaxPool2d):
                continue

            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))

            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):

        return self.net(x)


class ProbeModelBase(nn.Module):
    """
    A model that takes a ModelBase and makes only the last layer trainable,
    keeping the backbone frozen.
    """

    def __init__(self, model_base: ModelBase):
        super().__init__()

        backbone_layers = []
        fc_layers = []
        for name, module in model_base.net.named_children():

            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )

            if isinstance(module, nn.MaxPool2d):
                continue

            if isinstance(module, nn.Linear):
                fc_layers.append(nn.Flatten(1))
                fc_layers.append(module)
                continue

            backbone_layers.append(module)

        self.backbone = [
            nn.Sequential(*backbone_layers),
        ]

        self.fc = nn.Sequential(*fc_layers)

        self.backbone[0].eval()
        for name, param in self.backbone[0].named_parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone[0](x)
        x = self.fc(x)
        return x

    def to(self, device):
        self.backbone[0].to(device)
        self.fc.to(device)
        return self


__all__ = ["ModelBase", "ProbeModelBase"]
