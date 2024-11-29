import random

import torch
from PIL import ImageFilter
from torch import nn
from torchvision import models, transforms
import torch.nn.functional as F

class SimCLR(nn.Module):
    def  __init__(self,
                 proj_hidden_dim: int = 2048,
                 proj_output_dim: int = 2048,
                 ):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # remove classification head

        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=1)
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        return F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

class SimCLRLoader:
    def __init__(self, mean, std, size):
        """
        Initialize the SimCLR loader with configurable normalization parameters and image size.

        :param mean: Mean value for normalization
        :param std: Standard deviation value for normalization
        :param size: Target size for image resizing
        """
        self.mean = mean
        self.std = std
        self.size = size

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(self.size, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([self.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ])

        self.transform = self.TwoCropsTransform(self.augmentation)

    class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""
        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return torch.stack([q, k], dim=0)

    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    def __call__(self, x):
        """
        Apply the transformation to an image and return the augmented version.
        Returns two augmented versions of the image (query and key).
        """
        return self.transform(x)
