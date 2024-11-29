import random

import torch
from PIL import ImageFilter, ImageOps, Image
from torch import nn
from torchvision import models, transforms


class BarlowTwins(nn.Module):
    def __init__(self, proj_hidden_dim: int = 2048, proj_output_dim: int = 2048):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(proj_output_dim, affine=False)

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # remove classification head

        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=False),
        )


    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=1)

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        return z1, z2

class BarlowTwinsLoader:
    def __init__(self, mean, std, size):
        """
        Initialize the Barlow Twins loader with configurable normalization parameters and image size.

        :param mean: Mean value for normalization
        :param std: Standard deviation value for normalization
        :param size: Target size for image resizing
        """
        self.mean = mean
        self.std = std
        self.size = size

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            self.GaussianBlur(p=1.0),
            self.Solarization(p=0.0),
            transforms.ToTensor(),
            self.normalize
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(self.size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            self.GaussianBlur(p=0.1),
            self.Solarization(p=0.2),
            transforms.ToTensor(),
            self.normalize
        ])

    class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return torch.stack([q, k], dim=0)

    class GaussianBlur(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, img):
            if random.random() < self.p:
                sigma = random.random() * 1.9 + 0.1
                return img.filter(ImageFilter.GaussianBlur(sigma))
            else:
                return img

    class Solarization(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, img):
            if random.random() < self.p:
                return ImageOps.solarize(img)
            else:
                return img

    def __call__(self, x):
        """
        Apply the transformation to an image and return the augmented version.
        Returns two augmented versions of the image (T and T').
        """
        return torch.stack([self.transform(x), self.transform_prime(x)], dim=0)