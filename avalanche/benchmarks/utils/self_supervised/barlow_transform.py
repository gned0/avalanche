import torch
from PIL import Image
from torchvision import transforms

from avalanche.benchmarks.utils.self_supervised.base_transform import (
    ContrastiveTransform,
)


class BarlowTwinsTransform(ContrastiveTransform):
    """Barlow Twins-specific data augmentation pipeline."""

    def __init__(self, mean, std, size):
        super().__init__(mean, std, size)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([self.GaussianBlur([0.1, 2.0])], p=0.0),
                self.Solarization(p=0.0),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([self.GaussianBlur([0.1, 2.0])], p=0.0),
                self.Solarization(p=0.2),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, x):
        return torch.stack([self.transform(x), self.transform_prime(x)], dim=0)
