import torch
from torchvision import transforms

from avalanche.benchmarks.utils.self_supervised.base_transform import (
    ContrastiveTransform,
)


class SimSiamTransform(ContrastiveTransform):
    """SimSiam-specific data augmentation pipeline."""

    def __init__(self, mean, std, size):
        super().__init__(mean, std, size)
        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([self.GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, x):
        return torch.stack([self.normalize(self.to_tensor(x)),
                            self.augmentation(x),
                            self.augmentation(x)], dim=0)
