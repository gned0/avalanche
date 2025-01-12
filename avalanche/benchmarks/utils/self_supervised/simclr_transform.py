from torchvision import transforms

from avalanche.benchmarks.utils.self_supervised.base_transform import (
    ContrastiveTransform,
)


class SimCLRTransform(ContrastiveTransform):
    """SimCLR-specific data augmentation pipeline."""

    def __init__(self, mean, std, size):
        super().__init__(mean, std, size)
        augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([self.GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.transform = self.TwoCropsTransform(augmentation)

    def __call__(self, x):
        return self.transform(x)
