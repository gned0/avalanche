import random
import torch
from torchvision import transforms
from PIL import ImageFilter, ImageOps


class ContrastiveTransform:
    """Base class for SSL data augmentation pipelines."""

    def __init__(self, mean, std, size):
        """
        Initialize common properties for data augmentation pipelines.

        :param mean: Mean value for normalization.
        :param std: Standard deviation for normalization.
        :param size: Target size for image resizing.
        """
        self.mean = mean
        self.std = std
        self.size = size
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.to_tensor = transforms.ToTensor()

    class GaussianBlur:
        """Gaussian blur augmentation."""

        def __init__(self, sigma=[0.1, 2.0]):
            self.sigma = sigma

        def __call__(self, img):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    class Solarization:
        """Solarization augmentation."""

        def __init__(self, p):
            self.p = p

        def __call__(self, img):
            if random.random() < self.p:
                return ImageOps.solarize(img)
            return img

    class TwoCropsTransform:
        """Apply two random augmentations to the same image."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return torch.stack([q, k], dim=0)
