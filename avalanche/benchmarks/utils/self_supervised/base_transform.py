import random
import torch
from torchvision import transforms
from PIL import ImageFilter, ImageOps, Image


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


class BaseTransformation:
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

    def build_pipeline(self, config: dict) -> transforms.Compose:
        """
        Build a transformation pipeline from a config dictionary.
        Each step is added only if its corresponding key exists in the config.

        :param config: Dictionary with keys for each transformation.
        :return: A torchvision.transforms.Compose pipeline.
        """
        pipeline = []

        if 'crop_scale' in config:
            pipeline.append(transforms.RandomResizedCrop(
                self.size,
                scale=config['crop_scale'],
                interpolation=config.get('interpolation', Image.BICUBIC)
            ))

        if 'color_jitter_params' in config and 'color_jitter_p' in config:
            pipeline.append(transforms.RandomApply(
                [transforms.ColorJitter(*config['color_jitter_params'])],
                p=config['color_jitter_p']
            ))

        if 'grayscale_p' in config:
            pipeline.append(transforms.RandomGrayscale(
                p=config['grayscale_p']
            ))

        if 'gaussian_blur_p' in config:
            pipeline.append(transforms.RandomApply(
                [self.GaussianBlur(config.get('gaussian_blur_sigma', [0.1, 2.0]))],
                p=config['gaussian_blur_p']
            ))

        if 'solarization_p' in config:
            pipeline.append(self.Solarization(p=config['solarization_p']))

        if 'horizontal_flip_p' in config:
            pipeline.append(transforms.RandomHorizontalFlip(
                p=config['horizontal_flip_p']
            ))


        pipeline.append(transforms.ToTensor())
        pipeline.append(self.normalize)

        return transforms.Compose(pipeline)
