import torch
from mpl_toolkits.mplot3d.proj3d import transform
from torchvision import transforms

from avalanche.benchmarks.utils.self_supervised.base_transform import BaseTransformation


class CIFARTransformation(BaseTransformation):
    """CIFAR-specific (weaker) data augmentation pipeline."""

    def __init__(self, mean, std, size, config=None):
        super().__init__(mean, std, size)
        if config is None:
            # Define default parameters for CIFAR.
            config = {
                'crop_scale': (0.2, 1.0),
                'horizontal_flip_p': 0.5,
                'color_jitter_params': (0.4, 0.4, 0.4, 0.1),
                'color_jitter_p': 0.8,
                'grayscale_p': 0.2,
                # No blur or solarization for CIFAR
            }
        self.augmentation = self.build_pipeline(config)

    def __call__(self, x):
        """Normalized clean view and two randomly augmented views."""
        return torch.stack([
            self.normalize(self.to_tensor(x)),
            self.augmentation(x),
            self.augmentation(x)
        ], dim=0)

class SimCLRTransformation(BaseTransformation):
    """Symmetric data augmentation pipeline from SimCLR."""

    def __init__(self, mean, std, size, config=None):
        super().__init__(mean, std, size)
        if config is None:
            # Define default parameters for SimCLR.
            config = {
                'crop_scale': (0.2, 1.0),
                'horizontal_flip_p': 0.5,
                'color_jitter_params': (0.8, 0.8, 0.8, 0.2),
                'color_jitter_p': 0.8,
                'grayscale_p': 0.2,
                'gaussian_blur_sigma': [0.1, 2.0],
                # 'gaussian_blur_p': 0.5,
                # No solarization
            }
        self.augmentation = self.build_pipeline(config)

    def __call__(self, x):
        return torch.stack([
            self.normalize(self.to_tensor(x)),
            self.augmentation(x),
            self.augmentation(x)
        ], dim=0)


class SimSiamTransformation(BaseTransformation):
    """SimSiam-specific data augmentation pipeline with configurable options."""

    def __init__(self, mean, std, size, config=None):
        super().__init__(mean, std, size)

        if config is None:
            config = {
                'crop_scale': (0.2, 1.0),
                'horizontal_flip_p': 0.5,
                'color_jitter_params': (0.4, 0.4, 0.4, 0.1),
                'color_jitter_p': 0.8,
                'grayscale_p': 0.2,
                'gaussian_blur_sigma': [0.1, 2.0],
                'gaussian_blur_p': 0.5,
            }

        self.augmentation = self.build_pipeline(config)

    def __call__(self, x):
        """Normalized clean view and two randomly augmented views."""
        return torch.stack([
            self.normalize(self.to_tensor(x)),
            self.augmentation(x),
            self.augmentation(x)
        ], dim=0)

class AsymmetricTransformation(BaseTransformation):
    """
    Asymmetric data augmentation pipeline used in the original version of
    Barlow Twins and BYOL.
    """
    def __init__(self, mean, std, size, base_config=None, solarization_config=None):
        super().__init__(mean, std, size)
        if base_config is None:
            base_config = {
                'crop_scale': (0.2, 1.0),
                'horizontal_flip_p': 0.5,
                'color_jitter_params': (0.4, 0.4, 0.2, 0.1),
                'color_jitter_p': 0.8,
                'grayscale_p': 0.2,
                'gaussian_blur_sigma': [0.1, 2.0],
                'gaussian_blur_p': 0.0,
                # No solarization.
            }
        if solarization_config is None:
            # For the second branch, enable solarization.
            solarization_config = {'solarization_p': 0.2}
        self.augmentation = self.build_pipeline(base_config)
        # Merge base and solarization configurations for the second view.
        config_prime = {**base_config, **solarization_config}
        self.augmentation_prime = self.build_pipeline(config_prime)

    def __call__(self, x):
        return torch.stack([
            self.normalize(self.to_tensor(x)),
            self.augmentation(x),
            self.augmentation_prime(x)
        ], dim=0)

class AsymmetricTransformationImageNet(BaseTransformation):
    """
    Asymmetric data augmentation pipeline for ImageNet images.
    """
    def __init__(self, mean, std, size, base_config=None):
        super().__init__(mean, std, size)
        if base_config is None:
            base_config = {
                'crop_scale': (0.08, 1.0),
                'horizontal_flip_p': 0.5,
                'color_jitter_params': (0.4, 0.4, 0.2, 0.1),
                'color_jitter_p': 0.8,
                'grayscale_p': 0.2,
                'gaussian_blur_sigma': [0.1, 2.0],
                'gaussian_blur_p': 1.0,
                # No solarization.
            }
        config_prime = {
            'crop_scale': (0.08, 1.0),
            'horizontal_flip_p': 0.5,
            'color_jitter_params': (0.4, 0.4, 0.2, 0.1),
            'color_jitter_p': 0.8,
            'grayscale_p': 0.2,
            'gaussian_blur_sigma': [0.1, 2.0],
            'gaussian_blur_p': 0.1,
            'solarization': 0.2
        }

        self.augmentation = self.build_pipeline(base_config)
        self.augmentation_prime = self.build_pipeline(config_prime)

        self.default = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, x):
        return torch.stack([
            self.default(x),
            self.augmentation(x),
            self.augmentation_prime(x)
        ], dim=0)