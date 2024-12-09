import copy
from typing import Optional

import torch
from torch import nn
from avalanche.core import SelfSupervisedPlugin, Template


class SelfDistillationPlugin(SelfSupervisedPlugin):
    """
    Base class for distillation-based self-supervised plugins.
    """

    def __init__(self,
                 distillation_loss: nn.Module,
                 output_dim: int,
                 hidden_dim: int):
        """
        Initialize the base plugin.
        :param distillation_loss: The loss module for distillation (e.g., CosineSimilarity, MSE).
        :param output_dim: The dimensionality of the output features.
        :param hidden_dim: Hidden layer size for the distillation predictor.
        """
        super().__init__()
        self.distiller: Optional[nn.Module] = None  # Frozen model
        self.distillation_loss = distillation_loss
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.distill_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.predictor_initialized = False

    def before_training(self, strategy, **kwargs):
        if not self.predictor_initialized:
            self.distill_predictor = self.distill_predictor.to(strategy.device)
            # Add predictor's parameters to the optimizer
            extra_params = {"params": self.distill_predictor.parameters()}
            strategy.optimizer.add_param_group(extra_params)
            self.predictor_initialized = True

    def before_backward(self, strategy, **kwargs):
        raise NotImplementedError("Subclasses must implement the before_backward method.")

    def after_training_exp(self, strategy: Template, **kwargs):
        self.distiller = copy.deepcopy(strategy.model)

    @torch.no_grad()
    def frozen_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distiller is None:
            raise RuntimeError("Distiller is not initialized.")
        return self.distiller(x)
