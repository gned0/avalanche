from typing import Any

import torch
from torch import nn

from avalanche.core import SelfSupervisedPlugin, Template


class CaSSLePlugin(SelfSupervisedPlugin):
    """
    Plugin implementation of the Contrastive Self-Supervised Learning (CaSSLe)
    approach, introduced in https://arxiv.org/pdf/2112.04215.
    This is essentially a feature distillation approach adapted for self-supervised models.
    """

    def __init__(self, distiller: nn.Module, device: torch.device, loss: nn.Module):
        super().__init__()
        self.device = device
        self.distiller = None # distiller not used until second experience
        self.distillation_loss = loss

        self.distill_predictor = nn.Sequential(
            nn.Linear(output_dim, distill_proj_hidden_dim),
            nn.BatchNorm1d(distill_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(distill_proj_hidden_dim, output_dim),
        )

    def before_training(self, strategy: Template, *args, **kwargs):
        extra_params = [{"params": self.distill_predictor.parameters()}]
        strategy.optimizer.add_param_group(extra_params)

    def before_backward(self, strategy, **kwargs):

        z1, z2 = strategy.mb_output[1].unbind(dim=1)

        z1_frozen, z2_frozen = self.frozen_forward(strategy.mb_x)[1].unbind(dim=1)
        p1 = self.distiller(z1)
        p2 = self.distiller(z2)

        additional_term = (self.distillation_loss(p1, z1_frozen) + self.distillation_loss(p2, z2_frozen)) / 2

        strategy.loss += additional_term

    def after_training_exp(self, strategy: "SelfSupervisedTemplate", **kwargs):
        # update frozen model
        pass

    def frozen_forward(self, x):
        # forward pass through the frozen model
        pass