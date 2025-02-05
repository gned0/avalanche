import torch
from torch import nn
import torch.nn.functional as F
from avalanche.models.self_supervised.simclr import SimCLR
from avalanche.training.plugins.self_distillation import SelfDistillationPlugin
from avalanche.training.self_supervised_losses import BarlowTwinsLoss, ContrastiveDistillLoss


class CaSSLePlugin(SelfDistillationPlugin):
    """
    Plugin implementation of the Contrastive Self-Supervised Learning (CaSSLe)
    approach, introduced in https://arxiv.org/pdf/2112.04215.
    This is essentially a distillation approach adapted for self-supervised models.
    """

    def __init__(self, loss: nn.Module, output_dim: int = 256, hidden_dim: int = 2048):
        super().__init__(
            distillation_loss=loss, output_dim=output_dim, hidden_dim=hidden_dim
        )

    def before_backward(self, strategy, **kwargs):
        if self.frozen_backbone is None:
            return

        additional_term = self.compute_additional_loss(strategy)
        print(f"SSL Loss: {strategy.loss}, Distillation Loss: {additional_term}") 
        #  no hyperparameter to weigh distillation loss with respect to the SSL loss.
        strategy.loss += additional_term

    def compute_additional_loss(self, strategy):
        if isinstance(self.distillation_loss, BarlowTwinsLoss):
            frozen_output = self.frozen_forward(strategy.mb_x)
            z1_frozen, z2_frozen = frozen_output["z_frozen"]

            z1, z2 = strategy.mb_output["z"]
            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            additional_term = (
                self.distillation_loss(
                    {"z": [p1, z1_frozen]}
                )  # loss argument has to be a dictionary
                + self.distillation_loss({"z": [p2, z2_frozen]})
            ) / 2
            return additional_term
        elif isinstance(self.distillation_loss, ContrastiveDistillLoss):
            frozen_output = self.frozen_forward(strategy.mb_x)
            z1_frozen, z2_frozen = frozen_output["z_frozen"]

            z1, z2 = strategy.mb_output["z"]
            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            additional_term = (
                self.distillation_loss(p1, p2, z1_frozen, z2_frozen)
                + self.distillation_loss(z1_frozen, z2_frozen, p1, p2)
            ) / 2
            return additional_term
        else:
            raise ValueError(
                f"Loss: {self.distillation_loss} is incompatible with CaSSLe"
            )
