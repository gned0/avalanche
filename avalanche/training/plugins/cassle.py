from torch import nn
from avalanche.training.plugins.self_distillation import SelfDistillationPlugin

class CaSSLePlugin(SelfDistillationPlugin):
    """
    Plugin implementation of the Contrastive Self-Supervised Learning (CaSSLe)
    approach, introduced in https://arxiv.org/pdf/2112.04215.
    This is essentially a distillation approach adapted for self-supervised models.
    """
    def __init__(self, loss: nn.Module, output_dim: int = 128, hidden_dim: int = 2048):
        super().__init__(distillation_loss=loss,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim)

    def before_backward(self, strategy, **kwargs):
        if self.distiller is None:
            return
        z1, z2 = strategy.mb_output[1].unbind(dim=1)
        z1_frozen, z2_frozen = self.frozen_forward(strategy.mb_x)[1].unbind(dim=1)

        # Compute additional loss
        p1 = self.distill_predictor(z1)
        p2 = self.distill_predictor(z2)
        additional_term = (self.distillation_loss(p1, z1_frozen)
                           + self.distillation_loss(p2, z2_frozen)) / 2
        strategy.loss += additional_term

