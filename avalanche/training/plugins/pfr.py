from torch import nn
from avalanche.training.plugins.self_distillation import SelfDistillationPlugin


class PFRPlugin(SelfDistillationPlugin):
    """
    Plugin implementation of the Projected Functional Regularization approach,
    introduced in https://arxiv.org/pdf/2112.15022.
    """
    def __init__(self, output_dim: int = 128, distill_barlow_lamb: float = 0.005):
        super().__init__(distillation_loss=nn.CosineSimilarity(dim=1),
                         output_dim=output_dim,
                         hidden_dim=output_dim // 2)
        self.distill_barlow_lamb = distill_barlow_lamb

    def before_backward(self, strategy, **kwargs):
        if self.distiller is None:
            return
        z1, z2 = strategy.mb_output[1].unbind(dim=1)
        z1_frozen, z2_frozen = self.frozen_forward(strategy.mb_x)[1].unbind(dim=1)

        # Compute additional loss
        p1 = self.distill_predictor(z1)
        p2 = self.distill_predictor(z2)
        additional_term = -(self.distillation_loss(p1, z1_frozen.detach()).mean()
                            + self.distillation_loss(p2, z2_frozen.detach()).mean()) * 0.5
        strategy.loss += additional_term