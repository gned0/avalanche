from torch import nn
from avalanche.training.plugins.self_distillation import SelfDistillationPlugin


class PFRPlugin(SelfDistillationPlugin):
    """
    Plugin implementation of the Projected Functional Regularization approach,
    introduced in https://arxiv.org/pdf/2112.15022.
    """

    def __init__(self, output_dim: int = 128, distill_barlow_lamb: float = 0.005):
        super().__init__(
            distillation_loss=nn.CosineSimilarity(dim=1),
            output_dim=output_dim,
            hidden_dim=output_dim // 2,
        )
        self.distill_barlow_lamb = distill_barlow_lamb

    def before_backward(self, strategy, **kwargs):
        if self.frozen_backbone is None:
            return
        # retrieve the representation from the current model
        f1, f2 = strategy.mb_output["f"]

        # frozen pass: retrieve the representation from the frozen model
        frozen_output = self.frozen_forward(strategy.mb_x)
        f1_frozen, f2_frozen = frozen_output["f_frozen"]

        p1 = self.distill_predictor(f1)
        p2 = self.distill_predictor(f2)

        additional_term = (
            -(
                self.distillation_loss(p1, f1_frozen.detach()).mean()
                + self.distillation_loss(p2, f2_frozen.detach()).mean()
            )
            * 0.5
        )
        print(f"Additional term: {additional_term}")
        strategy.loss += additional_term
