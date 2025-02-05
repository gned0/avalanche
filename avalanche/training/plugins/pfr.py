from torch import nn
from avalanche.training.plugins.self_distillation import SelfDistillationPlugin


class PFRPlugin(SelfDistillationPlugin):
    """
    Plugin implementation of the Projected Functional Regularization approach,
    introduced in https://arxiv.org/pdf/2112.15022.
    """

    def __init__(self, output_dim: int):
        super().__init__(
            distillation_loss=nn.CosineSimilarity(dim=1),
            output_dim=output_dim,
            hidden_dim=output_dim // 2,
        )
        self.counter = 0
        self.ssl_loss_accum = 0.0
        self.distillation_loss_accum = 0.0

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

        ssl_loss = strategy.loss.item() if strategy.loss is not None else 0.0
        distill_loss = additional_term.item()

        self.ssl_loss_accum += ssl_loss
        self.distillation_loss_accum += distill_loss
        self.counter += 1

        # Check if counter reaches 40
        if self.counter == 40:
            avg_ssl_loss = self.ssl_loss_accum / 40
            avg_distillation_loss = self.distillation_loss_accum / 40
            print(f"Average SSL Loss: {avg_ssl_loss}, Average Distillation Loss: {avg_distillation_loss}")

            # Reset counter and accumulators
            self.counter = 0
            self.ssl_loss_accum = 0.0
            self.distillation_loss_accum = 0.0

        print(f"SSL Loss: {ssl_loss}, Distill Loss: {distill_loss}")
        print(f"Accumulated SSL Loss: {self.ssl_loss_accum}, Accumulated Distill Loss: {self.distillation_loss_accum}")

        strategy.loss += additional_term
