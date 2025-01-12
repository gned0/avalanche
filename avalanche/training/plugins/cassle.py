import torch
from torch import nn
import torch.nn.functional as F
from avalanche.models.self_supervised.simclr import SimCLR
from avalanche.training.plugins.self_distillation import SelfDistillationPlugin
from avalanche.training.self_supervised_losses import BarlowTwinsLoss, NTXentLoss


class CaSSLePlugin(SelfDistillationPlugin):
    """
    Plugin implementation of the Contrastive Self-Supervised Learning (CaSSLe)
    approach, introduced in https://arxiv.org/pdf/2112.04215.
    This is essentially a distillation approach adapted for self-supervised models.
    """

    def __init__(self, loss: nn.Module, output_dim: int = 128, hidden_dim: int = 2048):
        super().__init__(
            distillation_loss=loss, output_dim=output_dim, hidden_dim=hidden_dim
        )

    def before_backward(self, strategy, **kwargs):
        if self.frozen_backbone is None:
            return

        additional_term = self.compute_additional_loss(strategy)
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
        elif isinstance(self.distillation_loss, NTXentLoss):
            frozen_output = self.frozen_forward(strategy.mb_x)
            z1_frozen, z2_frozen = frozen_output["z_frozen"]

            z1, z2 = strategy.mb_output["z"]
            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            additional_term = (
                self.simclr_distill_loss_func(p1, p2, z1_frozen, z2_frozen)
                + self.simclr_distill_loss_func(z1_frozen, z2_frozen, p1, p2)
            ) / 2
            return additional_term
        else:
            raise ValueError(
                f"Loss: {self.distillation_loss} is incompatible with CaSSLe"
            )

    def simclr_distill_loss_func(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:

        device = z1.device

        b = z1.size(0)

        p = F.normalize(torch.cat([p1, p2]), dim=-1)
        z = F.normalize(torch.cat([z1, z2]), dim=-1)

        logits = torch.einsum("if, jf -> ij", p, z) / self.distillation_loss.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask.fill_diagonal_(True)

        # all matches excluding the main diagonal
        logit_mask = torch.ones_like(pos_mask, device=device)
        logit_mask.fill_diagonal_(True)
        logit_mask[:, b:].fill_diagonal_(True)
        logit_mask[b:, :].fill_diagonal_(True)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()
        return loss
