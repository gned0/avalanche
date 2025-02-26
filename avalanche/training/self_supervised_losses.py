import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamLoss(nn.Module):
    def __init__(self, version="simplified"):
        super().__init__()
        self.ver = version

    def criterion(self, p, z):
        if self.ver == "original":
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == "simplified":
            z = z.detach()  # stop gradient
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, out):

        p1, p2 = out["p"]
        z1, z2 = out["z"]

        loss1 = self.criterion(p1, z2)
        loss2 = self.criterion(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=5e-3, scale_loss=0.025, device="cuda"):
        super().__init__()
        self.lambd = lambd
        self.scale_loss = scale_loss
        self.device = device

    def forward(self, out):
        z1, z2 = out["z"]
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD

        N = z1.size(0)
        D = z1.size(1)

        corr = torch.einsum("bi, bj -> ij", z1_norm, z2_norm) / N

        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        cdif[~diag.bool()] *= self.lambd
        loss = self.scale_loss * cdif.sum()
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, out):
        z1, z2 = out["z"]
        device = z1.device

        b = z1.size(0)
        z = torch.cat((z1, z2), dim=0)
        z = F.normalize(z, dim=-1)

        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask[:, b:].fill_diagonal_(True)
        pos_mask[b:, :].fill_diagonal_(True)

        # all matches excluding the main diagonal
        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()
        return loss

class ContrastiveDistillLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):

        super(ContrastiveDistillLoss, self).__init__()
        self.temperature = temperature

    def forward(
            self,
            p1: torch.Tensor,
            p2: torch.Tensor,
            z1: torch.Tensor,
            z2: torch.Tensor,
    ) -> torch.Tensor:

        device = z1.device
        b = z1.size(0)

        # Normalize and concatenate predictions and representations.
        p = F.normalize(torch.cat([p1, p2]), dim=-1)  # (2*b, feature_dim)
        z = F.normalize(torch.cat([z1, z2]), dim=-1)  # (2*b, feature_dim)

        # Compute similarity logits scaled by temperature
        logits = torch.einsum("if, jf -> ij", p, z) / self.temperature

        # For numerical stability subtract the maximum logit per sample.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask.fill_diagonal_(True)

        logit_mask = torch.ones_like(pos_mask, device=device)
        logit_mask.fill_diagonal_(True)
        logit_mask[:, b:].fill_diagonal_(True)
        logit_mask[b:, :].fill_diagonal_(True)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out) -> torch.Tensor:
        p1, p2 = out["p"]
        z1, z2 = out["z"]

        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Mean Squared Error between p and z'
        loss1 = 2 - 2 * (p1 * z2.detach()).sum(dim=-1)
        loss2 = 2 - 2 * (p2 * z1.detach()).sum(dim=-1)

        return (loss1 + loss2).mean()