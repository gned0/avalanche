import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def criterion(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, p, z):

        p1, p2 = torch.unbind(p, dim=0)
        z1, z2 = torch.unbind(z, dim=0)

        loss1 = self.criterion(p1, z2)
        loss2 = self.criterion(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=5e-3, device='cuda'):
        super().__init__()
        self.lambd = lambd
        self.device = device

    def forward(self, z1, z2):
        z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD

        N = z1.size(0)
        D = z1.size(1)

        # cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / N # DxD
        c_diff = (c - torch.eye(D, device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambd
        loss = c_diff.sum()

        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        indexes = torch.arange(z.size(0)).cuda()
        half = indexes[:z.size(0) // 2]
        indexes = half.repeat(2)

        z = F.normalize(z, dim=-1)

        sim = torch.exp(torch.einsum("if, jf -> ij", z, z) / self.temperature)

        pos_mask = indexes.view(-1, 1) == indexes.view(1, -1)
        neg_mask = ~pos_mask

        pos = torch.sum(sim * pos_mask, dim=1)
        neg = torch.sum(sim * neg_mask, dim=1)

        loss = -torch.mean(torch.log(pos / (pos + neg)))

        return loss
