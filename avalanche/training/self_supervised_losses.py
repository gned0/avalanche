import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamLoss(nn.Module):
    def __init__(self, criterion=nn.CosineSimilarity(dim=1)):
        super().__init__()
        self.criterion = criterion

    def forward(self, p, z):
        p1, p2 = torch.unbind(p, dim=0)
        z1, z2 = torch.unbind(z, dim=0)
        # stop gradient on projections to avoid collapse
        z1 = z1.detach()
        z2 = z2.detach()
        return -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=5e-3):
        super().__init__()
        self.lambd = lambd

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # cross-correlation matrix
        c = z1.T @ z2 / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        return on_diag + self.lambd * off_diag

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t())  # Shape: (2*batch_size, 2*batch_size)
        sim_matrix = sim_matrix / self.temperature

        # Labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(z.device)

        # Mask to exclude self-similarities
        mask = torch.eye(labels.size(0), dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix[~mask].view(labels.size(0), -1)

        # NT-Xent Loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
