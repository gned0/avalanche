import torch.nn as nn

class SimSiamLoss(nn.Module):
    def __init__(self, criterion=nn.CosineSimilarity(dim=1)):
        super().__init__()
        self.criterion = criterion

    def forward(self, p1, p2, z1, z2):
        # stop gradient on projections to avoid collapse
        z1 = z1.detach()
        z2 = z2.detach()
        return -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5