import torch.nn as nn

class SimSiamLoss(nn.Module):
    def __init__(self, criterion=nn.CosineSimilarity()):
        super().__init__()
        self.criterion = criterion

    def forward(self, z1, z2, p1, p2):
        p1 = p1.detach()
        p2 = p2.detach()
        return 0.5 * (self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean())