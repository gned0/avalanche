import torch.nn
from torch import nn


class SimSiam(torch.nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        projector_in_dim: int = 512,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 2048,
        pred_hidden_dim: int = 512,
    ):
        super().__init__()

        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=1)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {"f": [f1, f2], "z": [z1, z2], "p": [p1, p2]}
