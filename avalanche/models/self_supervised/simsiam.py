import torch.nn
from torch import nn
from torchvision.transforms import transforms

from avalanche import models
import torchvision.models as models
from avalanche.models.self_supervised import loader

"""Things that can be generalized such as image transformation and backbones are here
'soldered' to the model. This is not a good practice, but it is a simple way to prototype
a self-supervised model and test its integration with Avalanche"""

class SimSiam(torch.nn.Module):
    def __init__(self,
                 input_size: int = 28 * 28,
                 proj_hidden_dim: int = 2048,
                 proj_output_dim: int = 256,
                 pred_hidden_dim: int = 512,
                 ):
        super().__init__()

        """
        Provisional implementation of simsiam, architecture does not follow the original paper yet.
        Pseudo-code of actual version to implement:
         # f: backbone + projection mlp
         # h: prediction mlp
         for x in loader: # load a minibatch x with n samples
         x1, x2 = aug(x), aug(x) # random augmentation
         z1, z2 = f(x1), f(x2) # projections, n-by-d
         p1, p2 = h(z1), h(z2) # predictions, n-by-d
         L = D(p1, z2)/2 + D(p2, z1)/2 # loss
         L.backward() # back-propagate
         update(f, h) # SGD update
         def D(p, z): # negative cosine similarity
            z = z.detach() # stop gradient
            p = normalize(p, dim=1) # l2-normalize
            z = normalize(z, dim=1) # l2-normalize
            return-(p*z).sum(dim=1).mean()
        """
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ])

        self.backbone = models.resnet50(pretrained=True)

        self.backbone.fc = nn.Identity() # remove classification head

        self.projector = nn.Sequential(
            nn.Linear(2048, proj_hidden_dim, bias=False),
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
        x1, x2 = x, x # aug on Tensors to be fixed later

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1, z2

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
