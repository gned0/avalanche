import torch.nn
from avalanche.models.base_model import BaseModel

"""Things that can be generalized such as image transformation and backbones are here
'soldered' to the model. This is not a good practice, but it is a simple way to prototype
a self-supervised model and test its integration with Avalanche"""

class SimSiam(torch.nn.Module):
    def __init__(self, hidden_size: int = 2048, projection_size: int = 256):
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

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, projection_size),
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(projection_size, projection_size),
            torch.nn.ReLU(),
            torch.nn.Linear(projection_size, projection_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return z, z, p, p
