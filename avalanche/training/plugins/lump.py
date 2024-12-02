import torch

from avalanche.core import SelfSupervisedPlugin
from avalanche.training import ReservoirSamplingBuffer
import numpy as np

class LUMPPlugin(SelfSupervisedPlugin):
    """
    Plugin implementation of the Lifelong Unsupervised Mixup (LUMP) approach,
    introduced in https://arxiv.org/abs/2110.06976.
    This approach aims at enhancing the robustness of learned representation
    in self-supervised scenarios by revisiting the attributes of the past task
    that are similar to the current one (?).
    """
    def __init__(self, alpha: float, mem_size:int):
        super().__init__()
        self.alpha = alpha
        self.storage_policy = ReservoirSamplingBuffer(max_size=mem_size)

    def before_training_iteration(self, strategy, **kwargs):
        if len(self.storage_policy.buffer) == 0:
            # first experience, mixup not applied, no need to change
            # the dataloader.
            return
        buf_inputs_1, buf_inputs_2 = 0, 0 # get stored inputs by random sampling from buffer
        inputs_1 = strategy.mbatch[0]
        inputs_2 = strategy.mbatch[1]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Perform mixup
        mixed_inputs_1 = lam * inputs_1 + (1 - lam) * buf_inputs_1
        mixed_inputs_2 = lam * inputs_2 + (1 - lam) * buf_inputs_2
        strategy.mbatch = (mixed_inputs_1, mixed_inputs_2, *strategy.mbatch[2:])

    def after_training_exp(self, strategy: "SelfSupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs) # ???