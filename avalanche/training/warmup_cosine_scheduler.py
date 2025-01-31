import math
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        """
        Linear Warmup followed by Cosine Annealing Learning Rate Scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for linear warmup.
            total_steps (int): Total number of training steps.
            eta_min (float, optional): Minimum learning rate. Defaults to 0.
            last_epoch (int, optional): The index of last epoch. Defaults to -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * float(step) / float(max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        elif step < self.total_steps:
            # Cosine annealing
            cosine_steps = step - self.warmup_steps
            cosine_total = self.total_steps - self.warmup_steps
            return [
                self.eta_min + (base_lr - self.eta_min) *
                0.5 * (1 + math.cos(math.pi * cosine_steps / cosine_total))
                for base_lr in self.base_lrs
            ]
        else:
            # After total_steps, keep eta_min
            return [self.eta_min for _ in self.base_lrs]
