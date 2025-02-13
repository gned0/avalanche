import torch
from typing import Tuple
from torchvision import transforms
from avalanche.core import SelfSupervisedPlugin
from avalanche.training import ReservoirSamplingBuffer
import numpy as np


class LUMPPlugin(SelfSupervisedPlugin):
    """

    """

    def __init__(self, buffer_size: int, device: torch.device, transform, alpha: float = 0.1,):
        super().__init__()
        self.alpha = alpha
        self.buffer = ReservoirBuffer(buffer_size, device)
        self.transform = transform

    def before_training_iteration(self, strategy, **kwargs):
        if self.buffer.is_empty():
            # first experience, mixup not applied, no need to change
            # the dataloader.
            return
        else:
            buf_inputs1, buf_inputs2 = self.buffer.get_data(
                strategy.mb_x.shape[0]
            )  # get stored inputs by random sampling from buffer
            inputs1, inputs2 = torch.unbind(strategy.mb_x, dim=1)
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

            # Perform self-supervised mixup
            mixed_inputs_1 = lam * inputs1 + (1 - lam) * buf_inputs1
            mixed_inputs_2 = lam * inputs2 + (1 - lam) * buf_inputs2
            mixed_inputs = torch.stack([mixed_inputs_1, mixed_inputs_2], dim=1)
            strategy.mbatch = (mixed_inputs, *strategy.mbatch[1:])

    def after_training_iteration(self, strategy: "SelfSupervisedTemplate", **kwargs):
        inputs1, inputs2 = torch.unbind(strategy.mb_x, dim=1)
        self.buffer.add_data(examples=inputs1, labels=inputs2)


class ReservoirBuffer:
    """
    A reservoir sampling buffer for self-supervised learning (SSL) tasks.
    Stores pairs of inputs (inputs1, inputs2) for SSL tasks.
    """

    def __init__(self, buffer_size: int, device: torch.device):
        """
        Initialize the reservoir buffer.

        :param buffer_size: Maximum size of the buffer.
        :param device: Device on which the buffer tensors are stored.
        """
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0  # Tracks the number of examples seen so far

        # Initialize buffer tensors (None initially, will be initialized on the first call to add_data)
        self.inputs1 = None
        self.inputs2 = None

    def _init_tensors(self, examples: torch.Tensor, labels: torch.Tensor):
        """
        Initialize the buffer tensors with the same shape as the input tensors.

        :param examples: Tensor representing inputs1 (shape: [batch_size, ...]).
        :param labels: Tensor representing inputs2 (shape: [batch_size, ...]).
        """
        self.inputs1 = torch.zeros(
            (self.buffer_size, *examples.shape[1:]),
            dtype=examples.dtype,
            device=self.device,
        )
        self.inputs2 = torch.zeros(
            (self.buffer_size, *labels.shape[1:]),
            dtype=labels.dtype,
            device=self.device,
        )

    def add_data(self, examples: torch.Tensor, labels: torch.Tensor):
        """
        Add new data to the reservoir buffer using reservoir sampling.

        :param examples: Tensor representing inputs1.
        :param labels: Tensor representing inputs2.
        """
        if self.inputs1 is None or self.inputs2 is None:
            # Initialize buffer tensors on the first call
            self._init_tensors(examples, labels)

        batch_size = examples.shape[0]

        for i in range(batch_size):
            # Determine the index to store this example in the buffer
            index = self._reservoir_sampling_index(self.num_seen_examples)
            self.num_seen_examples += 1  # Increment the counter for seen examples

            if index >= 0:
                # Replace the example at the chosen index
                self.inputs1[index] = examples[i]
                self.inputs2[index] = labels[i]

    def _reservoir_sampling_index(self, num_seen_examples: int) -> int:
        """
        Reservoir sampling algorithm to determine the index for a new example.

        :param num_seen_examples: Number of examples seen so far.
        :return: The index in the buffer where the example should be stored, or -1 if not stored.
        """
        if num_seen_examples < self.buffer_size:
            return num_seen_examples  # Fill the buffer initially
        else:
            rand = np.random.randint(0, num_seen_examples + 1)
            if rand < self.buffer_size:
                return rand  # Replace an existing example
            else:
                return -1  # Do not store this example

    def get_data(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a random sample of data from the buffer.

        :param size: Number of items to retrieve.
        :return: A tuple (inputs1, inputs2) of the sampled data.
        """
        if self.inputs1 is None or self.inputs2 is None or self.num_seen_examples == 0:
            raise ValueError("Buffer is empty or not initialized!")

        # Determine the number of available examples in the buffer
        available_size = min(self.num_seen_examples, self.buffer_size)

        # Adjust the requested size if it exceeds the available size
        size = min(size, available_size)

        # Randomly sample indices from the buffer
        indices = np.random.choice(available_size, size=size, replace=False)

        return self.inputs1[indices], self.inputs2[indices]

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.
        """
        return self.num_seen_examples == 0

