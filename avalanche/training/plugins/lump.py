import torch
from typing import Tuple
from torchvision import transforms
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

    def __init__(self, alpha: float, buffer_size: int, device: torch.device, transform):
        super().__init__()
        self.alpha = alpha
        self.buffer = Buffer(buffer_size, device)
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

            # Perform mixup
            mixed_inputs_1 = lam * inputs1 + (1 - lam) * buf_inputs1
            mixed_inputs_2 = lam * inputs2 + (1 - lam) * buf_inputs2
            mixed_inputs = torch.stack([mixed_inputs_1, mixed_inputs_2], dim=1)
            strategy.mbatch = (mixed_inputs, *strategy.mbatch[1:])

    def after_training_iteration(self, strategy: "SelfSupervisedTemplate", **kwargs):
        inputs1, inputs2 = torch.unbind(strategy.mb_x, dim=1)
        self.buffer.add_data(examples=inputs1, labels=inputs2)


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, mode="reservoir"):
        assert mode in ["ring", "reservoir"]
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.attributes = ["examples", "labels"]

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith("els") else torch.float32
                setattr(
                    self,
                    attr_str,
                    torch.zeros(
                        (self.buffer_size, *attr.shape[1:]),
                        dtype=typ,
                        device=self.device,
                    ),
                )

    def add_data(self, examples, labels):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                self.labels[index] = labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms = None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(
            min(self.num_seen_examples, self.examples.shape[0]),
            size=size,
            replace=False,
        )
        if transform is None:
            transform = lambda x: x
        # import pdb
        # pdb.set_trace()
        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(
                self.device
            ),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            transform = lambda x: x
        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
