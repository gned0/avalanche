################################################################################
# Copyright (c) 2024 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 07-01-2024                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List

from avalanche.evaluation import Metric, GenericPluginMetric


class LearningRateMetric(Metric[float]):
    """
    Learning Rate Metric.

    Instances of this metric keep track of the learning rate
    during training.

    Each time `result` is called, this metric emits the current learning rate.
    """

    def __init__(self):
        """
        Creates an instance of the learning rate metric.
        """
        self._current_lr = 0.0

    def update(self, optimizer) -> None:
        """Update the learning rate.

        :param optimizer: The optimizer used in training.
        :return: None.
        """
        for param_group in optimizer.param_groups:
            self._current_lr = param_group['lr']
            break

    def result(self) -> float:
        """Returns the current learning rate.

        :return: The current learning rate, as a float.
        """
        return self._current_lr

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._current_lr = 0.0


class LearningRatePluginMetric(GenericPluginMetric[float, LearningRateMetric]):
    def __init__(self, reset_at, emit_at, mode):
        self._lr = LearningRateMetric()
        super(LearningRatePluginMetric, self).__init__(self._lr, reset_at, emit_at, mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._lr.update(strategy.optimizer)


class MinibatchLearningRate(LearningRatePluginMetric):
    """
    The minibatch learning rate metric.
    This plugin metric only works at training time.

    This metric logs the learning rate after each iteration.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchLearningRate metric.
        """
        super(MinibatchLearningRate, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "LearningRate_MB"


class EpochLearningRate(LearningRatePluginMetric):
    """
    The learning rate at the end of each epoch.
    This plugin metric only works at training time.

    The learning rate will be logged after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochLearningRate metric.
        """
        super(EpochLearningRate, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "LearningRate_Epoch"


class ExperienceLearningRate(LearningRatePluginMetric):
    """
    At the end of each experience, this metric reports
    the learning rate.
    This plugin metric only works at training time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceLearningRate metric
        """
        super(ExperienceLearningRate, self).__init__(
            reset_at="experience", emit_at="experience", mode="train"
        )

    def __str__(self):
        return "LearningRate_Exp"


def learning_rate_metrics(
    *, minibatch=False, epoch=False, experience=False
) -> List[LearningRatePluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics for learning rate.

    :param minibatch: If True, will return a metric able to log
        the minibatch learning rate at training time.
    :param epoch: If True, will return a metric able to log
        the epoch learning rate at training time.
    :param experience: If True, will return a metric able to log
        the learning rate on each training experience.

    :return: A list of plugin metrics.
    """

    metrics: List[LearningRatePluginMetric] = []
    if minibatch:
        metrics.append(MinibatchLearningRate())

    if epoch:
        metrics.append(EpochLearningRate())

    if experience:
        metrics.append(ExperienceLearningRate())

    return metrics


__all__ = [
    "LearningRateMetric",
    "MinibatchLearningRate",
    "EpochLearningRate",
    "ExperienceLearningRate",
    "learning_rate_metrics",
]
