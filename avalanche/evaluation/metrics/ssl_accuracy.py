from typing import List
from .accuracy import (
    AccuracyPluginMetric,
    AccuracyPerTaskPluginMetric,
    MinibatchAccuracy,
    EpochAccuracy,
    RunningEpochAccuracy,
    ExperienceAccuracy,
    StreamAccuracy,
    TrainedExperienceAccuracy,
)

class SSLAccuracyPluginMetric(AccuracyPluginMetric):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

class SSLAccuracyPerTaskPluginMetric(AccuracyPerTaskPluginMetric):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y, strategy.mb_task_id)

class SSLMinibatchAccuracy(MinibatchAccuracy):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

class SSLEpochAccuracy(EpochAccuracy):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

class SSLRunningEpochAccuracy(RunningEpochAccuracy):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

class SSLExperienceAccuracy(ExperienceAccuracy):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

class SSLStreamAccuracy(StreamAccuracy):
    def update(self, strategy):
        self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

class SSLTrainedExperienceAccuracy(TrainedExperienceAccuracy):
    def update(self, strategy):
        if strategy.experience.current_experience <= self._current_experience:
            self._metric.update(strategy.mb_output["logits"], strategy.mb_y)

def ssl_accuracy_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[AccuracyPluginMetric]:
    metrics: List[AccuracyPluginMetric] = []
    if minibatch:
        metrics.append(SSLMinibatchAccuracy())
    if epoch:
        metrics.append(SSLEpochAccuracy())
    if epoch_running:
        metrics.append(SSLRunningEpochAccuracy())
    if experience:
        metrics.append(SSLExperienceAccuracy())
    if stream:
        metrics.append(SSLStreamAccuracy())
    if trained_experience:
        metrics.append(SSLTrainedExperienceAccuracy())
    return metrics

__all__ = [
    "SSLAccuracyPluginMetric",
    "SSLAccuracyPerTaskPluginMetric",
    "SSLMinibatchAccuracy",
    "SSLEpochAccuracy",
    "SSLRunningEpochAccuracy",
    "SSLExperienceAccuracy",
    "SSLStreamAccuracy",
    "SSLTrainedExperienceAccuracy",
    "ssl_accuracy_metrics",
]
