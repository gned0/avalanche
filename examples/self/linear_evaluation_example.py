import torch
from os.path import expanduser

from scipy.spatial.distance import seuclidean
from torch import nn
from torchvision import models

from avalanche.benchmarks import SplitCIFAR10
from avalanche.models.self_supervised import SimSiam, SimSiamLoader
from avalanche.training import Naive

from avalanche.evaluation.metrics import (
    loss_metrics, accuracy_metrics,
)
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.logging import InteractiveLogger
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.self_supervised_losses import SimSiamLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Self-supervised pre-training

    self_supervised_model = SimSiam()
    loader = SimSiamLoader((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 62)

    # create the benchmark
    benchmark = SplitCIFAR10(
        n_experiences=1, dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
        train_transform=loader,
        eval_transform=loader
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.01)
    criterion = SimSiamLoss()

    # create strategy
    strategy = SelfNaive(
        self_supervised_model,
        optimizer,
        criterion,
        train_epochs=10,
        device=device,
        train_mb_size=256,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])

    # Linear evaluation

    pre_trained_backbone = self_supervised_model.backbone

    # Create a linear classifier
    classifier = nn.Linear(512, 10)  # 512 is the feature size of ResNet18

    # Combine backbone and classifier
    model = nn.Sequential(pre_trained_backbone, classifier).to(device)

    # create the benchmark
    benchmark = SplitCIFAR10(
        n_experiences=1, dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # create strategy
    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_epochs=20,
        device=device,
        train_mb_size=256,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])


if __name__ == "__main__":
    main()
