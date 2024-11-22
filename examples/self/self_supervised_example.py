"""
A simple example on how to use self-supervised models.
"""

import torch
from os.path import expanduser
from torch import nn
from avalanche.training.self_supervised.strategy_wrappers.self_naive import SelfNaive
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import InteractiveLogger
from avalanche.training.self_supervised_losses import SimSiamLoss
from avalanche.models.self_supervised import SimSiam, SimSiamLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    self_supervised_model = SimSiam()

    # create the benchmark
    benchmark = SplitMNIST(
        n_experiences=5, dataset_root=expanduser("~") + "/.avalanche/data/mnist/",
        train_transform=SimSiamLoader((0.1307,), (0.3081,), 58),
        eval_transform=SimSiamLoader((0.1307,), (0.3081,), 58)
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    # adapt input layer for MNIST
    self_supervised_model.backbone.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.01)
    criterion = SimSiamLoss()

    # create strategy
    strategy = SelfNaive(
        self_supervised_model,
        optimizer,
        criterion,
        train_epochs=1,
        device=device,
        train_mb_size=128,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])


if __name__ == "__main__":
    main()
