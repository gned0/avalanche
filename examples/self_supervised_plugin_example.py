import torch
from os.path import expanduser
from torch import nn

from avalanche.benchmarks import SplitCIFAR10
from avalanche.training.self_supervised.strategy_wrappers.self_naive import SelfNaive

"""
Example on how to use a self-supervised model with experience replay.
"""

from avalanche.evaluation.metrics import (
    loss_metrics,
)
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.logging import InteractiveLogger
from avalanche.training.self_supervised_losses import SimSiamLoss
from avalanche.models.self_supervised import SimSiam, SimSiamLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    self_supervised_model = SimSiam()
    loader = SimSiamLoader((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 62)


    # create the benchmark
    benchmark = SplitCIFAR10(
        n_experiences=5, dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
        train_transform=loader,
        eval_transform=loader
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
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
        plugins=[ReplayPlugin()],
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
