import torch
from os.path import expanduser

from avalanche.training.self_supervised.strategy_wrappers.self_naive import SelfNaive

"""
A simple example on how to use self-supervised models.
"""

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import InteractiveLogger
from avalanche.training.supervised import FromScratchTraining
from avalanche.training.self_supervised_losses import SimSiamLoss
from avalanche.models.self_supervised import SimSiam

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create the benchmark
    benchmark = SplitMNIST(
        n_experiences=1, dataset_root=expanduser("~") + "/.avalanche/data/mnist/"
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    self_supervised_model = SimSiam(hidden_size=128)
    optimizer = torch.optim.SGD(self_supervised_model.parameters(), lr=0.01)
    criterion = SimSiamLoss()

    # create strategy
    strategy = SelfNaive(
        self_supervised_model,
        optimizer,
        criterion,
        train_epochs=1,
        device=device,
        train_mb_size=32,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])


if __name__ == "__main__":
    main()
