import sys
import argparse
import torch
from os.path import expanduser
from os.path import abspath, dirname
import os
sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.benchmarks import SplitCIFAR10
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase
from avalanche.training import Naive
from avalanche.evaluation.metrics import (
    loss_metrics, accuracy_metrics, forgetting_metrics, confusion_matrix_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch import nn

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model weights
    state_dict = torch.load(args.weights_path, map_location=device)
    classifier = ModelBase(feature_dim=10, arch="resnet18", bn_splits=8)

    # Freeze all layers except the last fc
    for name, param in classifier.named_parameters():
        if name not in ["net.9.weight", "net.9.bias"]:
            param.requires_grad = False

    new_state_dict = {k: v for k, v in state_dict.items() if "net.9" not in k}
    msg = classifier.load_state_dict(new_state_dict, strict=False)
    assert set(msg.missing_keys) == {"net.9.weight", "net.9.bias"}

    # Benchmark
    benchmark = SplitCIFAR10(
        n_experiences=1,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
        seed=1234,
    )

    # Metrics and logging
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(experience=True, stream=True),
        accuracy_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=10, save_image=False, stream=True),
        loggers=[interactive_logger],
    )

    # Optimizer and strategy
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=30.0, momentum=0.9, weight_decay=0
    )
    criterion = nn.CrossEntropyLoss()
    strategy = Naive(
        classifier,
        optimizer,
        criterion,
        train_epochs=args.epochs,
        device=device,
        train_mb_size=512,
        evaluator=eval_plugin,
    )

    # Train and evaluate
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    args = parser.parse_args()
    main(args)