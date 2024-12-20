import argparse
import torch
from os.path import expanduser
import sys
from os.path import abspath, dirname
import os
sys.path.insert(0, abspath(dirname(__file__) + "/../.."))
from avalanche.benchmarks import SplitCIFAR10
from avalanche.benchmarks.utils.self_supervised.barlow_transform import BarlowTwinsTransform
from avalanche.benchmarks.utils.self_supervised.simclr_transform import SimCLRTransform
from avalanche.benchmarks.utils.self_supervised.simsiam_transform import SimSiamTransform
from avalanche.models.self_supervised import SimSiam, SimCLR, BarlowTwins
from avalanche.models.self_supervised.backbones.cifar_resnet18 import ModelBase
from avalanche.training.self_supervised.strategy_wrappers import SelfNaive
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.self_supervised_losses import SimSiamLoss, BarlowTwinsLoss, NTXentLoss
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Select SSL method
    if args.ssl_method == "simsiam":
        model = SimSiam(backbone=ModelBase(feature_dim=128, arch="resnet18", bn_splits=8))
        transform = SimSiamTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = SimSiamLoss()
    elif args.ssl_method == "barlow":
        model = BarlowTwins(backbone=ModelBase(feature_dim=128, arch="resnet18", bn_splits=8))
        transform = BarlowTwinsTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = BarlowTwinsLoss()
    elif args.ssl_method == "simclr":
        model = SimCLR(backbone=ModelBase(feature_dim=128, arch="resnet18", bn_splits=8))
        transform = SimCLRTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32)
        loss_fn = NTXentLoss()
    else:
        raise ValueError(f"Unsupported SSL method: {args.ssl_method}")

    # Benchmark
    benchmark = SplitCIFAR10(
        n_experiences=1,
        dataset_root=expanduser("~") + "/.avalanche/data/cifar10/",
        train_transform=transform,
        eval_transform=transform,
        seed=1234,
    )

    # Metrics and logging
    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]

    if args.log_file:
        text_logger = TextLogger(open(args.log_file, 'a'))
        loggers.append(text_logger)

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=False, epoch=True, experience=True),
        loggers=loggers,
    )

    # Optimizer and strategy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06, weight_decay=5e-4, momentum=0.9)
    strategy = SelfNaive(
        model,
        optimizer,
        loss_fn,
        train_epochs=args.epochs,
        device=device,
        plugins=[
            LRSchedulerPlugin(
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                step_granularity="epoch",
            ),
        ],
        train_mb_size=512,
        evaluator=eval_plugin,
    )

    # Train
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience, num_workers=1)
        strategy.eval(benchmark.test_stream[:])

    # Save weights
    save_path = args.save_path
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_method", type=str, default="simsiam",
                        help="Self-supervised learning method: simsiam, barlow, or simclr.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model weights.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional: Path to a .txt file to save text logs. If not provided, text logging is disabled.")
    args = parser.parse_args()
    main(args)
